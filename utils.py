from __future__ import annotations

import enum
import abc
from dataclasses import dataclass
import array_api_compat.numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from types import ModuleType
from array_api_compat import device, to_device, size

import parallelproj


@dataclass
class TOFParameters:
    """
    generic time of flight (TOF) parameters for a scanner with 385ps FWHM TOF

    num_tofbins: int
        number of time of flight bins
    tofbin_width: float
        width of the TOF bin in spatial units (mm)
    sigma_tof: float
        standard deviation of Gaussian TOF kernel in spatial units (mm)
    num_sigmas: float
        number of sigmas after which TOF kernel is truncated
    tofcenter_offset: float
        offset of center of central TOF bin from LOR center in spatial units (mm)
    """
    num_tofbins: int = 29
    tofbin_width: float = 13 * 0.01302 * 299.792 / 2  # 13 TOF "small" TOF bins of 0.01302[ns] * (speed of light / 2) [mm/ns]
    sigma_tof: float = (299.792 / 2) * (
        0.385 / 2.355)  # (speed_of_light [mm/ns] / 2) * TOF FWHM [ns] / 2.355
    num_sigmas: float = 3.
    tofcenter_offset: float = 0


class SinogramSpatialAxisOrder(enum.Enum):
    """order of spatial axis in a sinogram R (radial), V (view), P (plane)"""

    RVP = enum.auto()
    """[radial,view,plane]"""
    RPV = enum.auto()
    """[radial,plane,view]"""
    VRP = enum.auto()
    """[view,radial,plane]"""
    VPR = enum.auto()
    """[view,plane,radial]"""
    PRV = enum.auto()
    """[plane,radial,view]"""
    PVR = enum.auto()
    """[plane,view,radial]"""


class PETScannerModule(abc.ABC):

    def __init__(
            self,
            xp: ModuleType,
            dev: str,
            num_lor_endpoints: int,
            affine_transformation_matrix: npt.NDArray | None = None) -> None:
        """abstract base class for PET scanner module

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        num_lor_endpoints : int
            number of LOR endpoints in the module
        affine_transformation_matrix : npt.NDArray | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """

        self._xp = xp
        self._dev = dev
        self._num_lor_endpoints = num_lor_endpoints
        self._lor_endpoint_numbers = self.xp.arange(num_lor_endpoints,
                                                    device=self.dev)

        if affine_transformation_matrix is None:
            self._affine_transformation_matrix = self.xp.eye(4,
                                                             device=self.dev)
        else:
            self._affine_transformation_matrix = affine_transformation_matrix

    @property
    def xp(self) -> ModuleType:
        """array module to use for storing the LOR endpoints"""
        return self._xp

    @property
    def dev(self) -> str:
        """device to use for storing the LOR endpoints"""
        return self._dev

    @property
    def num_lor_endpoints(self) -> int:
        """total number of LOR endpoints in the module

        Returns
        -------
        int
        """
        return self._num_lor_endpoints

    @property
    def lor_endpoint_numbers(self) -> npt.NDArray:
        """array enumerating all the LOR endpoints in the module

        Returns
        -------
        npt.NDArray
        """
        return self._lor_endpoint_numbers

    @property
    def affine_transformation_matrix(self) -> npt.NDArray:
        """4x4 affine transformation matrix

        Returns
        -------
        npt.NDArray
        """
        return self._affine_transformation_matrix

    @abc.abstractmethod
    def get_raw_lor_endpoints(self,
                              inds: npt.NDArray | None = None) -> npt.NDArray:
        """mapping from LOR endpoint indices within module to an array of "raw" world coordinates

        Parameters
        ----------
        inds : npt.NDArray | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        npt.NDArray
            a 3 x len(inds) float array with the world coordinates of the LOR endpoints
        """
        if inds is None:
            inds = self.lor_endpoint_numbers
        raise NotImplementedError

    def get_lor_endpoints(self,
                          inds: npt.NDArray | None = None) -> npt.NDArray:
        """mapping from LOR endpoint indices within module to an array of "transformed" world coordinates

        Parameters
        ----------
        inds : npt.NDArray | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        npt.NDArray
            a 3 x len(inds) float array with the world coordinates of the LOR endpoints including an affine transformation
        """

        raw_lor_endpoints = self.get_raw_lor_endpoints(inds)

        tmp = self.xp.ones((raw_lor_endpoints.shape[0], 4), device=self.dev)
        tmp[:, :-1] = raw_lor_endpoints

        return (tmp @ self.affine_transformation_matrix.T)[:, :3]

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           annotation_fontsize: float = 0,
                           annotation_prefix: str = '',
                           annotation_offset: int = 0,
                           transformed: bool = True,
                           **kwargs) -> None:
        """show the LOR coordinates in a 3D scatter plot

        Parameters
        ----------
        ax : plt.Axes
            3D matplotlib axes
        annotation_fontsize : float, optional
            fontsize of LOR endpoint number annotation, by default 0
        annotation_prefix : str, optional
            prefix for annotation, by default ''
        annotation_offset : int, optional
            number to add to crystal number, by default 0
        transformed : bool, optional
            use transformed instead of raw coordinates, by default True
        """

        if transformed:
            all_lor_endpoints = self.get_lor_endpoints()
        else:
            all_lor_endpoints = self.get_raw_lor_endpoints()

        # convert to numpy array
        all_lor_endpoints = np.asarray(to_device(all_lor_endpoints, 'cpu'))

        ax.scatter(all_lor_endpoints[:, 0], all_lor_endpoints[:, 1],
                   all_lor_endpoints[:, 2], **kwargs)

        ax.set_box_aspect([
            ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')
        ])

        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')

        if annotation_fontsize > 0:
            for i in self.lor_endpoint_numbers:
                ax.text(all_lor_endpoints[i, 0],
                        all_lor_endpoints[i, 1],
                        all_lor_endpoints[i, 2],
                        f'{annotation_prefix}{i+annotation_offset}',
                        fontsize=annotation_fontsize)


class RegularPolygonPETScannerModule(PETScannerModule):

    def __init__(
            self,
            xp: ModuleType,
            dev: str,
            radius: float,
            num_sides: int,
            num_lor_endpoints_per_side: int,
            lor_spacing: float,
            ax0: int = 2,
            ax1: int = 1,
            affine_transformation_matrix: npt.NDArray | None = None) -> None:
        """regular Polygon PET scanner module

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        device: str
            device to use for storing the LOR endpoints
        radius : float
            inner radius of the regular polygon
        num_sides: int
            number of sides of the regular polygon
        num_lor_endpoints_per_sides: int
            number of LOR endpoints per side
        lor_spacing : float
            spacing between the LOR endpoints in the polygon direction
        ax0 : int, optional
            axis number for the first direction, by default 2
        ax1 : int, optional
            axis number for the second direction, by default 1
        affine_transformation_matrix : npt.NDArray | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """

        self._radius = radius
        self._num_sides = num_sides
        self._num_lor_endpoints_per_side = num_lor_endpoints_per_side
        self._ax0 = ax0
        self._ax1 = ax1
        self._lor_spacing = lor_spacing
        super().__init__(xp, dev, num_sides * num_lor_endpoints_per_side,
                         affine_transformation_matrix)

    @property
    def radius(self) -> float:
        """inner radius of the regular polygon

        Returns
        -------
        float
        """
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides of the regular polygon

        Returns
        -------
        int
        """
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side

        Returns
        -------
        int
        """
        return self._num_lor_endpoints_per_side

    @property
    def ax0(self) -> int:
        """axis number for the first module direction

        Returns
        -------
        int
        """
        return self._ax0

    @property
    def ax1(self) -> int:
        """axis number for the second module direction

        Returns
        -------
        int
        """
        return self._ax1

    @property
    def lor_spacing(self) -> float:
        """spacing between the LOR endpoints in a module along the polygon

        Returns
        -------
        float
        """
        return self._lor_spacing

    # abstract method from base class to be implemented
    def get_raw_lor_endpoints(self,
                              inds: npt.NDArray | None = None) -> npt.NDArray:
        if inds is None:
            inds = self.lor_endpoint_numbers

        side = inds // self.num_lor_endpoints_per_side
        tmp = inds - side * self.num_lor_endpoints_per_side
        tmp = self.xp.astype(
            tmp, float) - (self.num_lor_endpoints_per_side / 2 - 0.5)

        phi = 2 * self.xp.pi * self.xp.astype(side, float) / self.num_sides

        lor_endpoints = self.xp.zeros((self.num_lor_endpoints, 3),
                                      device=self.dev)
        lor_endpoints[:, self.ax0] = self.xp.cos(
            phi) * self.radius - self.xp.sin(phi) * self.lor_spacing * tmp
        lor_endpoints[:, self.ax1] = self.xp.sin(
            phi) * self.radius + self.xp.cos(phi) * self.lor_spacing * tmp

        return lor_endpoints


class ModularizedPETScannerGeometry:
    """description of a PET scanner geometry consisting of LOR endpoint modules"""

    def __init__(self, modules: tuple[PETScannerModule]):
        """
        Parameters
        ----------
        modules : tuple[PETScannerModule]
            a tuple of scanner modules
        """

        # member variable that determines whether we want to use
        # a numpy or cupy array to store the array of all lor endpoints
        self._modules = modules
        self._num_modules = len(self._modules)
        self._num_lor_endpoints_per_module = self.xp.asarray(
            [x.num_lor_endpoints for x in self._modules], device=self.dev)
        self._num_lor_endpoints = int(
            self.xp.sum(self._num_lor_endpoints_per_module))

        self.setup_all_lor_endpoints()

    def setup_all_lor_endpoints(self) -> None:
        """calculate the position of all lor endpoints by iterating over
           the modules and calculating the transformed coordinates of all
           module endpoints
        """

        self._all_lor_endpoints_index_offset = self.xp.asarray([
            int(sum(self._num_lor_endpoints_per_module[:i]))
            for i in range(size(self._num_lor_endpoints_per_module))
        ],
                                                               device=self.dev)

        self._all_lor_endpoints = self.xp.zeros((self._num_lor_endpoints, 3),
                                                device=self.dev,
                                                dtype=self.xp.float32)

        for i, module in enumerate(self._modules):
            self._all_lor_endpoints[
                int(self._all_lor_endpoints_index_offset[i]):int(
                    self._all_lor_endpoints_index_offset[i] +
                    module.num_lor_endpoints), :] = module.get_lor_endpoints()

        self._all_lor_endpoints_module_number = [
            int(self._num_lor_endpoints_per_module[i]) * [i]
            for i in range(self._num_modules)
        ]

        self._all_lor_endpoints_module_number = self.xp.asarray(
            [i for r in self._all_lor_endpoints_module_number for i in r],
            device=self.dev)

    @property
    def modules(self) -> tuple[PETScannerModule]:
        """tuple of modules defining the scanner"""
        return self._modules

    @property
    def num_modules(self) -> int:
        """the number of modules defining the scanner"""
        return self._num_modules

    @property
    def num_lor_endpoints_per_module(self) -> npt.NDArray:
        """numpy array showing how many LOR endpoints are in every module"""
        return self._num_lor_endpoints_per_module

    @property
    def num_lor_endpoints(self) -> int:
        """the total number of LOR endpoints in the scanner"""
        return self._num_lor_endpoints

    @property
    def all_lor_endpoints_index_offset(self) -> npt.NDArray:
        """the offset in the linear (flattend) index for all LOR endpoints"""
        return self._all_lor_endpoints_index_offset

    @property
    def all_lor_endpoints_module_number(self) -> npt.NDArray:
        """the module number of all LOR endpoints"""
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints(self) -> npt.NDArray:
        """the world coordinates of all LOR endpoints"""
        return self._all_lor_endpoints

    @property
    def xp(self) -> ModuleType:
        """module indicating whether the LOR endpoints are stored as numpy or cupy array"""
        return self._modules[0].xp

    @property
    def dev(self) -> str:
        return self._modules[0].dev

    def linear_lor_endpoint_index(
        self,
        module: npt.NDArray,
        index_in_module: npt.NDArray,
    ) -> npt.NDArray:
        """transform the module + index_in_modules indices into a flattened / linear LOR endpoint index

        Parameters
        ----------
        module : npt.NDArray
            containing module numbers
        index_in_module : npt.NDArray
            containing index in modules

        Returns
        -------
        npt.NDArray
            the flattened LOR endpoint index
        """
        #    index_in_module = self._xp.asarray(index_in_module)

        return self.xp.take(self.all_lor_endpoints_index_offset,
                            module) + index_in_module

    def get_lor_endpoints(self, module: npt.NDArray,
                          index_in_module: npt.NDArray) -> npt.NDArray:
        """get the coordinates for LOR endpoints defined by module and index in module

        Parameters
        ----------
        module : npt.NDArray
            the module number of the LOR endpoints
        index_in_module : npt.NDArray
            the index in module number of the LOR endpoints

        Returns
        -------
        npt.NDArray | cpt.NDArray
            the 3 world coordinates of the LOR endpoints
        """
        return self.xp.take(self.all_lor_endpoints,
                            self.linear_lor_endpoint_index(
                                module, index_in_module),
                            axis=0)

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           show_linear_index: bool = True,
                           **kwargs) -> None:
        """show all LOR endpoints in a 3D plot

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        show_linear_index : bool, optional
            annotate the LOR endpoints with the linear LOR endpoint index
        **kwargs : keyword arguments
            passed to show_lor_endpoints() of the scanner module
        """
        for i, module in enumerate(self.modules):
            if show_linear_index:
                offset = np.asarray(
                    to_device(self.all_lor_endpoints_index_offset[i], 'cpu'))
                prefix = f''
            else:
                offset = 0
                prefix = f'{i},'

            module.show_lor_endpoints(ax,
                                      annotation_offset=offset,
                                      annotation_prefix=prefix,
                                      **kwargs)


class RegularPolygonPETScannerGeometry(ModularizedPETScannerGeometry):
    """description of a PET scanner geometry consisting stacked regular polygons"""

    def __init__(self, xp: ModuleType, dev: str, radius: float, num_sides: int,
                 num_lor_endpoints_per_side: int, lor_spacing: float,
                 num_rings: int, ring_positions: npt.NDArray,
                 symmetry_axis: int) -> None:
        """
        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        radius : float
            radius of the scanner
        num_sides : int
            number of sides (faces) of each regular polygon
        num_lor_endpoints_per_side : int
            number of LOR endpoints in each side (face) of each polygon
        lor_spacing : float
            spacing between the LOR endpoints in each side
        num_rings : int
            the number of rings (regular polygons)
        ring_positions : npt.NDArray
            1D array with the coordinate of the rings along the ring axis
        symmetry_axis : int
            the ring axis (0,1,2)
        """

        self._radius = radius
        self._num_sides = num_sides
        self._num_lor_endpoints_per_side = num_lor_endpoints_per_side
        self._num_rings = num_rings
        self._lor_spacing = lor_spacing
        self._symmetry_axis = symmetry_axis
        self._ring_positions = ring_positions

        if symmetry_axis == 0:
            self._ax0 = 2
            self._ax1 = 1
        elif symmetry_axis == 1:
            self._ax0 = 0
            self._ax1 = 2
        elif symmetry_axis == 2:
            self._ax0 = 1
            self._ax1 = 0

        modules = []

        for ring in range(num_rings):
            aff_mat = xp.eye(4, device=dev)
            aff_mat[symmetry_axis, -1] = ring_positions[ring]

            modules.append(
                RegularPolygonPETScannerModule(
                    xp,
                    dev,
                    radius,
                    num_sides,
                    num_lor_endpoints_per_side=num_lor_endpoints_per_side,
                    lor_spacing=lor_spacing,
                    affine_transformation_matrix=aff_mat,
                    ax0=self._ax0,
                    ax1=self._ax1))

        modules = tuple(modules)
        super().__init__(modules)

        self._all_lor_endpoints_index_in_ring = self.xp.arange(
            self.num_lor_endpoints, device=dev
        ) - self.all_lor_endpoints_ring_number * self.num_lor_endpoints_per_module[
            0]

    @property
    def radius(self) -> float:
        """radius of the scanner"""
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides (faces) of each polygon"""
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side (face) in each polygon"""
        return self._num_lor_endpoints_per_side

    @property
    def num_rings(self) -> int:
        """number of rings (regular polygons)"""
        return self._num_rings

    @property
    def lor_spacing(self) -> float:
        """the spacing between the LOR endpoints in every side (face) of each polygon"""
        return self._lor_spacing

    @property
    def symmetry_axis(self) -> int:
        """The symmetry axis. Also called axial (or ring) direction."""
        return self._symmetry_axis

    @property
    def all_lor_endpoints_ring_number(self) -> npt.NDArray:
        """the ring (regular polygon) number of all LOR endpoints"""
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints_index_in_ring(self) -> npt.NDArray:
        """the index withing the ring (regular polygon) number of all LOR endpoints"""
        return self._all_lor_endpoints_index_in_ring

    @property
    def num_lor_endpoints_per_ring(self) -> int:
        """the number of LOR endpoints per ring (regular polygon)"""
        return int(self._num_lor_endpoints_per_module[0])

    @property
    def ring_positions(self) -> npt.NDArray:
        """the ring (regular polygon) positions"""
        return self._ring_positions


class DemoPETScanner(RegularPolygonPETScannerGeometry):

    def __init__(self,
                 xp: ModuleType,
                 dev: str,
                 num_rings: int = 36,
                 symmetry_axis: int = 2) -> None:

        ring_positions = 5.32 * xp.arange(
            num_rings, device=dev, dtype=xp.float32) + (xp.astype(
                xp.arange(num_rings, device=dev) // 9, xp.float32)) * 2.8
        ring_positions -= 0.5 * xp.max(ring_positions)
        super().__init__(xp,
                         dev,
                         radius=0.5 * (744.1 + 2 * 8.51),
                         num_sides=34,
                         num_lor_endpoints_per_side=16,
                         lor_spacing=4.03125,
                         num_rings=num_rings,
                         ring_positions=ring_positions,
                         symmetry_axis=symmetry_axis)


class PETLORDescriptor(abc.ABC):
    """abstract base class to describe which modules / indices in modules of a 
       modularized PET scanner are in coincidence; defining geometrical LORs"""

    def __init__(self, scanner: ModularizedPETScannerGeometry) -> None:
        """
        Parameters
        ----------
        scanner : ModularizedPETScannerGeometry
            a modularized PET scanner 
        """
        self._scanner = scanner

    @abc.abstractmethod
    def get_lor_coordinates(self,
                            **kwargs) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """return the start and end coordinates of all (or a subset of) LORs"""
        raise NotImplementedError

    @property
    def scanner(self) -> ModularizedPETScannerGeometry:
        """the scanner for which coincidences are described"""
        return self._scanner

    @property
    def xp(self) -> ModuleType:
        """array module to use for storing the LOR endpoints"""
        return self.scanner.xp

    @property
    def dev(self) -> str:
        """device to use for storing the LOR endpoints"""
        return self.scanner.dev


class RegularPolygonPETLORDescriptor(PETLORDescriptor):

    def __init__(
        self,
        scanner: RegularPolygonPETScannerGeometry,
        radial_trim: int = 3,
        max_ring_difference: int | None = None,
    ) -> None:
        """Coincidence descriptor for a regular polygon PET scanner where
           we have coincidences within and between "rings (polygons of modules)" 
           The geometrical LORs can be sorted into a sinogram having a
           "plane", "view" and "radial" axis.

        Parameters
        ----------
        scanner : RegularPolygonPETScannerGeometry
            a regular polygon PET scanner
        radial_trim : int, optional
            number of geometrial LORs to disregard in the radial direction, by default 3
        max_ring_difference : int | None, optional
            maximim ring difference to consider for coincidences, by default None means
            all ring differences are included
        """

        super().__init__(scanner)

        self._radial_trim = radial_trim

        if max_ring_difference is None:
            self._max_ring_difference = self.scanner.num_rings - 1
        else:
            self._max_ring_difference = max_ring_difference

        self._num_rad = (self.scanner.num_lor_endpoints_per_ring +
                         1) - 2 * self._radial_trim
        self._num_views = self.scanner.num_lor_endpoints_per_ring // 2

        self._setup_plane_indices()
        self._setup_view_indices()

    @property
    def radial_trim(self) -> int:
        """number of geometrial LORs to disregard in the radial direction"""
        return self._radial_trim

    @property
    def max_ring_difference(self) -> int:
        """the maximum ring difference"""
        return self._max_ring_difference

    @property
    def num_planes(self) -> int:
        """number of planes in the sinogram"""
        return self._num_planes

    @property
    def num_rad(self) -> int:
        """number of radial elements in the sinogram"""
        return self._num_rad

    @property
    def num_views(self) -> int:
        """number of views in the sinogram"""
        return self._num_views

    @property
    def start_plane_index(self) -> npt.NDArray:
        """start plane for all planes"""
        return self._start_plane_index

    @property
    def end_plane_index(self) -> npt.NDArray:
        """end plane for all planes"""
        return self._end_plane_index

    @property
    def start_in_ring_index(self) -> npt.NDArray:
        """start index within ring for all views - shape (num_view, num_rad)"""
        return self._start_in_ring_index

    @property
    def end_in_ring_index(self) -> npt.NDArray:
        """end index within ring for all views - shape (num_view, num_rad)"""
        return self._end_in_ring_index

    def _setup_plane_indices(self) -> None:
        """setup the start / end plane indices (similar to a Michelogram)
        """
        self._start_plane_index = self.xp.arange(self.scanner.num_rings,
                                                 dtype=self.xp.int32,
                                                 device=self.dev)
        self._end_plane_index = self.xp.arange(self.scanner.num_rings,
                                               dtype=self.xp.int32,
                                               device=self.dev)

        for i in range(1, self._max_ring_difference + 1):
            tmp1 = self.xp.arange(self.scanner.num_rings - i,
                                  dtype=self.xp.int16,
                                  device=self.dev)
            tmp2 = self.xp.arange(self.scanner.num_rings - i,
                                  dtype=self.xp.int16,
                                  device=self.dev) + i

            self._start_plane_index = self.xp.concat(
                (self._start_plane_index, tmp1, tmp2))
            self._end_plane_index = self.xp.concat(
                (self._end_plane_index, tmp2, tmp1))

        self._num_planes = self._start_plane_index.shape[0]

    def _setup_view_indices(self) -> None:
        """setup the start / end view indices
        """
        n = self.scanner.num_lor_endpoints_per_ring

        m = 2 * (n // 2)

        self._start_in_ring_index = self.xp.zeros(
            (self._num_views, self._num_rad),
            dtype=self.xp.int32,
            device=self.dev)
        self._end_in_ring_index = self.xp.zeros(
            (self._num_views, self._num_rad),
            dtype=self.xp.int32,
            device=self.dev)

        for view in np.arange(self._num_views):
            self._start_in_ring_index[view, :] = (
                self.xp.concat(
                    (self.xp.arange(m) // 2, self.xp.asarray([n // 2]))) -
                view)[self._radial_trim:-self._radial_trim]
            self._end_in_ring_index[view, :] = (
                self.xp.concat(
                    (self.xp.asarray([-1]), -((self.xp.arange(m) + 4) // 2))) -
                view)[self._radial_trim:-self._radial_trim]

        # shift the negative indices
        self._start_in_ring_index = self.xp.where(
            self._start_in_ring_index >= 0, self._start_in_ring_index,
            self._start_in_ring_index + n)
        self._end_in_ring_index = self.xp.where(self._end_in_ring_index >= 0,
                                                self._end_in_ring_index,
                                                self._end_in_ring_index + n)

    def get_lor_indices(
        self,
        views: None | npt.ArrayLike = None,
        sinogram_order=SinogramSpatialAxisOrder.RVP
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """return the start and end indices of all LORs / or a subset of views

        Parameters
        ----------
        views : None | npt.ArrayLike, optional
            the views to consider, by default None means all views
        sinogram_order : SinogramSpatialAxisOrder, optional
            the order of the sinogram axes, by default SinogramSpatialAxisOrder.RVP

        Returns
        -------
        start_mods, end_mods, start_inds, end_inds 
        """

        if views is None:
            views = self.xp.arange(self.num_views, device=self.dev)

        # setup the module and in_module (in_ring) indices for all LORs in PVR order
        start_inring_inds = self.xp.reshape(
            self.xp.take(self.start_in_ring_index, views, axis=0), (-1, ))
        end_inring_inds = self.xp.reshape(
            self.xp.take(self.end_in_ring_index, views, axis=0), (-1, ))

        start_mods, start_inds = self.xp.meshgrid(self.start_plane_index,
                                                  start_inring_inds,
                                                  indexing='ij')
        end_mods, end_inds = self.xp.meshgrid(self.end_plane_index,
                                              end_inring_inds,
                                              indexing='ij')

        # reshape to PVR dimensions (radial moving fastest, planes moving slowest)
        sinogram_spatial_shape = (self.num_planes, views.shape[0],
                                  self.num_rad)
        start_mods = self.xp.reshape(start_mods, sinogram_spatial_shape)
        end_mods = self.xp.reshape(end_mods, sinogram_spatial_shape)
        start_inds = self.xp.reshape(start_inds, sinogram_spatial_shape)
        end_inds = self.xp.reshape(end_inds, sinogram_spatial_shape)

        if sinogram_order is not SinogramSpatialAxisOrder.PVR:
            if sinogram_order is SinogramSpatialAxisOrder.RVP:
                new_order = (2, 1, 0)
            elif sinogram_order is SinogramSpatialAxisOrder.RPV:
                new_order = (2, 0, 1)
            elif sinogram_order is SinogramSpatialAxisOrder.VRP:
                new_order = (1, 2, 0)
            elif sinogram_order is SinogramSpatialAxisOrder.VPR:
                new_order = (1, 0, 2)
            elif sinogram_order is SinogramSpatialAxisOrder.PRV:
                new_order = (0, 2, 1)

            start_mods = self.xp.permute_dims(start_mods, new_order)
            end_mods = self.xp.permute_dims(end_mods, new_order)

            start_inds = self.xp.permute_dims(start_inds, new_order)
            end_inds = self.xp.permute_dims(end_inds, new_order)

        return start_mods, end_mods, start_inds, end_inds 

    def get_lor_coordinates(
        self,
        views: None | npt.ArrayLike = None,
        sinogram_order=SinogramSpatialAxisOrder.RVP
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:

        """return the start and end coordinates of all LORs / or a subset of views

        Parameters
        ----------
        views : None | npt.ArrayLike, optional
            the views to consider, by default None means all views
        sinogram_order : SinogramSpatialAxisOrder, optional
            the order of the sinogram axes, by default SinogramSpatialAxisOrder.RVP

        Returns
        -------
        xstart, xend : npt.ArrayLike
           2 dimensional floating point arrays containing the start and end coordinates of all LORs
        """

        start_mods, end_mods, start_inds, end_inds = self.get_lor_indices(views, sinogram_order)
        sinogram_spatial_shape = start_mods.shape
 
        start_mods = self.xp.reshape(start_mods, (-1, ))
        start_inds = self.xp.reshape(start_inds, (-1, ))

        end_mods = self.xp.reshape(end_mods, (-1, ))
        end_inds = self.xp.reshape(end_inds, (-1, ))

        x_start = self.xp.reshape(
            self.scanner.get_lor_endpoints(start_mods, start_inds),
            sinogram_spatial_shape + (3, ))
        x_end = self.xp.reshape(
            self.scanner.get_lor_endpoints(end_mods, end_inds),
            sinogram_spatial_shape + (3, ))

        return x_start, x_end

    def show_views(self,
                   ax: plt.Axes,
                   views: npt.ArrayLike,
                   planes: npt.ArrayLike,
                   lw: float = 0.2,
                   **kwargs) -> None:
        """show all LORs of a single view in a given plane

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        view : int
            the view number
        plane : int
            the plane number
        lw : float, optional
            the line width, by default 0.2
        """

        xs, xe = self.get_lor_coordinates(
            views=views, sinogram_order=SinogramSpatialAxisOrder.RVP)
        xs = self.xp.reshape(self.xp.take(xs, planes, axis=2), (-1, 3))
        xe = self.xp.reshape(self.xp.take(xe, planes, axis=2), (-1, 3))

        p1s = np.asarray(to_device(xs, 'cpu'))
        p2s = np.asarray(to_device(xe, 'cpu'))

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)


class DemoPETScannerLORDescriptor(RegularPolygonPETLORDescriptor):

    def __init__(self,
                 xp: ModuleType,
                 dev: str,
                 num_rings: int = 9,
                 radial_trim: int = 65,
                 max_ring_difference: int | None = None,
                 symmetry_axis: int = 2) -> None:

        scanner = DemoPETScanner(xp,
                                 dev,
                                 num_rings,
                                 symmetry_axis=symmetry_axis)

        super().__init__(scanner,
                         radial_trim=radial_trim,
                         max_ring_difference=max_ring_difference)


class RegularPolygonPETProjector(parallelproj.LinearOperator):

    def __init__(self,
                 lor_descriptor: RegularPolygonPETLORDescriptor,
                 img_shape: tuple[int, int, int],
                 voxel_size: tuple[float, float, float],
                 img_origin: None | npt.ArrayLike = None,
                 views: None | npt.ArrayLike = None,
                 resolution_model: None | parallelproj.LinearOperator = None,
                 tof: bool = False):
        """Regular polygon PET projector

        Parameters
        ----------
        lor_descriptor : RegularPolygonPETLORDescriptor
            descriptor of the LOR start / end points
        img_shape : tuple[int, int, int]
            shape of the image to be projected
        voxel_size : tuple[float, float, float]
            the voxel size of the image to be projected
        img_origin : None | npt.ArrayLike, optional
            the origin of the image to be projected, by default None 
            means that image is "centered" in the scanner
        views : None | npt.ArrayLike, optional
            sinogram views to be projected, by default None
            means that all views are being projected
        resolution_model : None | parallelproj.LinearOperator, optional
            an image-based resolution model applied before forward projection, by default None
            means an isotropic 4.5mm FWHM Gaussian smoothing is used
        tof: bool, optional, default False
            whether to use non-TOF or TOF projections
        """

        super().__init__()
        self._dev = lor_descriptor.dev

        self._lor_descriptor = lor_descriptor
        self._img_shape = img_shape
        self._voxel_size = self.xp.asarray(voxel_size,
                                           dtype=self.xp.float32,
                                           device=self._dev)

        if img_origin is None:
            self._img_origin = (-(self.xp.asarray(
                self._img_shape, dtype=self.xp.float32, device=self._dev) / 2)
                                + 0.5) * self._voxel_size
        else:
            self._img_origin = self.xp.asarray(img_origin,
                                               dtype=self.xp.float32,
                                               device=self._dev)

        if views is None:
            self._views = self.xp.arange(self._lor_descriptor.num_views,
                                         device=self._dev)
        else:
            self._views = views

        if resolution_model is None:
            self._resolution_model = parallelproj.GaussianFilterOperator(
                self.in_shape, sigma=4.5 / (2.355 * self._voxel_size))
        else:
            self._resolution_model = resolution_model

        self._xstart, self._xend = lor_descriptor.get_lor_coordinates(
            views=self._views, sinogram_order=SinogramSpatialAxisOrder['RVP'])

        self._tof = tof
        self._tof_parameters = TOFParameters()

    @property
    def in_shape(self) -> tuple[int, int, int]:
        return self._img_shape

    @property
    def out_shape(self) -> tuple[int, int, int]:
        if self.tof:
            out_shape = (self._lor_descriptor.num_rad, self._views.shape[0],
                         self._lor_descriptor.num_planes,
                         self.tof_parameters.num_tofbins)
        else:
            out_shape = (self._lor_descriptor.num_rad, self._views.shape[0],
                         self._lor_descriptor.num_planes)

        return out_shape

    @property
    def xp(self) -> ModuleType:
        return self._lor_descriptor.xp

    @property
    def tof(self) -> bool:
        return self._tof

    @tof.setter
    def tof(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError('tof must be a boolean')
        self._tof = value

    @property
    def tof_parameters(self) -> TOFParameters:
        return self._tof_parameters

    @tof_parameters.setter
    def tof_parameters(self, value: TOFParameters) -> None:
        if not isinstance(value, TOFParameters):
            raise ValueError('tof_parameters must be a TOFParameters object')
        self._tof_parameters = value

    def _apply(self, x):
        """nonTOF forward projection of input image x including image based resolution model"""

        dev = device(x)
        x_sm = self._resolution_model(x)

        if not self.tof:
            x_fwd = parallelproj.joseph3d_fwd(self._xstart, self._xend, x_sm,
                                              self._img_origin,
                                              self._voxel_size)
        else:
            x_fwd = parallelproj.joseph3d_fwd_tof_sino(
                self._xstart, self._xend, x_sm, self._img_origin,
                self._voxel_size, self._tof_parameters.tofbin_width,
                self.xp.asarray([self._tof_parameters.sigma_tof],
                                dtype=self.xp.float32,
                                device=dev),
                self.xp.asarray([self._tof_parameters.tofcenter_offset],
                                dtype=self.xp.float32,
                                device=dev), self.tof_parameters.num_sigmas,
                self.tof_parameters.num_tofbins)

        return x_fwd

    def _adjoint(self, y):
        """nonTOF back projection of sinogram y"""
        dev = device(y)

        if not self.tof:
            y_back = parallelproj.joseph3d_back(self._xstart, self._xend,
                                                self._img_shape,
                                                self._img_origin,
                                                self._voxel_size, y)
        else:
            y_back = parallelproj.joseph3d_back_tof_sino(
                self._xstart, self._xend, self._img_shape, self._img_origin,
                self._voxel_size, y, self._tof_parameters.tofbin_width,
                self.xp.asarray([self._tof_parameters.sigma_tof],
                                dtype=self.xp.float32,
                                device=dev),
                self.xp.asarray([self._tof_parameters.tofcenter_offset],
                                dtype=self.xp.float32,
                                device=dev), self.tof_parameters.num_sigmas,
                self.tof_parameters.num_tofbins)

        return self._resolution_model.adjoint(y_back)



def distributed_subset_order(n: int) -> list[int]:
    """subset order that maximizes distance between subsets

    Parameters
    ----------
    n : int
        number of subsets

    Returns
    -------
    list[int]
    """    
    l = [x for x in range(n)]
    o = []

    for i in range(n):
        if (i % 2) == 0:
            o.append(l.pop(0))
        else:
            o.append(l.pop(len(l)//2))

    return o

