import requests
import zipfile
import io
from pathlib import Path

import nibabel as nib
from pymirc.image_operations import zoom3d
import array_api_compat.numpy as np
from array_api_compat import to_device, device, get_namespace
from math import ceil

import numpy.typing as npt
from types import ModuleType


def download_data(
        zip_file_url:
    str = 'https://zenodo.org/record/8067595/files/brainweb_petmr_v2.zip',
        force: bool = False,
        out_path: Path | None = None):
    """download simulated brainweb PET/MR images

    Parameters
    ----------
    zip_file_url : str, optional
        by default 'https://zenodo.org/record/8067595/files/brainweb_petmr_v2.zip'
    force : bool, optional
        force download even if data is already present, by default False
    out_path : Path | None, optional
        output path for the data, by default None
    """

    if out_path is None:
        out_path = Path('.') / 'data'
    out_path.mkdir(parents=True, exist_ok=True)

    if not (out_path / 'subject54').exists() or force:
        print('downloading data')
        r = requests.get(zip_file_url)
        print('download finished')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(out_path)
        print(f'extracted data into {out_path}')
    else:
        print('data already present')


def load_brain_image(i: int,
                     xp: ModuleType,
                     dev: str,
                     voxel_size: tuple[float, float, float] = (1., 1., 1.),
                     axial_fov_mm: None | float = None,
                     normalize_emission: bool = True,
                     verbose: bool = False) -> tuple[npt.NDArray, npt.NDArray]:
    """load a brainweb PET emission / attenuation data set

    Parameters
    ----------
    i : int
        ID of the data set (0-59)
        we have 20 subjects with 3 images each
    xp : ModuleType
        array module type to use
    dev : str
        device to use (cpu or cuda)
    voxel_size : None | tuple[float, float, float], optional
        voxel size, by default (1., 1., 1.)
    axial_fov_mm : None | float, optional
        by default None means do not crop axial FOV
    normalize_emission : bool, optional
        divide emission image to maximum, by default True
    verbose : bool, optional
        verbose output, by default False

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        normalized emission image and attenuation image (1/mm)
    """
    subject_dirs = sorted(list(Path('data').glob('subject??')))
    subject_index = i // 3
    image_index = i % 3
    if verbose:
        print(
            f'\rloading image {(i+1):03} {subject_dirs[subject_index]} image_{image_index:03}.nii.gz'
        )
    emission_img = nib.load(subject_dirs[subject_index] /
                            f'image_{image_index}.nii.gz').get_fdata()
    scale = emission_img.max()

    attenuation_img = nib.load(subject_dirs[subject_index] /
                               f'attenuation_image.nii.gz').get_fdata()

    if axial_fov_mm is not None:
        # clip axial extent of the images
        start = int(0.5 * emission_img.shape[2] - 0.5 * axial_fov_mm)
        stop = int(ceil(start + axial_fov_mm))
        emission_img = emission_img[:, :, start:stop]
        attenuation_img = attenuation_img[:, :, start:stop]

    if voxel_size is not None:
        # regrid images to desired voxel size
        emission_img = zoom3d(emission_img, 1 / np.array(voxel_size))
        attenuation_img = zoom3d(attenuation_img, 1 / np.array(voxel_size))

    if normalize_emission:
        emission_img = emission_img / scale

    return to_device(xp.asarray(emission_img, dtype=xp.float32),
                     dev), to_device(
                         xp.asarray(attenuation_img, dtype=xp.float32), dev)


def load_brain_image_batch(ids, xp, dev, **kwargs):
    for i, ii in enumerate(ids):
        em_img, att_img = load_brain_image(ii, xp, dev, **kwargs)

        if i == 0:
            img_shape = em_img.shape
            em_img_batch = xp.zeros((len(ids), 1) + img_shape,
                                    device=dev,
                                    dtype=xp.float32)
            att_img_batch = xp.zeros((len(ids), 1) + img_shape,
                                     device=dev,
                                     dtype=xp.float32)

        em_img_batch[i, 0, ...] = em_img
        att_img_batch[i, 0, ...] = att_img

    return em_img_batch, att_img_batch


def simulate_data_batch(
    emission_image_batch: npt.NDArray,
    attenuation_image_batch: npt.NDArray,
    subset_projectors: npt.NDArray,
    sens: float = 1.,
    contam_fraction: float = 0.4,
    random_seed: int | None = None
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Simulate a batch of emission data from a batch of emission and attenuation images

    Parameters
    ----------
    emission_image_batch : npt.NDArray
        batch of emission images with shape (batch_size, 1, *image_shape)
    attenuation_image_batch : npt.NDArray
        batch of attenuation images with shape (batch_size, 1, *image_shape)
    subset_projectors : npt.NDArray
        subset projectors
    sens : float, optional
        sensitivity value that determines number of prompts, by default 1.
    contam_fraction : float, optional
        contamination fraction, by default 0.4
    random_seed : int | None, optional
        random seed for reproducibility, by default None -> not set

    Returns
    -------
    npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
        emission_data_batch, correction_batch, contamination_batch, adjoint_ones_batch
    """

    xp = get_namespace(emission_image_batch)
    dev = device(emission_image_batch)

    if 'torch' in xp.__name__:
        xp.manual_seed(random_seed)
    else:
        xp.random.seed(random_seed)

    num_subsets = subset_projectors.num_subsets
    batch_size = emission_image_batch.shape[0]

    # mini batch of multiplicative corrections (attenuation and normalization)
    correction_batch = xp.zeros(
        (num_subsets, batch_size) + subset_projectors.out_shapes[0],
        device=dev,
        dtype=xp.float32)

    # mini batch of emission data
    emission_data_batch = xp.zeros(
        (num_subsets, batch_size) + subset_projectors.out_shapes[0],
        device=dev,
        dtype=xp.float32)

    # calculate the adjoint ones (back projection of the multiplicative corrections) - sensitivity images
    adjoint_ones_batch = xp.zeros(
        (num_subsets, batch_size, 1) + subset_projectors.in_shape,
        device=dev,
        dtype=xp.float32)

    # mini batch of additive contamination (scatter)
    contamination_batch = xp.zeros(
        (num_subsets, batch_size) + subset_projectors.out_shapes[0],
        device=dev,
        dtype=xp.float32)

    for j in range(num_subsets):
        for i in range(batch_size):
            correction_batch[
                j, i, ...] = sens * xp.exp(-subset_projectors.apply_subset(
                    attenuation_image_batch[i, 0, ...], j))

            adjoint_ones_batch[j, i, 0,
                               ...] = subset_projectors.adjoint_subset(
                                   correction_batch[j, i, ...], j)

            emission_data_batch[j, i, ...] = correction_batch[
                j, i, ...] * subset_projectors.apply_subset(
                    emission_image_batch[i, 0, ...], j)

            contamination_batch[j, i, ...] = (
                1 /
                (1 - contam_fraction)) * emission_data_batch[j, i, ...].mean()
            emission_data_batch[j, i, ...] += contamination_batch[j, i, ...]

            if 'torch' in xp.__name__:
                emission_data_batch[j, i,
                                    ...] = xp.poisson(emission_data_batch[j, i,
                                                                          ...])
            else:
                emission_data_batch[j, i, ...] = xp.random.poisson(
                    emission_data_batch[j, i, ...])

    return emission_data_batch, correction_batch, contamination_batch, adjoint_ones_batch
