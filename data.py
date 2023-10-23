import requests
import zipfile
import io
from pathlib import Path

import nibabel as nib
from pymirc.image_operations import zoom3d
import array_api_compat.numpy as np
from array_api_compat import to_device
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

    return xp.asarray(to_device(emission_img, dev),
                      dtype=xp.float32), xp.asarray(to_device(
                          attenuation_img, dev),
                                                    dtype=xp.float32)


def load_brain_image_batch(ids, xp, dev, **kwargs):
    for i in ids:
        em_img, att_img = load_brain_image(i, xp, dev, **kwargs)

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
