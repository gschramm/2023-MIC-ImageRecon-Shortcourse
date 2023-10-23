import requests
import zipfile
import io
from pathlib import Path

import nibabel as nib
from pymirc.image_operations import zoom3d
import array_api_compat.numpy as np 
from math import ceil

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


if __name__ == '__main__':

    i = 10
    voxel_size = (2.66, 2.66, 2.66)
    subject_dirs = sorted(list(Path('data').glob('subject??'))) 
    axial_fov_mm = 9*2.66

    subject_index = i // 3
    image_index = i % 3
    print(
        f'\rloading image {(i+1):03} {subject_dirs[subject_index]} image_{image_index:03}.nii.gz',
        end='')
    tmp = nib.load(subject_dirs[subject_index] /
                   f'image_{image_index}.nii.gz').get_fdata()
    scale = tmp.max()

    if axial_fov_mm is not None:
        start = int(0.5*tmp.shape[2] - 0.5* axial_fov_mm)
        stop = int(ceil(start +  axial_fov_mm))
        tmp = tmp[:,:,start:stop]
        

    # regrid images to desired voxel size
    tmp = zoom3d(tmp, 1/ np.array(voxel_size))

