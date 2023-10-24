from __future__ import annotations

import torch
import parallelproj
from torch.utils.data import Dataset, DataLoader
from array_api_compat import device, to_device
from pathlib import Path
from shutil import rmtree

from time import time


def create_dummy_data(root: str,
                      img_shape: tuple[int, int, int] = (128, 128, 90),
                      sino_shape: tuple[int, ...] = (257, 180, 400),
                      num_datasets: int = 15) -> None:
    """create a number of PET dummy data sets (images and sinograms)

    Parameters
    ----------
    root : str, optional
        data root direction
    img_shape : tuple[int, int, int], optional
        shape of the images, by default (128, 128, 90)
    sino_shape : tuple[int, ...], optional
        shape of the sinograms, by default (257, 180, 400)
    num_datasets : int, optional
        number of data sets to create, by default 8
    """

    root_dir = Path(root)
    root_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_datasets):
        acq_dir = (root_dir / f'acquisition_{i:03}')
        acq_dir.mkdir(exist_ok=True)

        print(f'creating {acq_dir.name}', end='\r')

        torch.save(torch.full(img_shape, 1 + 100 * i, dtype=torch.float32),
                   acq_dir / 'high_quality_image.pt')
        torch.save(torch.full(img_shape, 2 + 100 * i, dtype=torch.float32),
                   acq_dir / 'sensitivity_image.pt')
        torch.save(torch.full(sino_shape, 3 + 100 * i, dtype=torch.int16),
                   acq_dir / 'emission_sinogram.pt')
        torch.save(torch.full(sino_shape, 4 + 100 * i, dtype=torch.float32),
                   acq_dir / 'correction_sinogram.pt')
        torch.save(torch.full(sino_shape, 5 + 100 * i, dtype=torch.float32),
                   acq_dir / 'contamination_sinogram.pt')

    print()


class PETDataSet(Dataset):
    """Dummy PET data set consisting of images and sinograms"""

    def __init__(self,
                 root_dir: str,
                 pattern: str = 'acquisition_*',
                 verbose: bool = False) -> None:
        """

        Parameters
        ----------
        root_dir : str
            root of data director
        pattern : str, optional
            pattern of sub directories to be included, by default 'acquisition_*'
        verbose : bool, optional
            verbose output, by default False

        Note
        ----
        We expect to find the following files in each sub directory:
        - high_quality_image.pt
        - sensitivity_image.pt
        - emission_sinogram.pt
        - correction_sinogram.pt
        - contamination_sinogram.pt
        """

        self._root_dir: Path = Path(root_dir)
        self._pattern: str = pattern
        self._acquisition_dirs: list[Path] = sorted(
            list(self._root_dir.glob(self._pattern)))
        self._verbose: bool = verbose

    def __len__(self) -> int:
        return len(self._acquisition_dirs)

    def __getitem__(self, idx):
        acq_dir = self._acquisition_dirs[idx]

        if self._verbose:
            print(f'loading {str(acq_dir)}')

        sample = {}
        sample['high_quality_image'] = torch.load(acq_dir /
                                                  'high_quality_image.pt')
        sample['sensitivity_image'] = torch.load(acq_dir /
                                                 'sensitivity_image.pt')
        sample['emission_sinogram'] = torch.load(acq_dir /
                                                 'emission_sinogram.pt')
        sample['correction_sinogram'] = torch.load(acq_dir /
                                                   'correction_sinogram.pt')
        sample['contamination_sinogram'] = torch.load(
            acq_dir / 'contamination_sinogram.pt')

        return sample


#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------


class CustomBatchCollator:

    def __init__(self, dim: int = 0) -> None:
        self._dim = dim

    def __call__(self, batch: dict[list[torch.Tensor]]) -> dict[torch.Tensor]:
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = torch.stack([x[key] for x in batch],
                                          dim=self._dim)

        return batch_dict


if __name__ == '__main__':
    if parallelproj.cuda_present:
        dev = 'cuda'
    else:
        dev = 'cpu'

    #----------------------------------------------------------------------
    #--- (1) create a dummy data sets -------------------------------------
    #----------------------------------------------------------------------
    data_root_path = '/tmp/dummy_data'
    create_dummy_data(root=data_root_path,
                      num_datasets=8,
                      img_shape=(128, 128, 90),
                      sino_shape=(257, 180, 400))

    #----------------------------------------------------------------------
    #--- (2) create a pytorch dataset object that describes ---------------
    #---     how to load the data                           ---------------
    #----------------------------------------------------------------------
    pet_dataset = PETDataSet(data_root_path, verbose=True)

    # load a single sample from our data set
    print('\nloading single data set\n')
    sample = pet_dataset[0]
    print(sample['high_quality_image'].shape,
          device(sample['high_quality_image']),
          sample['emission_sinogram'].shape)

    #----------------------------------------------------------------------
    #--- (3) create a data loader that can sample mini batches  -----------
    #----------------------------------------------------------------------
    pet_dataloader = DataLoader(pet_dataset,
                                batch_size=3,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=CustomBatchCollator(dim=0))

    for epoch in range(5):
        print('\n--------------------------------')
        print(f'loading mini batches - epoch {epoch:03}')
        print('--------------------------------\n')
        ta = time()
        for i_batch, sample_batched in enumerate(pet_dataloader):
            # push tensors to device
            high_quality_image_batched = to_device(
                sample_batched['high_quality_image'], dev)
            sensitivity_image_batched = to_device(
                sample_batched['sensitivity_image'], dev)
            emission_sinogram_batched = to_device(
                sample_batched['emission_sinogram'], dev)
            correction_sinogram_batched = to_device(
                sample_batched['correction_sinogram'], dev)
            contamination_sinogram_batched = to_device(
                sample_batched['contamination_sinogram'], dev)

            print('batch id: ', i_batch, high_quality_image_batched.shape,
                  device(high_quality_image_batched),
                  emission_sinogram_batched.shape)
        tb = time()
        print(
            f'\ntime needed to sample mini batch {((tb-ta)/len(pet_dataloader)):.4f}s'
        )

    # delete the temporary data directory
    rmtree(data_root_path)
