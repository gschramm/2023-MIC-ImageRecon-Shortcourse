from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from array_api_compat import device
from pathlib import Path


def create_dummy_data(root: str = 'data/dummy_data',
                      img_shape: tuple[int, int, int] = (128, 128, 90),
                      sino_shape: tuple[int, ...] = (257, 180, 400),
                      num_datasets: int = 8) -> None:
    """create a number of PET dummy data sets (images and sinograms)

    Parameters
    ----------
    root : str, optional
        data root direction, by default 'data/dummy_data'
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
            print(f'loading {acq_dir.name}')

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


if __name__ == '__main__':
    #----------------------------------------------------------------------
    #--- (1) create a dummy data sets -------------------------------------
    #----------------------------------------------------------------------
    data_root_path = 'data/dummy_data'
    create_dummy_data(root=data_root_path)


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
                                pin_memory=True)

    print('\nloading mini batches - 1st epoch\n')
    for i_batch, sample_batched in enumerate(pet_dataloader):
        print('batch id: ', i_batch,
              sample_batched['high_quality_image'].size(),
              device(sample['high_quality_image']),
              sample_batched['emission_sinogram'].size())

    print('\nloading mini batches - 2nd epoch\n')
    for i_batch, sample_batched in enumerate(pet_dataloader):
        print('batch id: ', i_batch,
              sample_batched['high_quality_image'].size(),
              device(sample['high_quality_image']),
              sample_batched['emission_sinogram'].size())
