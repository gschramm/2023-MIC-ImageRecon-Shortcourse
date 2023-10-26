""" minimal example that shows how to setup pytorch data sets and data loaders
    to handle image and sinogram data 
"""

from __future__ import annotations

import tempfile
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

        print(f'creating dummy data {str(acq_dir)}', end='\r')

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
                 add_channel_dim: bool = True,
                 verbose: bool = False) -> None:
        """

        Parameters
        ----------
        root_dir : str
            root of data director
        pattern : str, optional
            pattern of sub directories to be included, by default 'acquisition_*'
        add_channel_dim : bool, optional
            add an extra channel dimension to all images, by default True
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
        self._add_channel_dim: bool = add_channel_dim
        self._verbose: bool = verbose

    def __len__(self) -> int:
        return len(self._acquisition_dirs)

    def __getitem__(self, idx):
        acq_dir = self._acquisition_dirs[idx]

        if self._verbose:
            print(f'loading {str(acq_dir)}')

        sample = {}

        if self._add_channel_dim:
            # we use unsqueeze to add a channel dimension to the images
            sample['high_quality_image'] = torch.unsqueeze(
                torch.load(acq_dir / 'high_quality_image.pt'), 0)
            sample['sensitivity_image'] = torch.unsqueeze(
                torch.load(acq_dir / 'sensitivity_image.pt'), 0)
        else:
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


def custom_collate_fn(batch: list[dict[torch.Tensor]]) -> dict[torch.Tensor]:
    """custom collate function for the PET data set that stacks images along
       the "0" dimension and sinograms along the "1" dimension

    Parameters
    ----------
    batch : list[dict[torch.Tensor]]
        list of samples from data set belonging to the same mini batch 

    Returns
    -------
    dict[torch.Tensor]
        dictionary with stacked tensors
    """
    batch_dict = {}
    for key in batch[0].keys():
        if '_sinogram' in key:
            batch_dict[key] = torch.stack([x[key] for x in batch], dim=1)
        else:
            batch_dict[key] = torch.stack([x[key] for x in batch], dim=0)

    return batch_dict


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

if __name__ == '__main__':
    if parallelproj.cuda_present:
        dev = 'cuda'
    else:
        dev = 'cpu'

    #----------------------------------------------------------------------
    #--- (1) create a dummy data sets -------------------------------------
    #----------------------------------------------------------------------
    tmp_data_dir = tempfile.TemporaryDirectory()
    create_dummy_data(root=tmp_data_dir.name,
                      num_datasets=9,
                      img_shape=(128, 128, 90),
                      sino_shape=(257, 180, 400))

    #----------------------------------------------------------------------
    #--- (2) create a pytorch dataset object that describes ---------------
    #---     how to load the data                           ---------------
    #----------------------------------------------------------------------
    pet_dataset = PETDataSet(tmp_data_dir.name,
                             add_channel_dim=True,
                             verbose=True)

    # load a single sample from our data set
    print('\n-----------------------')
    print('loading single data set')
    print('-----------------------\n')
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
                                pin_memory=False)

    # hint: we can also use a custom function to collate the data set samples in a different way
    # e.g. along a different dimension
    # to do so use: collate_fn=custom_collate_fn

    for epoch in range(6):
        print('\n--------------------------------')
        print(f'loading mini batches - epoch {epoch:03}')
        print('--------------------------------\n')
        ta = time()

        # loop over the data loader as we would do in a training loop
        for i_batch, sample_batched in enumerate(pet_dataloader):
            # push batch tensors to device
            high_quality_image_batched = sample_batched['high_quality_image'].to(dev)
            sensitivity_image_batched = sample_batched['sensitivity_image'].to(dev)
            emission_sinogram_batched = sample_batched['emission_sinogram'].to(dev)
            correction_sinogram_batched = sample_batched['correction_sinogram'].to(dev)
            contamination_sinogram_batched = sample_batched['contamination_sinogram'].to(dev)

            print(f'batch id: {i_batch}')
            print(
                f'...high_quality_image_batch shape / device .: {high_quality_image_batched.shape} / {device(high_quality_image_batched)}'
            )
            print(
                f'...emission_sinogram_batch  shape / device .: {emission_sinogram_batched.shape} / {device(emission_sinogram_batched)}\n'
            )

        tb = time()
        print(
            f'\naverage time needed to sample mini batch {((tb-ta)/len(pet_dataloader)):.4f}s'
        )

    # delete the temporary data directory
    tmp_data_dir.cleanup()
