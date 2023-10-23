from __future__ import annotations

import utils
import parallelproj
import array_api_compat.torch as torch
from array_api_compat import to_device

from layers import EMUpdateModule
from data import load_brain_image, load_brain_image_batch


class SimpleOSEMVarNet(torch.nn.Module):
    """dummy cascaded model that includes layers combining projections and convolutions"""

    def __init__(self,
                 osem_update_modules: torch.nn.Module,
                 device,
                 neural_net: torch.nn.Module | None = None) -> None:

        super().__init__()

        self._osem_update_modules = osem_update_modules
        self._neural_net_weight = torch.nn.Parameter(torch.tensor(1.0))

        self._num_subsets = len(osem_update_modules)
        self._subset_order = torch.randperm(self._num_subsets)

        if neural_net is None:
            self._neural_net = torch.nn.Sequential(
                torch.nn.Conv3d(1, 10, 3, padding='same', device=device),
                torch.nn.ReLU(),
                torch.nn.Conv3d(10, 10, 3, padding='same', device=device),
                torch.nn.ReLU(),
                torch.nn.Conv3d(10, 10, 3, padding='same', device=device),
                torch.nn.ReLU(),
                torch.nn.Conv3d(10, 1, 3, padding='same', device=device),
            )
        else:
            self._neural_net = neural_net

    @property
    def neural_net_weight(self) -> torch.Tensor:
        return self._neural_net_weight

    @property
    def neural_net(self) -> torch.nn.Module:
        return self._neural_net

    def forward(self, x: torch.Tensor, emission_data_batch: torch.Tensor,
                correction_batch: torch.Tensor,
                contamination_batch: torch.Tensor,
                adjoint_ones_batch: torch.Tensor) -> torch.Tensor:

        for j in range(self._num_subsets):
            subset = self._subset_order[j]
            x_em = osem_update_modules[subset](x, emission_data_batch[subset,
                                                                      ...],
                                               correction_batch[subset, ...],
                                               contamination_batch[subset,
                                                                   ...],
                                               adjoint_ones_batch[subset, ...])

            x_nn = self._neural_net(x)

            # fusion of EM update and neural net update with trainable weight
            # we use an ReLU activation to ensure that the output of each block is non-negative
            x = torch.nn.ReLU()(x_em + self._neural_net_weight * x_nn)

        return x


# device variable (cpu or cuda) that determines whether calculations
# are performed on the cpu or cuda gpu

if parallelproj.cuda_present:
    dev = 'cuda'
else:
    dev = 'cpu'

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#--- setup the scanner / LOR geometry ---------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# setup a line of response descriptor that describes the LOR start / endpoints of
# a "narrow" clinical PET scanner with 9 rings
num_rings = 9
lor_descriptor = utils.DemoPETScannerLORDescriptor(torch, dev, num_rings=9)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#--- load the brainweb images -----------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# image properties
ids = (0, 1)
batch_size = len(ids)
voxel_size = (2.66, 2.66, 2.66)

emission_image_batch, attenuation_image_batch = load_brain_image_batch(
    ids,
    torch,
    dev,
    voxel_size=voxel_size,
    axial_fov_mm=num_rings * voxel_size[2],
    verbose=True)

img_shape = tuple(emission_image_batch.shape[2:])

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

num_subsets = 34

subset_projectors = parallelproj.SubsetOperator([
    utils.RegularPolygonPETNonTOFProjector(
        lor_descriptor,
        img_shape,
        voxel_size,
        views=torch.arange(i,
                           lor_descriptor.num_views,
                           num_subsets,
                           device=dev)) for i in range(num_subsets)
])

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

# mini batch of multiplicative corrections (attenuation and normalization)
correction_batch = torch.zeros(
    (num_subsets, batch_size) + subset_projectors.out_shapes[0],
    device=dev,
    dtype=torch.float32)

# mini batch of emission data
emission_data_batch = torch.zeros(
    (num_subsets, batch_size) + subset_projectors.out_shapes[0],
    device=dev,
    dtype=torch.float32)

# calculate the adjoint ones (back projection of the multiplicative corrections) - sensitivity images
adjoint_ones_batch = torch.zeros(
    (num_subsets, batch_size, 1) + subset_projectors.in_shape,
    device=dev,
    dtype=torch.float32)

# mini batch of additive contamination (scatter)
contamination_batch = torch.zeros(
    (num_subsets, batch_size) + subset_projectors.out_shapes[0],
    device=dev,
    dtype=torch.float32)

for j in range(num_subsets):
    for i in range(batch_size):
        correction_batch[j, i,
                         ...] = torch.exp(-subset_projectors.apply_subset(
                             attenuation_image_batch[i, 0, ...], j))

        adjoint_ones_batch[j, i, 0, ...] = subset_projectors.adjoint_subset(
            correction_batch[j, i, ...], j)

        emission_data_batch[j, i, ...] = correction_batch[
            j, i, ...] * subset_projectors.apply_subset(
                emission_image_batch[i, 0, ...], j)

        contamination_batch[j, i, ...] = emission_data_batch[j, i, ...].mean()
        emission_data_batch[j, i, ...] += contamination_batch[j, i, ...]
        emission_data_batch[j, i,
                            ...] = torch.poisson(emission_data_batch[j, i,
                                                                     ...])

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

osem_update_modules = [
    EMUpdateModule(projector) for projector in subset_projectors.operators
]

x = torch.ones((batch_size, 1) + subset_projectors.in_shape,
               device=dev,
               dtype=torch.float32)

osem_var_net = SimpleOSEMVarNet(osem_update_modules, dev, neural_net=None)

subset_order = torch.randperm(num_subsets)
print(f'OSEM recon')
for j in range(num_subsets):
    subset = subset_order[j]
    x = osem_update_modules[subset](x, emission_data_batch[subset, ...],
                                    correction_batch[subset, ...],
                                    contamination_batch[subset, ...],
                                    adjoint_ones_batch[subset, ...])

y = osem_var_net(x, emission_data_batch, correction_batch, contamination_batch,
                 adjoint_ones_batch)

# calculate the sum of squared differences loss between y and the true emission images
loss = ((y - emission_image_batch)**2).sum()
# backpropagate the gradients
loss.backward()
