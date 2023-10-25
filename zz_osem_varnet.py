from __future__ import annotations

import utils
import parallelproj
import array_api_compat.torch as torch
from array_api_compat import to_device

from layers import EMUpdateModule
from models import Unet3D
from data import load_brain_image, load_brain_image_batch, simulate_data_batch
from math import ceil

import tempfile
from pathlib import Path

class SimpleOSEMVarNet(torch.nn.Module):
    """dummy cascaded model that includes layers combining projections and convolutions"""

    def __init__(self,
                 osem_update_modules: torch.nn.Module,
                 device,
                 neural_net: torch.nn.Module | None = None) -> None:

        super().__init__()

        self._osem_update_modules = osem_update_modules

        self._num_subsets = len(osem_update_modules)
        self._subset_order = torch.randperm(self._num_subsets)
        self._neural_net_weight = torch.nn.ParameterList(
            [torch.ones(1, device=device) for _ in range(self._num_subsets)])

        if neural_net is None:
            self._neural_net = Unet3D(num_features=8).to(device)
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
            x_em = self._osem_update_modules[subset](
                x, emission_data_batch[subset, ...], correction_batch[subset,
                                                                      ...],
                contamination_batch[subset, ...], adjoint_ones_batch[subset,
                                                                     ...])

            x_nn = self._neural_net(x)

            # fusion of EM update and neural net update with trainable weight
            # we use an ReLU activation to ensure that the output of each block is non-negative
            x = torch.nn.ReLU()(x_em + self._neural_net_weight[j] * x_nn)

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
num_rings = 4
radial_trim = 181
lor_descriptor = utils.DemoPETScannerLORDescriptor(torch,
                                                   dev,
                                                   num_rings=num_rings,
                                                   radial_trim=radial_trim)
axial_fov_mm = float(lor_descriptor.scanner.num_rings *
                     (lor_descriptor.scanner.ring_positions[1] -
                      lor_descriptor.scanner.ring_positions[0]))

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#--- load the brainweb images -----------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# image properties
ids = tuple([i for i in range(40)])
batch_size = len(ids)
voxel_size = (2.5, 2.5, 2.66)
#voxel_size = (2.66, 2.66, 2.66)

emission_image_database, attenuation_image_database = load_brain_image_batch(
    ids,
    torch,
    dev,
    voxel_size=voxel_size,
    axial_fov_mm=0.95 * axial_fov_mm,
    verbose=True)

img_shape = tuple(emission_image_database.shape[2:])

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

num_subsets = 4

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

print(f'simulating data')

emission_data_database, correction_database, contamination_database, adjoint_ones_database = simulate_data_batch(
    emission_image_database, attenuation_image_database, subset_projectors)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

osem_update_modules = [
    EMUpdateModule(projector) for projector in subset_projectors.operators
]

osem_database = torch.ones((batch_size, 1) + subset_projectors.in_shape,
                           device=dev,
                           dtype=torch.float32)

osem_var_net = SimpleOSEMVarNet(osem_update_modules, dev, neural_net=None)

num_osem_iter = 34 // num_subsets

for i in range(num_osem_iter):
    print(f'OSEM iteration {(i+1):003}/{num_osem_iter:003}', end='\r')
    subset_order = torch.randperm(num_subsets)
    for j in range(num_subsets):
        subset = subset_order[j]
        osem_database = osem_update_modules[subset](
            osem_database, emission_data_database[subset, ...],
            correction_database[subset, ...],
            contamination_database[subset, ...], adjoint_ones_database[subset,
                                                                       ...])
                                                                        ...])

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# model training
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(osem_var_net.parameters(), lr=1e-3)

num_datasets = emission_image_database.shape[0]
num_epochs = 500
batch_size = 5

print('\nmodel training\n')

osem_var_net.train()

loss_arr = torch.zeros(num_epochs)

for epoch in range(num_epochs):
    batch_inds = torch.split(torch.randperm(num_datasets), batch_size)

    for ib, batch_ind in enumerate(batch_inds):
        x_fwd = osem_var_net(osem_database[batch_ind, ...],
                             emission_data_database[:, batch_ind, ...],
                             correction_database[:, batch_ind, ...],
                             contamination_database[:, batch_ind, ...],
                             adjoint_ones_database[:, batch_ind, ...])

        loss = loss_fn(x_fwd, emission_image_database[batch_ind, ...])

        print(f'{(epoch+1):03}/{num_epochs:03} {(ib+1):03} {loss:.2E}',
              end='\r')

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss_arr[epoch] = loss

output_dir = tempfile.TemporaryDirectory(dir = '.', prefix = 'run_osem_varnet')
output_path = Path(output_dir.name) / 'model_state.pt'
torch.save(osem_var_net.state_dict(), output_path.name)
print(f'save model to {output_path.name}')

##--------------------------------------------------------------------------------
##--------------------------------------------------------------------------------
## model evaluation
##--------------------------------------------------------------------------------
##--------------------------------------------------------------------------------
#
#osem_var_net.load_state_dict(torch.load('osem_varnet_4ss.pt',
#                                        map_location=dev))
#osem_var_net.eval()
#
#x_fwd = osem_var_net(osem_database[:, ...], emission_data_database[:, :, ...],
#                     correction_database[:, :, ...],
#                     contamination_database[:, :,
#                                            ...], adjoint_ones_database[:, :,
#