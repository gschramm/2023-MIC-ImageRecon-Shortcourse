from __future__ import annotations

#""" TODO: - non-random subset order
#          - depth independent from num subsets
#          - batch size validation
#          - unet resnet style - for better pre-training
#"""        

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

def distributed_subset_order(n: int) -> list[int]:
    l = [x for x in range(n)]
    o = []

    for i in range(n):
        if (i % 2) == 0:
            o.append(l.pop(0))
        else:
            o.append(l.pop(len(l)//2))

    return o


class SimpleOSEMVarNet(torch.nn.Module):
    """dummy cascaded model that includes layers combining projections and convolutions"""

    def __init__(self,
                 osem_update_modules: torch.nn.Module,
                 neural_net: torch.nn.Module,
                 depth: int,
                 device) -> None:

        super().__init__()

        self._osem_update_modules = osem_update_modules

        self._num_subsets = len(osem_update_modules)
        self._subset_order = distributed_subset_order(self._num_subsets)

        self._neural_net = neural_net
        self._depth = depth

        self._neural_net_weight = torch.nn.ParameterList(
            [torch.ones(1, device=device) for _ in range(self._depth)])

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

        for j in range(self._depth):
            subset = self._subset_order[j % self._num_subsets]
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

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

# device variable (cpu or cuda) that determines whether calculations
# are performed on the cpu or cuda gpu

if parallelproj.cuda_present:
    dev = 'cuda'
else:
    dev = 'cpu'

num_datasets = 60
num_training = 40
num_validation = 20
num_subsets = 1
depth = 8
num_epochs = 500
batch_size = 3
num_features = 16


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
ids = tuple([i for i in range(num_datasets)])
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

num_osem_iter = 102 // num_subsets

subset_order = distributed_subset_order(num_subsets)

for i in range(num_osem_iter):
    print(f'OSEM iteration {(i+1):003}/{num_osem_iter:003}', end='\r')
    for j in range(num_subsets):
        subset = subset_order[j]
        osem_database = osem_update_modules[subset](
            osem_database, emission_data_database[subset, ...],
            correction_database[subset, ...],
            contamination_database[subset, ...], adjoint_ones_database[subset,
                                                                       ...])

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# model training
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

print('\npostrecon unet training\n')


post_recon_unet = Unet3D(num_features=num_features).to(dev)
post_recon_unet.train()

loss_fn_post = torch.nn.MSELoss()
optimizer_post = torch.optim.Adam(post_recon_unet.parameters(), lr=1e-3)

loss_arr_post = torch.zeros(num_epochs)

for epoch in range(num_epochs):
    batch_inds = torch.split(torch.randperm(num_training), batch_size)

    for ib, batch_ind in enumerate(batch_inds):
        x_fwd_post = post_recon_unet(osem_database[batch_ind, ...])
        loss_post = loss_fn_post(x_fwd_post, emission_image_database[batch_ind, ...])

        print(f'{(epoch+1):03}/{num_epochs:03} {(ib+1):03} {loss_post:.2E}',
              end='\r')

        # Backpropagation
        loss_post.backward()
        optimizer_post.step()
        optimizer_post.zero_grad()

    loss_arr_post[epoch] = loss_post

    if (epoch+1) % 20 == 0:
        post_recon_unet.eval()
        x_fwd_post = post_recon_unet(osem_database[num_training:(num_training+num_validation), ...])
        val_loss_post = loss_fn_post(x_fwd_post, emission_image_database[num_training:(num_training+num_validation), ...])
        print(f'{(epoch+1):03}/{num_epochs:03} val_loss {val_loss_post:.2E}')
        post_recon_unet.train()
   

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# model training
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

print('\nvarnet training\n')

unet = Unet3D(num_features=num_features).to(dev)
osem_var_net = SimpleOSEMVarNet(osem_update_modules, unet, depth, dev)
osem_var_net.train()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(osem_var_net.parameters(), lr=1e-3)

loss_arr = torch.zeros(num_epochs)

for epoch in range(num_epochs):
    batch_inds = torch.split(torch.randperm(num_training), batch_size)

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

    if (epoch+1) % 20 == 0:
        osem_var_net.eval()
        x_fwd = osem_var_net(osem_database[num_training:(num_training+num_validation), ...],
                             emission_data_database[:,num_training:(num_training+num_validation) , ...],
                             correction_database[:,num_training:(num_training+num_validation) , ...],
                             contamination_database[:,num_training:(num_training+num_validation) , ...],
                             adjoint_ones_database[:,num_training:(num_training+num_validation) , ...])
        val_loss = loss_fn(x_fwd, emission_image_database[num_training:(num_training+num_validation), ...])
        print(f'{(epoch+1):03}/{num_epochs:03} val_loss {val_loss:.2E}')
        osem_var_net.train()
 
output_dir = tempfile.TemporaryDirectory(dir = '.', prefix = 'run_osem_varnet_')
output_path = Path(output_dir.name) / 'model_state.pt'
torch.save(osem_var_net.state_dict(), str(output_path))
print(f'save model to {str(output_path)}')

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
