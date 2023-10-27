from __future__ import annotations

# TODO: deeper UNet

import argparse
import json
from datetime import datetime
import utils
import parallelproj
import array_api_compat.torch as torch
from array_api_compat import to_device

from layers import EMUpdateModule
from models import Unet3D, SimpleOSEMVarNet, PostReconNet
from data import load_brain_image, load_brain_image_batch, simulate_data_batch
from math import ceil

import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description='OSEM-VARNet reconstruction')
parser.add_argument('--num_datasets', type=int, default=60)
parser.add_argument('--num_training', type=int, default=40)
parser.add_argument('--num_validation', type=int, default=20)
parser.add_argument('--num_subsets', type=int, default=4)
parser.add_argument('--depth', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--num_epochs_post', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_features', type=int, default=8)
parser.add_argument('--num_rings', type=int, default=4)
parser.add_argument('--radial_trim', type=int, default=181)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--voxel_size',
                    nargs='+',
                    type=float,
                    default=[2.5, 2.5, 2.66])

args = parser.parse_args()

num_datasets = args.num_datasets
num_training = args.num_training
num_validation = args.num_validation
num_subsets = args.num_subsets
depth = args.depth
num_epochs = args.num_epochs
num_epochs_post = args.num_epochs_post
batch_size = args.batch_size
num_features = args.num_features
num_rings = args.num_rings
radial_trim = args.radial_trim
random_seed = args.random_seed
voxel_size = tuple(args.voxel_size)

# device variable (cpu or cuda) that determines whether calculations
# are performed on the cpu or cuda gpu
if parallelproj.cuda_present:
    dev = 'cuda'
else:
    dev = 'cpu'

output_dir = Path(
    'run_osem_varnet') / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
output_dir.mkdir(exist_ok=True, parents=True)

with open(output_dir / 'input_cfg.json', 'w') as f:
    json.dump(vars(args), f)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#--- setup the scanner / LOR geometry ---------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# setup a line of response descriptor that describes the LOR start / endpoints of
# a "narrow" clinical PET scanner with 9 rings
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
    emission_image_database,
    attenuation_image_database,
    subset_projectors,
    random_seed=random_seed)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

osem_update_modules = [
    EMUpdateModule(projector) for projector in subset_projectors.operators
]

osem_database = torch.ones((num_datasets, 1) + subset_projectors.in_shape,
                           device=dev,
                           dtype=torch.float32)

num_osem_iter = 102 // num_subsets

subset_order = utils.distributed_subset_order(num_subsets)

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

post_recon_unet = PostReconNet(Unet3D(num_features=num_features).to(dev))
post_recon_unet.train()

loss_fn_post = torch.nn.MSELoss()
optimizer_post = torch.optim.Adam(post_recon_unet.parameters(), lr=1e-3)

loss_arr_post = torch.zeros(num_epochs_post)

min_val_loss_post = float('inf')

for epoch in range(num_epochs_post):
    batch_inds = torch.split(torch.randperm(num_training), batch_size)

    for ib, batch_ind in enumerate(batch_inds):
        x_fwd_post = post_recon_unet(osem_database[batch_ind, ...])
        loss_post = loss_fn_post(x_fwd_post, emission_image_database[batch_ind,
                                                                     ...])

        print(
            f'{(epoch+1):03}/{num_epochs_post:03} {(ib+1):03} {loss_post:.2E}',
            end='\r')

        # Backpropagation
        loss_post.backward()
        optimizer_post.step()
        optimizer_post.zero_grad()

    loss_arr_post[epoch] = loss_post

    if (epoch + 1) % 20 == 0:
        post_recon_unet.eval()
        x_fwd_post = post_recon_unet(
            osem_database[num_training:(num_training + num_validation), ...])
        val_loss_post = loss_fn_post(
            x_fwd_post, emission_image_database[num_training:(num_training +
                                                              num_validation),
                                                ...])
        print(
            f'{(epoch+1):03}/{num_epochs_post:03} val_loss {val_loss_post:.2E}'
        )
        post_recon_unet.train()

        if val_loss_post < min_val_loss_post:
            min_val_loss_post = val_loss_post
            torch.save(post_recon_unet.neural_net.state_dict(),
                       output_dir / 'post_recon_model_best_state.pt')

torch.save(loss_arr_post, output_dir / 'training_loss_post.pt')
torch.save(post_recon_unet.neural_net.state_dict(),
           output_dir / 'post_recon_model_last_state.pt')

del post_recon_unet

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# model training
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

print('\nvarnet training\n')

unet = Unet3D(num_features=num_features).to(dev)
unet.load_state_dict(torch.load(output_dir / 'post_recon_model_best_state.pt'))

osem_var_net = SimpleOSEMVarNet(osem_update_modules, unet, depth, dev)
osem_var_net.train()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(osem_var_net.parameters(), lr=1e-3)

loss_arr = torch.zeros(num_epochs)
min_val_loss = float('inf')

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

    if (epoch + 1) % 20 == 0:
        osem_var_net.eval()
        val_loss = 0
        for iv in range(num_training, (num_training + num_validation)):
            x_fwd = osem_var_net(osem_database[iv:(iv + 1), ...],
                                 emission_data_database[:, iv:(iv + 1), ...],
                                 correction_database[:, iv:(iv + 1), ...],
                                 contamination_database[:, iv:(iv + 1), ...],
                                 adjoint_ones_database[:, iv:(iv + 1), ...])
            val_loss += float(
                loss_fn(x_fwd, emission_image_database[iv:(iv + 1), ...]))

        val_loss /= num_validation
        print(f'{(epoch+1):03}/{num_epochs:03} val_loss {val_loss:.2E}')
        osem_var_net.train()

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(osem_var_net.state_dict(),
                       output_dir / 'model_best_state.pt')

torch.save(loss_arr, output_dir / 'training_loss.pt')
torch.save(osem_var_net.state_dict(), output_dir / 'model_last_state.pt')
