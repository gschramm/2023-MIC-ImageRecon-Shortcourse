"""minimal script that evaluates trained OSEM varnets
"""

from __future__ import annotations

import argparse
import json
import utils
import parallelproj
import array_api_compat.torch as torch

from layers import EMUpdateModule
from models import Unet3D, SimpleOSEMVarNet, PostReconNet
from data import load_brain_image_batch, simulate_data_batch

from pathlib import Path

import array_api_compat.numpy as np
import pymirc.viewer as pv

parser = argparse.ArgumentParser(description='OSEM-VARNet evaluation')
parser.add_argument('--run_dir')
parser.add_argument('--sens', type=float, default=1)

args = parser.parse_args()

run_dir = Path(args.run_dir)
sens = args.sens

with open(run_dir / 'input_cfg.json', 'r') as f:
    cfg = json.load(f)

num_datasets = cfg['num_datasets']
num_training = cfg['num_training']
num_validation = cfg['num_validation']
num_subsets = cfg['num_subsets']
depth = cfg['depth']
num_epochs = cfg['num_epochs']
num_epochs_post = cfg['num_epochs_post']
batch_size = cfg['batch_size']
num_features = cfg['num_features']
num_rings = cfg['num_rings']
radial_trim = cfg['radial_trim']
random_seed = cfg['random_seed']
voxel_size = tuple(cfg['voxel_size'])

if 'fusion_mode' in cfg:
    fusion_mode = cfg['fusion_mode']
else:
    fusion_mode = 'simple'

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
dataset_ids = tuple(
    [i for i in range(num_training, num_training + num_validation)])

val_loss_post = {}
val_loss = {}

for i, dataset_id in enumerate(dataset_ids):
    emission_image_database, attenuation_image_database = load_brain_image_batch(
        (dataset_id, ),
        torch,
        dev,
        voxel_size=voxel_size,
        axial_fov_mm=0.95 * axial_fov_mm,
        verbose=True)

    img_shape = tuple(emission_image_database.shape[2:])

    # setup a filter operator to post filter the input OSEM's for reference
    filt_op = parallelproj.GaussianFilterOperator(img_shape, 1.0)

    if i == 0:
        pred_val_post = torch.zeros((num_validation, ) + img_shape,
                                    device='cpu',
                                    dtype=torch.float32)
        pred_val = torch.zeros((num_validation, ) + img_shape,
                               device='cpu',
                               dtype=torch.float32)
        input_images = torch.zeros((num_validation, ) + img_shape,
                                   device='cpu',
                                   dtype=torch.float32)
        input_images_sm = torch.zeros((num_validation, ) + img_shape,
                                      device='cpu',
                                      dtype=torch.float32)
        ref_images = torch.zeros((num_validation, ) + img_shape,
                                 device='cpu',
                                 dtype=torch.float32)

    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------

    subset_projectors = parallelproj.SubsetOperator([
        utils.RegularPolygonPETProjector(
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

    emission_data_database, correction_database, contamination_database, adjoint_ones_database = simulate_data_batch(
        emission_image_database,
        attenuation_image_database,
        subset_projectors,
        sens=sens,
        random_seed=random_seed)

    del attenuation_image_database

    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------

    osem_update_modules = [
        EMUpdateModule(projector) for projector in subset_projectors.operators
    ]

    osem_database = torch.ones(
        (emission_image_database.shape[0], 1) + subset_projectors.in_shape,
        device=dev,
        dtype=torch.float32)

    num_osem_iter = 102 // num_subsets

    subset_order = utils.distributed_subset_order(num_subsets)

    for _ in range(num_osem_iter):
        for j in range(num_subsets):
            subset = subset_order[j]
            osem_database = osem_update_modules[subset](
                osem_database, emission_data_database[subset, ...],
                correction_database[subset,
                                    ...], contamination_database[subset, ...],
                adjoint_ones_database[subset, ...])

    input_images[i, ...] = osem_database.detach().cpu().squeeze()
    input_images_sm[i, ...] = filt_op(osem_database.detach().cpu().squeeze())
    ref_images[i, ...] = emission_image_database.detach().cpu().squeeze()
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    # evaluate the models
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------

    loss_fn = torch.nn.MSELoss()

    post_recon_unet = PostReconNet(Unet3D(num_features=num_features).to(dev))
    post_recon_unet._neural_net.load_state_dict(
        torch.load(run_dir / 'post_recon_model_best_state.pt'))
    post_recon_unet.eval()

    x_fwd = post_recon_unet(osem_database)
    val_loss_post[dataset_id] = float(loss_fn(x_fwd, emission_image_database))
    pred_val_post[i, ...] = x_fwd.detach().cpu().squeeze()

    unet = Unet3D(num_features=num_features).to(dev)
    osem_var_net = SimpleOSEMVarNet(osem_update_modules, unet, depth, dev, fusion_mode=fusion_mode)
    osem_var_net.load_state_dict(torch.load(run_dir / 'model_best_state.pt'))
    osem_var_net.eval()

    x_fwd = osem_var_net(osem_database, emission_data_database,
                         correction_database, contamination_database,
                         adjoint_ones_database)
    val_loss[dataset_id] = float(loss_fn(x_fwd, emission_image_database))
    pred_val[i, ...] = x_fwd.detach().cpu().squeeze()

val_loss['mean'] = sum(val_loss.values()) / num_validation
val_loss_post['mean'] = sum(val_loss_post.values()) / num_validation

print(str(run_dir))
print(cfg)
print(val_loss_post['mean'], val_loss['mean'])

with open(run_dir / 'val_loss.json', 'w') as f:
    json.dump(val_loss, f)

with open(run_dir / 'val_loss_post.json', 'w') as f:
    json.dump(val_loss_post, f)

# show all inputs, predictions, and reference images
vi = pv.ThreeAxisViewer([
    np.asarray(input_images),
    np.asarray(input_images_sm),
    np.asarray(pred_val_post),
    np.asarray(pred_val),
    np.asarray(ref_images)
],
                        imshow_kwargs=dict(vmin=0, vmax=1.1),
                        ls='',
                        width=4)
