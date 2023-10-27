from __future__ import annotations

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

import array_api_compat.numpy as np
import pymirc.viewer as pv

parser = argparse.ArgumentParser(description='OSEM-VARNet evaluation')
parser.add_argument('--run_dir')

args = parser.parse_args()

run_dir = Path(args.run_dir)

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
ids = tuple([i for i in range(num_training, num_training + num_validation)])

if not (run_dir / 'gt_val.pt').exists():
    emission_image_database, attenuation_image_database = load_brain_image_batch(
        ids,
        torch,
        dev,
        voxel_size=voxel_size,
        axial_fov_mm=0.95 * axial_fov_mm,
        verbose=True)

    torch.save(emission_image_database, run_dir / 'gt_val.pt')
    torch.save(attenuation_image_database, run_dir / 'att_val.pt')
else:
    emission_image_database = torch.load(run_dir / 'gt_val.pt',
                                         map_location=dev)
    attenuation_image_database = torch.load(run_dir / 'att_val.pt',
                                            map_location=dev)

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

if not (run_dir / 'osem_val.pt').exists():
    osem_database = torch.ones(
        (emission_image_database.shape[0], 1) + subset_projectors.in_shape,
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
                correction_database[subset,
                                    ...], contamination_database[subset, ...],
                adjoint_ones_database[subset, ...])

    torch.save(osem_database, run_dir / 'osem_val.pt')
else:
    osem_database = torch.load(run_dir / 'osem_val.pt', map_location=dev)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# evaluate the models
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

if not (run_dir / 'pred_val_post.pt').exists():
    post_recon_unet = PostReconNet(Unet3D(num_features=num_features).to(dev))
    post_recon_unet._neural_net.load_state_dict(
        torch.load(run_dir / 'post_recon_model_best_state.pt'))
    post_recon_unet.eval()

    pred_val_post = post_recon_unet(osem_database)
    torch.save(pred_val_post, run_dir / 'pred_val_post.pt')
else:
    pred_val_post = torch.load(run_dir / 'pred_val_post.pt', map_location=dev)

if not (run_dir / 'pred_val.pt').exists():
    unet = Unet3D(num_features=num_features).to(dev)
    osem_var_net = SimpleOSEMVarNet(osem_update_modules, unet, depth, dev)
    osem_var_net.load_state_dict(torch.load(run_dir / 'model_best_state.pt'))
    osem_var_net.eval()

    pred_val = osem_var_net(osem_database, emission_data_database,
                            correction_database, contamination_database,
                            adjoint_ones_database)
    torch.save(pred_val, run_dir / 'pred_val.pt')
else:
    pred_val = torch.load(run_dir / 'pred_val.pt', map_location=dev)

vi = pv.ThreeAxisViewer([
    np.asarray(to_device(osem_database.detach().squeeze(), 'cpu')),
    np.asarray(to_device(pred_val_post.detach().squeeze(), 'cpu')),
    np.asarray(to_device(pred_val.detach().squeeze(), 'cpu')),
    np.asarray(to_device(emission_image_database.detach().squeeze(), 'cpu'))
],
                        imshow_kwargs=dict(vmin=0, vmax=1.1))
