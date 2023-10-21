from __future__ import annotations

import utils
import parallelproj
import array_api_compat.numpy as np
import array_api_compat.torch as torch
from array_api_compat import to_device
import matplotlib.pyplot as plt

from layers import EMUpdateModule

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
lor_descriptor = utils.DemoPETScannerLORDescriptor(torch, dev, num_rings=4)

# image properties
voxel_size = (2.66, 2.66, 2.66)
n0 = 160
n1 = n0
img_shape = (n0, n1, 2 * lor_descriptor.scanner.num_modules)

num_subsets = 34

subset_projectors = parallelproj.SubsetOperator([
    utils.RegularPolygonPETNonTOFProjector(lor_descriptor,
                                           img_shape,
                                           voxel_size,
                                           views=torch.arange(
                                               i, lor_descriptor.num_views,
                                               num_subsets))
    for i in range(num_subsets)
])

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

batch_size = 2

emission_image_batch = torch.zeros(
    (batch_size, 1) + subset_projectors.in_shape,
    device=dev,
    dtype=torch.float32,
    requires_grad=False)

emission_image_batch[:, 0, (n0 // 4):(3 * n0 // 4),
                     (n1 // 4):(3 * n1 // 4), :] = 1.
emission_image_batch[0, 0, (9 * n0 // 16):(11 * n0 // 16),
                     (9 * n1 // 16):(11 * n1 // 16), :] *= 2

emission_image_batch[1, 0, (5 * n0 // 16):(7 * n0 // 16),
                     (5 * n1 // 16):(7 * n1 // 16), :] *= 0.5

attenuation_image_batch = 0.01 * (emission_image_batch > 0)

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

num_iter = 136 // num_subsets

for i in range(num_iter):
    subset_order = torch.randperm(num_subsets)
    for j in range(num_subsets):
        subset = subset_order[j]
        print(f'OSEM iteration {(subset+1):03}/{(i+1):03}/{num_iter:03}',
              end='\r')
        x = osem_update_modules[subset](x, emission_data_batch[subset, ...],
                                        correction_batch[subset, ...],
                                        contamination_batch[subset, ...],
                                        adjoint_ones_batch[subset, ...])

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

sl = 1

kwgs = dict(vmax=1.1 * float(emission_image_batch.max()), cmap='Greys')

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 0].imshow(
    np.asarray(to_device(emission_image_batch[0, 0, :, :, sl], 'cpu')), **kwgs)
ax[0, 1].imshow(
    np.asarray(to_device(emission_image_batch[1, 0, :, :, sl], 'cpu')), **kwgs)
ax[1, 0].imshow(np.asarray(to_device(x[0, 0, :, :, sl], 'cpu')), **kwgs)
ax[1, 1].imshow(np.asarray(to_device(x[1, 0, :, :, sl], 'cpu')), **kwgs)

ax[0, 0].set_title(f'true img - slice {sl} - batch item 0', fontsize='small')
ax[0, 1].set_title(f'true img - slice {sl} - batch item 1', fontsize='small')
ax[1, 0].set_title(
    f'OSEM {num_iter}it/{num_subsets}ss - slice {sl} - batch item 0',
    fontsize='small')
ax[1, 1].set_title(
    f'OSEM {num_iter}it/{num_subsets}ss - slice {sl} - batch item 1',
    fontsize='small')
fig.tight_layout()
fig.show()