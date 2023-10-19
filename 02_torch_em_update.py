from __future__ import annotations

import utils
import parallelproj
import array_api_compat.numpy as np
import array_api_compat.torch as torch
from array_api_compat import to_device
import matplotlib.pyplot as plt

from layers import LinearSingleChannelOperator, AdjointLinearSingleChannelOperator


class EMUpdateModule(torch.nn.Module):

    def __init__(
        self,
        projector: parallelproj.LinearOperator,
    ) -> None:

        super().__init__()
        self._projector = projector

        self._fwd_op_layer = LinearSingleChannelOperator.apply
        self._adjoint_op_layer = AdjointLinearSingleChannelOperator.apply

    def forward(self, x: torch.Tensor, data: torch.Tensor,
                corrections: torch.Tensor, contamination: torch.Tensor,
                adjoint_ones: torch.Tensor) -> torch.Tensor:
        """forward pass of the EM update module

        Parameters
        ----------
        x : torch.Tensor
            mini batch of images with shape (batch_size, 1, *img_shape)
        data : torch.Tensor
            mini batch of emission data with shape (batch_size, *data_shape)
        corrections : torch.Tensor
            mini batch of multiplicative corrections with shape (batch_size, *data_shape)
        contamination : torch.Tensor
            mini batch of additive contamination with shape (batch_size, *data_shape)
        adjoint_ones : torch.Tensor
            mini batch of adjoint ones (back projection of multiplicative corrections) with shape (batch_size, 1, *img_shape)

        Returns
        -------
        torch.Tensor
            mini batch of EM updates with shape (batch_size, 1, *img_shape)
        """

        # remember that all variables contain a mini batch of images / data arrays
        # and that the fwd / adjoint operator layers directly operate on mini batches

        y = data / (corrections * self._fwd_op_layer(x, self._projector) +
                    contamination)

        return x * self._adjoint_op_layer(corrections * y,
                                          self._projector) / adjoint_ones


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

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
lor_descriptor = utils.DemoPETScannerLORDescriptor(torch, dev, num_rings=2)

# image properties
voxel_size = (2.66, 2.66, 2.66)
n0 = 160
n1 = n0
img_shape = (n0, n1, 2 * lor_descriptor.scanner.num_modules)

projector = utils.RegularPolygonPETNonTOFProjector(lor_descriptor, img_shape,
                                                   voxel_size)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

batch_size = 2

emission_image_batch = torch.zeros((batch_size, 1) + projector.in_shape,
                                   device=dev,
                                   dtype=torch.float32,
                                   requires_grad=False)

emission_image_batch[:, 0, (n0 // 4):(3 * n0 // 4),
                     (n1 // 4):(3 * n1 // 4), :] = 0.4
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
correction_batch = torch.zeros((batch_size, ) + projector.out_shape,
                               device=dev,
                               dtype=torch.float32)
# mini batch of emission data
emission_data_batch = torch.zeros((batch_size, ) + projector.out_shape,
                                  device=dev,
                                  dtype=torch.float32)

# calculate the adjoint ones (back projection of the multiplicative corrections) - sensitivity images
adjoint_ones_batch = torch.zeros((batch_size, 1) + projector.in_shape,
                                 device=dev,
                                 dtype=torch.float32)

# mini batch of additive contamination (scatter)
contamination_batch = torch.zeros((batch_size, ) + projector.out_shape,
                                  device=dev,
                                  dtype=torch.float32)

for i in range(batch_size):
    correction_batch[i,
                     ...] = torch.exp(-projector(attenuation_image_batch[i, 0,
                                                                         ...]))

    emission_data_batch[i, ...] = correction_batch[i, ...] * projector(
        emission_image_batch[i, 0, ...])

    contamination_batch[i, ...] = emission_data_batch[i, ...].mean()
    emission_data_batch[i, ...] += contamination_batch[i, ...]

    emission_data_batch[i, ...] = torch.poisson(emission_data_batch[i, ...])

    adjoint_ones_batch[i, 0, ...] = projector.adjoint(correction_batch[i, ...])

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

em_update_module = EMUpdateModule(projector)
x = torch.ones((batch_size, 1) + projector.in_shape,
               device=dev,
               dtype=torch.float32)

num_iter = 20
for i in range(num_iter):
    print(f'EM iteration {(i+1):03}/{num_iter:03}', end='\r')
    x = em_update_module(x, emission_data_batch, correction_batch,
                         contamination_batch, adjoint_ones_batch)

print('')
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
ax[1, 0].set_title(f'MLEM {num_iter}it - slice {sl} - batch item 0',
                   fontsize='small')
ax[1, 1].set_title(f'MLEM {num_iter}it- slice {sl} - batch item 1',
                   fontsize='small')
fig.tight_layout()
fig.show()