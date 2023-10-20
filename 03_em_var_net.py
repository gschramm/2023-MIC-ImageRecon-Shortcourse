from __future__ import annotations

import utils
import parallelproj
import array_api_compat.numpy as np
import array_api_compat.torch as torch
from array_api_compat import to_device
import matplotlib.pyplot as plt

from layers import EMUpdateModule


class SimpleEMVarNet(torch.nn.Module):
    """dummy cascaded model that includes layers combining projections and convolutions"""

    def __init__(self,
                 em_update_module: torch.nn.Module,
                 neural_net: torch.nn.Module | None = None,
                 num_blocks: int = 3) -> None:

        super().__init__()

        self._em_update_module = em_update_module
        self._neural_net_weight = torch.nn.Parameter(torch.tensor(1.0))
        self._num_blocks = num_blocks

        if neural_net is None:
            self._neural_net = torch.nn.Sequential(
                torch.nn.Conv3d(1, 10, 3, padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv3d(10, 10, 3, padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv3d(10, 10, 3, padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv3d(10, 1, 3, padding='same'),
            )
        else:
            self._neural_net = neural_net

    @property
    def neural_net_weight(self) -> torch.Tensor:
        return self._neural_net_weight

    @property
    def neural_net(self) -> torch.nn.Module:
        return self._neural_net

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def forward(self, x: torch.Tensor, emission_data_batch: torch.Tensor,
                correction_batch: torch.Tensor,
                contamination_batch: torch.Tensor,
                adjoint_ones_batch: torch.Tensor) -> torch.Tensor:

        for _ in range(self._num_blocks):
            x_em = self._em_update_module(x, emission_data_batch,
                                          correction_batch,
                                          contamination_batch,
                                          adjoint_ones_batch)
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
lor_descriptor = utils.DemoPETScannerLORDescriptor(torch, dev, num_rings=4)

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

em_var_net = SimpleEMVarNet(em_update_module, neural_net=None)

y = em_var_net(x, emission_data_batch, correction_batch, contamination_batch,
               adjoint_ones_batch)

# calculate the sum of squared differences loss between y and the true emission images
loss = ((y - emission_image_batch)**2).sum()
# backpropagate the gradients
loss.backward()