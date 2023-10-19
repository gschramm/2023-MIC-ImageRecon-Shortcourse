from __future__ import annotations

import utils
import parallelproj
import array_api_compat.torch as torch
from array_api_compat import device

from layers import LinearSingleChannelOperator, AdjointLinearSingleChannelOperator

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
lor_descriptor = utils.DemoPETScannerLORDescriptor(torch,
                                                   dev,
                                                   num_rings=2,
                                                   radial_trim=201)

# image properties
voxel_size = (2.66, 2.66, 2.66)
img_shape = (10, 10, 2 * lor_descriptor.scanner.num_modules)

projector = utils.RegularPolygonPETNonTOFProjector(
    lor_descriptor,
    img_shape,
    voxel_size,
    views=torch.arange(0, lor_descriptor.num_views, 34, device=dev))

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

batch_size = 2

x = torch.rand((batch_size, 1) + projector.in_shape,
               device=dev,
               dtype=torch.float32,
               requires_grad=True)

y = torch.rand((batch_size, ) + projector.out_shape,
               device=dev,
               dtype=torch.float32,
               requires_grad=True)

fwd_op_layer = LinearSingleChannelOperator.apply
adjoint_op_layer = AdjointLinearSingleChannelOperator.apply

f1 = fwd_op_layer(x, projector)
print('forward projection (Ax) .:', f1.shape, type(f1), device(f1))

b1 = adjoint_op_layer(y, projector)
print('back projection (A^T y) .:', b1.shape, type(b1), device(b1))

fb1 = adjoint_op_layer(fwd_op_layer(x, projector), projector)
print('back + forward projection (A^TAx) .:', fb1.shape, type(fb1),
      device(fb1))

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

# define a dummy loss function
dummy_loss = (fb1**2).sum()
# trigger the backpropagation
dummy_loss.backward()

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

if dev == 'cpu':
    print('skipping (slow) gradient checks on cpu')
else:
    print('Running forward projection layer gradient test')
    grad_test_fwd = torch.autograd.gradcheck(fwd_op_layer, (x, projector),
                                             eps=1e-1,
                                             atol=1e-4,
                                             rtol=1e-4)

    print('Running adjoint projection layer gradient test')
    grad_test_fwd = torch.autograd.gradcheck(adjoint_op_layer, (y, projector),
                                             eps=1e-1,
                                             atol=1e-4,
                                             rtol=1e-4)
