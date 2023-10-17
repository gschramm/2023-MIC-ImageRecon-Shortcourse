from __future__ import annotations

#TODO: batch + channel dim
import utils
import parallelproj
import array_api_compat.numpy as np
import array_api_compat.torch as torch
import matplotlib.pyplot as plt
from array_api_compat import device, to_device


class LinearDummyOperator():

    def __init__(self):
        self._A = torch.rand(4, 3)

    @property
    def in_shape(self):
        return (3, )

    @property
    def out_shape(self):
        return (4, )

    def __call__(self, x):
        return self._A @ x

    def adjoint(self, y):
        return self._A.T @ y


class LinearOperatorForwardLayer(torch.autograd.Function):
    """PET forward projection layer mapping a 3D image to a 4D sinogram

    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, operator: parallelproj.LinearOperator):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        return operator(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator
            # since forward takes two input arguments (x, operator)
            # we have to return two arguments (the latter is None)
            return operator.adjoint(grad_output), None


class LinearOperatorAdjointLayer(torch.autograd.Function):
    """ adjoint of PET forward projection layer mapping a 4D sinogram to a 3D image
    
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, operator: parallelproj.LinearOperator):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        return operator.adjoint(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator
            return operator(grad_output), None


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
                                                   radial_trim=171)

# image properties
voxel_size = (2.66, 2.66, 2.66)
img_shape = (10, 10, 2 * lor_descriptor.scanner.num_modules)

projector = utils.RegularPolygonPETNonTOFProjector(lor_descriptor, img_shape,
                                                   voxel_size)

#projector = LinearDummyOperator()

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

x = torch.rand(projector.in_shape,
               device=dev,
               dtype=torch.float32,
               requires_grad=True)

y = torch.rand(projector.out_shape,
               device=dev,
               dtype=torch.float32,
               requires_grad=True)

fwd_layer = LinearOperatorForwardLayer.apply
adjoint_layer = LinearOperatorAdjointLayer.apply

f1 = fwd_layer(x, projector)
f2 = projector(x.detach())

assert torch.allclose(f1, f2)

a1 = adjoint_layer(y, projector)
a2 = projector.adjoint(y.detach())

assert torch.allclose(a1, a2)

## define a dummy loss function
#dummy_loss = (y**2).sum()
## trigger the backpropagation
#dummy_loss.backward()

#grad_test_fwd = torch.autograd.gradcheck(fwd_layer, (x, projector),
#                                         eps=1e-1,
#                                         atol=1e-1,
#                                         rtol=1e-1)

#grad_test_fwd = torch.autograd.gradcheck(adjoint_layer, (y, projector),
#                                         eps=1e-1,
#                                         atol=1e-1,
#                                         rtol=1e-1)

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------