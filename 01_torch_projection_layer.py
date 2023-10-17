from __future__ import annotations

import utils
import parallelproj
import array_api_compat.numpy as np
import array_api_compat.torch as torch
import matplotlib.pyplot as plt
from array_api_compat import device, to_device

class LinearSingleChannel3DOperator(torch.autograd.Function):
    """
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

        batch_size = x.shape[0]
        y = torch.zeros((batch_size,1) + operator.out_shape, dtype = x.dtype, 
                        device = device(x))


        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i,0,:,:,:] = operator(x[i,0,:,:,:].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros((batch_size,1) + operator.in_shape, dtype = grad_output.dtype, 
                            device = device(grad_output))
            
            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i,0,:,:,:] = operator.adjoint(grad_output[i,0,:,:,:].detach())

            return x, None


class AdjointLinearSingleChannel3DOperator(torch.autograd.Function):
    """
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

        batch_size = x.shape[0]
        y = torch.zeros((batch_size,1) + operator.in_shape, dtype = x.dtype, 
                        device = device(x))


        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i,0,:,:,:] = operator.adjoint(x[i,0,:,:,:].detach())

        return y



    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros((batch_size,1) + operator.out_shape, dtype = grad_output.dtype, 
                            device = device(grad_output))
            
            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i,0,:,:,:] = operator(grad_output[i,0,:,:,:].detach())

            return x, None


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

projector = utils.RegularPolygonPETNonTOFProjector(lor_descriptor, img_shape,
                                                   voxel_size, 
                                                   views = torch.arange(0, lor_descriptor.num_views, 34, device = dev))

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

x = torch.rand((1,1) + projector.in_shape,
               device=dev,
               dtype=torch.float32,
               requires_grad=True)

y = torch.rand((1,1) + projector.out_shape,
               device=dev,
               dtype=torch.float32,
               requires_grad=True)

fwd_op_layer = LinearSingleChannel3DOperator.apply
adjoint_op_layer = AdjointLinearSingleChannel3DOperator.apply

f1 = fwd_op_layer(x, projector)
print('forward projection (Ax) .:', f1.shape, type(f1), device(f1))


b1 = adjoint_op_layer(y, projector)
print('back projection (A^T y) .:', b1.shape, type(b1), device(b1))

fb1 = adjoint_op_layer(fwd_op_layer(x, projector), projector)
print('back + forward projection (A^TAx) .:', fb1.shape, type(fb1), device(fb1))

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
