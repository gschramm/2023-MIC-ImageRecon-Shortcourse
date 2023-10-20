from __future__ import annotations

import array_api_compat.torch as torch
import parallelproj
from array_api_compat import device

class LinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing a linear operator acting on a mini batch of single channel images
    """

    # see also: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

    @staticmethod
    def forward(ctx, x: torch.Tensor,
                operator: parallelproj.LinearOperator) -> torch.Tensor:
        """forward pass of the linear operator

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass 
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, num_voxels_x, num_voxels_y, num_voxels_z)
        operator : parallelproj.LinearOperator
            linear operator that can act on a single 3D image

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, opertor.out_shape)
        """

        #https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros((batch_size, ) + operator.out_shape,
                        dtype=x.dtype,
                        device=device(x))

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i, ...] = operator(x[i, 0, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, operator.out_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size, 1, opertor.in_shape)
        """

        #For details on how to implement the backward pass, see
        #https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros((batch_size, 1) + operator.in_shape,
                            dtype=grad_output.dtype,
                            device=device(grad_output))

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, 0, ...] = operator.adjoint(grad_output[i, ...].detach())

            return x, None


class AdjointLinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing the adjoint of a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor,
                operator: parallelproj.LinearOperator) -> torch.Tensor:
        """forward pass of the adjoint of the linear operator

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass 
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, operator.out_shape)
        operator : parallelproj.LinearOperator
            linear operator that can act on a single 3D image

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, opertor.in_shape)
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros((batch_size, 1) + operator.in_shape,
                        dtype=x.dtype,
                        device=device(x))

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i, 0, ...] = operator.adjoint(x[i, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, 1, operator.in_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size, 1, opertor.out_shape)
        """

        #For details on how to implement the backward pass, see
        #https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros((batch_size, ) + operator.out_shape,
                            dtype=grad_output.dtype,
                            device=device(grad_output))

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, ...] = operator(grad_output[i, 0, ...].detach())

            return x, None


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

