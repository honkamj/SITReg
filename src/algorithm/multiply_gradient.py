"""Multiply gradients while backpropagating."""

from torch import Tensor
from torch.autograd import Function

# pylint: disable=arguments-differ, abstract-method


class _MultiplyBackward(Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, multiplier: float | int) -> Tensor:
        ctx.multiplier = multiplier
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.multiplier, None


def multiply_backward(tensor: Tensor, multiplier: float | int):
    """Multiply gradients while backpropagating"""
    return _MultiplyBackward.apply(tensor, multiplier)
