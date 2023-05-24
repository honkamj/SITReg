"""Custom activation implementations"""

from torch.autograd import Function
from torch.autograd.function import FunctionCtx, once_differentiable
from torch import Tensor
from torch.nn.functional import relu, leaky_relu


class _MemoryEfficientReLU(Function):  # pylint: disable=abstract-method
    """More memory efficient ReLU implementation"""

    @staticmethod
    def forward(  # type: ignore # pylint: disable=arguments-differ
        ctx: FunctionCtx, tensor: Tensor
    ) -> Tensor:
        (grad_needed,) = ctx.needs_input_grad  # type: ignore
        if grad_needed:
            ctx.save_for_backward(tensor > 0.0)
        return relu(tensor)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):  # type: ignore # pylint: disable=arguments-differ
        (grad_needed,) = ctx.needs_input_grad
        if grad_needed:
            boolean_mask: Tensor
            (boolean_mask,) = ctx.saved_tensors
            return grad_output * boolean_mask
        return None


memory_efficient_relu = _MemoryEfficientReLU.apply


class _MemoryEfficientLeakyReLU(Function):  # pylint: disable=abstract-method
    """More memory efficient leaky ReLU implementation"""

    @staticmethod
    def forward(  # type: ignore # pylint: disable=arguments-differ
        ctx: FunctionCtx, tensor: Tensor, negative_slope: float = 0.01
    ) -> Tensor:
        (grad_needed,) = ctx.needs_input_grad  # type: ignore
        if grad_needed:
            ctx.negative_slope = negative_slope  # type: ignore
            ctx.save_for_backward(tensor > 0.0)
        return leaky_relu(tensor, negative_slope=negative_slope)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore # pylint: disable=arguments-differ
        (grad_needed,) = ctx.needs_input_grad
        if grad_needed:
            boolean_mask: Tensor
            (boolean_mask,) = ctx.saved_tensors
            negative_slope: float = ctx.negative_slope
            return (
                grad_output * boolean_mask
                + negative_slope * grad_output * boolean_mask.logical_not()
            )
        return None


memory_efficient_leaky_relu = _MemoryEfficientLeakyReLU.apply


class MemoryEfficientLeakyReLU:
    """More memory efficient leaky ReLU implementation"""

    def __init__(self, negative_slope: float = 0.01) -> None:
        self._negative_slope = negative_slope

    def __call__(self, tensor: Tensor) -> Tensor:
        return memory_efficient_leaky_relu(tensor, self._negative_slope)
