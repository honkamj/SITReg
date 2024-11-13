"""Sum pool"""

from typing import Sequence

from torch import Tensor
from torch.nn.functional import avg_pool1d

from util.ndimensional_operators import avg_pool_nd_function


def sum_pool_nd(
    volume: Tensor,
    kernel_size: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
    separable: bool = False,
) -> Tensor:
    """Sum pool an N-dimension tensor"""
    if separable:
        n_dims = len(kernel_size)
        for dim in range(n_dims):
            pooled_volume = volume.moveaxis(-n_dims + dim, -1)
            other_than_last_pooled_dim_shape = pooled_volume.shape[:-1]
            pooled_volume = pooled_volume.reshape(-1, 1, pooled_volume.size(-1))
            pooled_volume = (
                avg_pool1d(  # pylint: disable=not-callable
                    pooled_volume,
                    kernel_size=kernel_size[dim],
                    padding=padding[dim],
                    stride=stride[dim],
                )
                * kernel_size[dim]
            )
            pooled_volume = pooled_volume.reshape(
                other_than_last_pooled_dim_shape + pooled_volume.shape[-1:]
            )
            volume = pooled_volume.moveaxis(-1, -n_dims + dim)
        return volume
    return avg_pool_nd_function(len(kernel_size))(
        volume, kernel_size=kernel_size, padding=padding, stride=stride, divisor_override=1
    )
