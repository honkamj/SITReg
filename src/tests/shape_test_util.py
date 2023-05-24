"""Util for testing correct handling of shapes"""

from itertools import chain, product
from typing import Iterator, Sequence

from torch import Tensor

from util.dimension_order import broadcast_to_by_leading_dims


class BroadcastShapeTestingUtil:
    """Namespace for testing utilities for tensors being correctly broadcasted"""
    BATCH_SHAPES = [(1,), (3,), (5,)]
    SPATIAL_SHAPES = [tuple(), (2,), (2, 3)]

    @classmethod
    def expand_tensor_shapes_for_testing(
            cls,
            *tensors: Tensor
        ) -> Iterator[Sequence[Tensor]]:
        """Expand tensor shapes from batch and spatial size

        E.g: Input tensors with shapes (3, 2) and (3, 3), yield:
        (1, 3, 2), (1, 3, 3)
        (5, 3, 2), (5, 3, 3),
        (1, 3, 2, 2), (1, 3, 3, 2)
        ...
        """
        shape_iterator = chain(
            product(
                cls.BATCH_SHAPES,
                cls.SPATIAL_SHAPES),
            [(tuple(), tuple())]
        )
        for batch_shape, spatial_shape in shape_iterator:
            reshaped_tensors = [
                broadcast_to_by_leading_dims(
                    tensor,
                    batch_shape + tuple(tensor.shape) + spatial_shape,
                    tensor.ndim)
                for tensor in tensors
            ]
            yield reshaped_tensors
