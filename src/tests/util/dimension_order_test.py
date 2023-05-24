"""Tests for dimension order utility functions"""

from itertools import repeat
from typing import Iterable, Sequence, Union
from unittest import TestCase

from torch import empty

from util.dimension_order import (broadcast_shapes_by_leading_dims,
                                  broadcast_tensors_by_leading_dims,
                                  broadcast_to_by_leading_dims,
                                  reduce_channel_shape_to_ones)


class LeadingDimsBroadcastingTests(TestCase):
    """Tests for broadcasting with leading dims"""
    SHAPES_AND_LEADING_DIMS: Sequence[
        tuple[
            Sequence[Sequence[int]],
            Union[Sequence[int], int]
        ]
    ] = (
        (
            ((4,), (3, 2)),
            (1, 1)
        ),
        (
            ((5,), (3, 2, 4)),
            1
        ),
        (
            ((6,), (3, 2, 4, 5)),
            (1, 1)
        ),
        (
            ((4,), (3, 2), (6, 7)),
            (1, 1, 2)
        ),
        (
            ((5,), (3, 2, 4), (6, 7)),
            (1, 1, 2)
        ),
        (
            ((2,), (3, 2, 4, 5), (6, 7)),
            (1, 1, 2)
        ),
        (
            ((2,),),
            (2,)
        ),
        (
            ((1, 2, 3, 5), (1, 2, 4, 5)),
            (2, 1)
        )
    )

    TARGET_SHAPES: Sequence[
        Sequence[
            Sequence[int]
        ]
    ] = (
        ((3, 4), (3, 2)),
        ((3, 5, 4), (3, 2, 4)),
        ((3, 6, 4, 5), (3, 2, 4, 5)),
        ((3, 4), (3, 2), (3, 6, 7)),
        ((3, 5, 4), (3, 2, 4), (3, 6, 7, 4)),
        ((3, 2, 4, 5), (3, 2, 4, 5), (3, 6, 7, 4, 5)),
        ((1, 2),),
        ((1, 2, 3, 4, 5), (1, 2, 4, 5))
    )

    INVALID_SHAPES_AND_LEADING_DIMS: Sequence[
        tuple[
            Sequence[Sequence[int]],
            Sequence[int]
        ]
    ] = (
        (
            ((4, 3), (3, 2)),
            (1, 1)
        ),
        (
            ((1, 2, 3, 6), (1, 2, 4, 5)),
            (2, 1)
        )
    )

    @staticmethod
    def _leading_dims_to_iterable(num_leading_dims: Union[int, Iterable[int]]) -> Iterable[int]:
        if isinstance(num_leading_dims, int):
            return repeat(num_leading_dims)
        else:
            return num_leading_dims

    def test_leading_dims_shape_broadcasting(
           self
        ) -> None:
        """Test that correct shapes are produced"""
        for (input_shapes, num_leading_dims), target_shapes in zip(
                self.SHAPES_AND_LEADING_DIMS,
                self.TARGET_SHAPES
            ):
            num_leading_dims_iterable = self._leading_dims_to_iterable(num_leading_dims)
            input_tensors = [empty(*input_shape) for input_shape in input_shapes]
            for (
                        input_shape,
                        leading_dims,
                        target_shape,
                        broadcasted_shape,
                        broadcasted_tensor
                    ) in zip(
                        input_shapes,
                        num_leading_dims_iterable,
                        target_shapes,
                        broadcast_shapes_by_leading_dims(input_shapes, num_leading_dims),
                        broadcast_tensors_by_leading_dims(input_tensors, num_leading_dims)):
                self.assertSequenceEqual(
                    broadcasted_shape,
                    target_shape
                )
                self.assertSequenceEqual(
                    broadcasted_tensor.shape,
                    target_shape
                )
                self.assertSequenceEqual(
                    broadcast_to_by_leading_dims(
                        empty(*input_shape),
                        target_shape,
                        leading_dims).shape,
                    target_shape
                )


    def test_invalid_leading_dims_shape_broadcasting(
           self
        ) -> None:
        """Test that error is raised"""
        for input_shapes, num_leading_dims in self.INVALID_SHAPES_AND_LEADING_DIMS:
            input_tensors = [empty(*input_shape) for input_shape in input_shapes]
            with self.assertRaises(RuntimeError):
                broadcast_shapes_by_leading_dims(input_shapes, num_leading_dims)
                broadcast_tensors_by_leading_dims(input_tensors, num_leading_dims)

    SHAPES_TO_REDUCE_AND_NUM_CHANNELS = (
        (
            (3, 2, 5, 4),
            1
        ),
        (
            (3, 2),
            1
        ),
        (
            (2,),
            1
        ),
        (
            (3, 2, 1, 5, 4),
            2
        ),
        (
            (3, 2, 1),
            2
        ),
        (
            (2, 1),
            2
        )
    )
    REDUCED_SHAPES = (
        (3, 1, 5, 4),
        (3, 1),
        (1,),
        (3, 1, 1, 5, 4),
        (3, 1, 1),
        (1, 1)
    )

    def test_correct_dims_reduced_to_one(self) -> None:
        """Test that correct dims are reduced to one"""
        for (shape_to_reduce, num_channels), reduced_shape in zip(
                self.SHAPES_TO_REDUCE_AND_NUM_CHANNELS,
                self.REDUCED_SHAPES):
            self.assertSequenceEqual(
                reduce_channel_shape_to_ones(shape_to_reduce, num_channels),
                reduced_shape
            )
