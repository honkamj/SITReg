"""Tests for dense deformation primitives"""

from unittest import TestCase

from torch import device, ones, rand, tensor
from torch.testing import assert_close

from algorithm.dense_deformation import (
    convert_normalized_to_voxel_coordinates,
    convert_voxel_to_normalized_coordinates,
    generate_normalized_coordinate_grid, generate_voxel_coordinate_grid,
    integrate_svf,
    interpolate)
from tests.shape_test_util import BroadcastShapeTestingUtil


class CoordinateConversionTests(TestCase):
    """Tests for coordinate conversion"""
    VOXEL_COORDINATES = [
        tensor([0.0, 4]),
        tensor([0.0, 2, 10])
    ]
    NORMALIZED_COORDINATES = [
        tensor([-1.0, 1]),
        tensor([-1.0, 0, 3]),
    ]
    SHAPES = [
        (4, 5),
        (4, 5, 6)
    ]

    def test_conversion_between_coordinates(self) -> None:
        """Test that coordinates are converted correctly"""
        for voxel_coordinates, normalized_coordinates, shape in zip(
                self.VOXEL_COORDINATES,
                self.NORMALIZED_COORDINATES,
                self.SHAPES):
            for voxel_coordinates, normalized_coordinates in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        voxel_coordinates,
                        normalized_coordinates):
                assert_close(
                    convert_normalized_to_voxel_coordinates(normalized_coordinates, shape),
                    voxel_coordinates)
                assert_close(
                    convert_voxel_to_normalized_coordinates(voxel_coordinates, shape),
                    normalized_coordinates)

    def test_inverse_consistency(self) -> None:
        """Check that mappings are inverses"""
        random_coordinates = 2 * rand(2, 3, 4, 5, 6) - 1
        shape = (15, 16, 17)
        assert_close(
            convert_voxel_to_normalized_coordinates(
                convert_normalized_to_voxel_coordinates(2 * random_coordinates, shape),
                shape
            ),
            2 * random_coordinates
        )
        assert_close(
            convert_normalized_to_voxel_coordinates(
                convert_voxel_to_normalized_coordinates(20 * random_coordinates, shape),
                (15, 16, 17)
            ),
            20 * random_coordinates
        )


class CoordinateGridTests(TestCase):
    """Tests for coordinate grid generation"""
    VOXEL_GRID = tensor(
            [
                [
                    [0.0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3]
                ],
                [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4]
                ]
            ]
    )[None]
    NORMALIZED_GRID = tensor(
            [
                [
                    [-1, -1, -1, -1, -1],
                    [-1 / 3, -1 / 3, -1 / 3, -1 / 3, -1 / 3],
                    [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
                    [1, 1, 1, 1, 1]
                ],
                [
                    [-1, -1/2, 0, 1/2, 1],
                    [-1, -1/2, 0, 1/2, 1],
                    [-1, -1/2, 0, 1/2, 1],
                    [-1, -1/2, 0, 1/2, 1]
                ]
            ]
    )[None]
    SHAPE = (4, 5)

    def test_coordinate_grid_generation(self) -> None:
        """Test that coordinate grids are generated correctly"""
        voxel_grid = generate_voxel_coordinate_grid(self.SHAPE, device('cpu'))
        normalized_grid = generate_normalized_coordinate_grid(self.SHAPE, device('cpu'))
        assert_close(
            voxel_grid,
            self.VOXEL_GRID)
        assert_close(
            normalized_grid,
            self.NORMALIZED_GRID)

    def test_consistency_with_conversion(self) -> None:
        """Check that grids are consistent with coordinate conversion"""
        shape = (15, 16, 17)
        voxel_grid = generate_voxel_coordinate_grid(shape, device('cpu'))
        normalized_grid = generate_normalized_coordinate_grid(shape, device('cpu'))
        assert_close(
            convert_voxel_to_normalized_coordinates(voxel_grid),
            normalized_grid
        )
        assert_close(
            convert_normalized_to_voxel_coordinates(normalized_grid),
            voxel_grid
        )


class InterpolationTests(TestCase):
    """Tests for interpolation"""

    GRID_SHAPES = (
        (1, 2, 2, 2),
        (1, 3, 15),
        (3, 3, 15, 16),
        (3, 2, 15, 16, 17),
        (3, 2, 15, 16, 17),
        (3, 2, 15, 16, 17),
        (2,)
    )
    VOLUME_SHAPES = (
        (1, 2, 2, 2),
        (2, 5, 13, 14, 15),
        (3, 2, 13, 14, 15),
        (1, 2, 13, 14),
        (1, 13, 14),
        (3, 5, 7, 13, 14),
        (3, 5, 7, 13, 14),
    )
    TARGET_SHAPES = (
        (1, 2, 2, 2),
        (2, 5, 15),
        (3, 2, 15, 16),
        (3, 2, 15, 16, 17),
        (3, 15, 16, 17),
        (3, 5, 7, 15, 16, 17),
        (3, 5, 7)
    )
    VOLUME = tensor(
        [
            [
                [1.0, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]
            ],
            [
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24]
            ]
        ]
    )
    POINTS = (
        tensor([0.5, 1.5, 2.0]),
        tensor([0.0, 0.0, 0.0]),
        tensor([1.0, 2.0, 3.0])
    )
    VALUES = (
        tensor(
            (7 + 11 + 19 + 23) / 4
        ),
        tensor(1.0),
        tensor(24.0)
    )

    def test_consistency_with_grid_generation(self) -> None:
        """Check that interpolation methods are consistent with grid generation"""
        shape = (15, 16, 17)
        voxel_grid = generate_voxel_coordinate_grid(shape, device('cpu'))
        assert_close(
            interpolate(voxel_grid, grid=voxel_grid),
            voxel_grid
        )

    def test_shape_consistency_for_interpolaton(self) -> None:
        """Check that shapes produced are correct"""
        for grid_shape, volume_shape, target_shape in zip(
                self.GRID_SHAPES,
                self.VOLUME_SHAPES,
                self.TARGET_SHAPES):
            grid = rand(*grid_shape) * 30
            volume = rand(*volume_shape)
            interpolated = interpolate(volume, grid=grid)
            assert_close(
                interpolated.shape,
                target_shape
            )

    def test_correct_values_generated(self) -> None:
        """Check that correct values are interpolated with different shapes"""
        for n_channels in range(1, 3):
            for grid, target in zip(self.POINTS, self.VALUES):
                target = target.expand(n_channels)
                volume = self.VOLUME.expand(n_channels, *self.VOLUME.shape)
                for grid, target in\
                        BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                            grid,
                            target):
                    if target.ndim > 1:
                        batched_volume = volume.expand(target.size(0), *volume.shape)
                    else:
                        batched_volume = volume[None]
                        target = target[None]
                    assert_close(
                        interpolate(batched_volume, grid),
                        target)

    def test_correct_values_generated_without_channels(self) -> None:
        """Check that correct values are interpolated with different shapes
        when volume has no channels"""
        for grid, target in zip(self.POINTS, self.VALUES):
            for grid, target in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        grid,
                        target):
                if target.ndim > 0:
                    batched_volume = self.VOLUME.expand(target.size(0), *self.VOLUME.shape)
                else:
                    batched_volume = self.VOLUME[None]
                    target = target[None]
                assert_close(
                    interpolate(batched_volume, grid),
                    target)


class IntegrateSvfTest(TestCase):
    """Tests for integrating stationary velocity field"""
    def test_correct_output_for_constant_flow(self) -> None:
        """Ensure that constant flow is integrated correctly"""
        flow = ones(1, 3, 64, 64, 64) * 1.72
        assert_close(
            integrate_svf(flow),
            flow
        )
