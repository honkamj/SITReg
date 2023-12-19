"""Tests for composable mappings"""

from typing import Optional
from unittest import TestCase

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import eye, matmul, rand, tensor
from torch.testing import assert_close

from algorithm.affine_transformation import convert_to_homogenous_coordinates
from algorithm.composable_mapping.affine import Affine, CPUComposableAffine
from algorithm.composable_mapping.factory import (
    ComposableFactory,
    CoordinateSystemFactory,
)
from algorithm.composable_mapping.grid_mapping import (
    GridCoordinateMapping,
    GridMappingArgs,
    GridVolume,
)
from algorithm.composable_mapping.interface import IRegularGridTensor
from algorithm.composable_mapping.masked_tensor import MaskedTensor, VoxelCoordinateGrid
from algorithm.dense_deformation import generate_voxel_coordinate_grid
from algorithm.interpolator import LinearInterpolator


class _CountingInterpolator(LinearInterpolator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.counter = 0

    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        self.counter += 1
        return super().__call__(volume, coordinates)


class _CountingCPUComposableAffine(CPUComposableAffine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.counter = 0

    def as_matrix(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        self.counter += 1
        return super().as_matrix(device, dtype)


class _CountingVoxelGridTensor(VoxelCoordinateGrid):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.counter = 0

    def generate_values(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        self.counter += 1
        return super().generate_values(device, dtype)


class _CountingMaskedTensor(MaskedTensor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.counter = 0

    def generate_values(
        self, device: Optional[torch_device] = None, dtype: Optional[torch_dtype] = None
    ) -> Tensor:
        self.counter += 1
        return super().generate_values(device, dtype)


class ComposableMappingTests(TestCase):
    """Tests for composable mappings"""

    def test_affine_composition(self) -> None:
        """Test that affine composition works correctly"""
        matrix_1 = tensor(
            [
                [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        matrix_2 = tensor(
            [
                [[2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [0.0, 5.0, -1.0], [0.0, 0.0, 1.0]],
            ]
        )
        input_vector = tensor([[-5.0, -2.0], [3.0, 2.0]])
        expected_output = matmul(
            matmul(matrix_2, matrix_1),
            convert_to_homogenous_coordinates(input_vector)[..., None],
        )[..., :-1, 0]
        transformation_1 = Affine(matrix_1)
        transformation_2 = Affine(matrix_2)
        composition = transformation_2.compose_affine(transformation_1)
        assert_close(composition(input_vector), expected_output)
        assert_close(transformation_2(transformation_1(input_vector)), expected_output)

    def test_cpu_affine_composition(self) -> None:
        """Test that affine composition works correctly"""
        matrix_1 = tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        matrix_2 = tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        input_vector = tensor([[-5.0, -2.0], [3.0, 2.0]])
        expected_output = matmul(
            matmul(matrix_2, matrix_1),
            convert_to_homogenous_coordinates(input_vector)[..., None],
        )[..., :-1, 0]
        expected_inverse_output = matmul(
            matrix_1.inverse(),
            convert_to_homogenous_coordinates(input_vector)[..., None],
        )[..., :-1, 0]
        cpu_composable_1 = CPUComposableAffine(matrix_1)
        cpu_composable_2 = CPUComposableAffine(matrix_2)
        lazy_inverse_1 = cpu_composable_1.invert()
        assert_close(
            cpu_composable_2.compose_affine(cpu_composable_1)(input_vector),
            expected_output,
        )
        assert_close(
            cpu_composable_2.compose_cpu_affine(cpu_composable_1).cache()(input_vector),
            expected_output,
        )
        assert_close(cpu_composable_2(cpu_composable_1(input_vector)), expected_output)
        assert_close(lazy_inverse_1(input_vector), expected_inverse_output)

    def test_cpu_composed_affine_cache(self) -> None:
        """Test caching of CPU composed affine"""
        counting_lazy_transformation = _CountingCPUComposableAffine(eye(2))
        counting_cached_lazy_transformation = counting_lazy_transformation.cache()
        self.assertEqual(counting_lazy_transformation.counter, 0)
        counting_cached_lazy_transformation.as_matrix(torch_device("cpu"))
        self.assertEqual(counting_lazy_transformation.counter, 1)
        counting_cached_lazy_transformation.as_matrix(torch_device("cpu"))
        self.assertEqual(counting_lazy_transformation.counter, 1)

    def test_non_cpu_and_cpu_affine_composition(self) -> None:
        """Test that affine composition works correctly"""
        matrix_1 = tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        matrix_2 = tensor(
            [
                [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        input_vector = tensor([[-5.0, -2.0], [3.0, 2.0]])
        expected_output = matmul(
            matmul(matrix_2, matrix_1),
            convert_to_homogenous_coordinates(input_vector)[..., None],
        )[..., :-1, 0]
        cpu_composable_1 = CPUComposableAffine(matrix_1)
        transformation_2 = Affine(matrix_2)
        composition = transformation_2.compose_affine(cpu_composable_1)
        assert_close(composition(input_vector), expected_output)
        assert_close(transformation_2(cpu_composable_1(input_vector)), expected_output)

    def test_grid_volume(self) -> None:
        """Test that grid volumes work correctly"""
        data = tensor(
            [
                [[1.0, 0.0, 5.0], [0.0, -2.0, 0.0], [2.0, -3.0, 4.0], [2.0, 0.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 4.0, 0.0], [-1.0, 5.0, 3.0], [-2.0, 0.0, 1.0]],
            ]
        )[None]
        mask = tensor(
            [[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
        )[None]
        interpolator = _CountingInterpolator(padding_mode="border")
        volume = GridVolume(
            data=data,
            mask=mask,
            n_channel_dims=1,
            grid_mapping_args=GridMappingArgs(
                interpolator=interpolator, mask_outside_fov=True, mask_threshold=1.0
            ),
        )
        input_points = (
            tensor([1.0, 1.0]),
            tensor([2.5, 2.0])[None, ..., None, None],
            tensor([2.5, 2.01])[None, ..., None, None],
        )
        output_points = (
            tensor([-2.0, 4.0])[None],
            tensor([2.5, 2.0])[None, ..., None, None],
            tensor([2.5, 2.0])[None, ..., None, None],
        )
        output_masks = (
            tensor([0.0])[None],
            tensor([1.0])[None, ..., None, None],
            tensor([0.0])[None, ..., None, None],
        )
        for input_point, expected_output, expected_mask in zip(
            input_points, output_points, output_masks
        ):
            output = volume(MaskedTensor(input_point))
            assert_close(
                output.generate_values(data.device, data.dtype), expected_output
            )
            assert_close(output.mask, expected_mask)
        count_before = interpolator.counter
        assert_close(
            data,
            volume(VoxelCoordinateGrid((4, 3))).generate_values(
                data.device, data.dtype
            ),
        )
        self.assertEqual(interpolator.counter, count_before)
        volume(
            VoxelCoordinateGrid((3, 2)).apply_affine(CPUComposableAffine(1.1 * eye(3)))
        )
        self.assertEqual(
            interpolator.counter,
            count_before + 2,  # Both mask and the main volume are interpolated
        )

    def test_grid_mapping(self) -> None:
        """Test that grid volumes work correctly"""
        data = tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [-2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0],
                ],
            ]
        )[None]
        mask = tensor(
            [[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
        )[None]
        interpolator = _CountingInterpolator(padding_mode="border")
        mapping = GridCoordinateMapping(
            displacement_field=data,
            mask=mask,
            grid_mapping_args=GridMappingArgs(
                interpolator=interpolator, mask_outside_fov=True, mask_threshold=1.0
            ),
        )
        input_points = (
            tensor([0.3, 1.5]),
            tensor([2.0, 1.5])[None, ..., None, None],
            tensor([2.0, 2.01])[None, ..., None, None],
        )
        output_points = (
            tensor([0.3, 1.5])[None],
            tensor([3.0, -0.5])[None, ..., None, None],
            tensor([3.0, 0.01])[None, ..., None, None],
        )
        output_masks = (
            tensor([0.0])[None],
            tensor([1.0])[None, ..., None, None],
            tensor([0.0])[None, ..., None, None],
        )
        for input_point, expected_output, expected_mask in zip(
            input_points, output_points, output_masks
        ):
            output = mapping(MaskedTensor(input_point))
            assert_close(
                output.generate_values(data.device, data.dtype), expected_output
            )
            assert_close(output.mask, expected_mask)
        count_before = interpolator.counter
        assert_close(
            data + generate_voxel_coordinate_grid((4, 3), data.device),
            mapping(VoxelCoordinateGrid((4, 3))).generate_values(
                data.device, data.dtype
            ),
        )
        self.assertEqual(interpolator.counter, count_before)
        mapping(
            VoxelCoordinateGrid((3, 2)).apply_affine(CPUComposableAffine(1.1 * eye(3)))
        )
        self.assertEqual(
            interpolator.counter,
            count_before + 2,  # Both mask and the main volume are interpolated
        )

    def test_voxel_grid_caching(self) -> None:
        """Test that voxel grid tensors are cached correctly"""
        translation_matrix = eye(3)
        translation_matrix[0, -1] = 3
        translation_matrix[1, -1] = 3
        voxel_grid = _CountingVoxelGridTensor((5, 4), Affine(translation_matrix))
        self.assertEqual(voxel_grid.counter, 0)
        middle_generated_values = voxel_grid.generate_values(
            torch_device("cpu"), translation_matrix.dtype
        )
        assert_close(
            middle_generated_values,
            3 + generate_voxel_coordinate_grid((5, 4), torch_device("cpu")),
        )
        self.assertEqual(voxel_grid.counter, 1)
        voxel_grid.generate_values(torch_device("cpu"), translation_matrix.dtype)
        self.assertEqual(voxel_grid.counter, 2)
        cached_voxel_grid = voxel_grid.cache()
        cached_voxel_grid.generate_values(torch_device("cpu"), translation_matrix.dtype)
        self.assertEqual(voxel_grid.counter, 3)
        cached_voxel_grid.generate_values(torch_device("cpu"), translation_matrix.dtype)
        self.assertEqual(voxel_grid.counter, 3)
        affine_matrix = eye(3)
        affine_matrix[0, 0] = 2
        affine_matrix[1, 1] = 2
        affine_transformed = cached_voxel_grid.apply_affine(Affine(affine_matrix))
        generated_values = affine_transformed.generate_values(
            torch_device("cpu"), translation_matrix.dtype
        )
        self.assertEqual(voxel_grid.counter, 3)
        assert_close(
            generated_values,
            2 * (3 + generate_voxel_coordinate_grid((5, 4), torch_device("cpu"))),
        )
        self.assertTrue(isinstance(affine_transformed, IRegularGridTensor))

    def test_masked_tensor_caching(self) -> None:
        """Test that voxel grid tensors are cached correctly"""
        translation_matrix = eye(3)
        translation_matrix[0, -1] = 3
        translation_matrix[1, -1] = 3
        input_tensor = rand((5, 2))
        masked_tensor = _CountingMaskedTensor(
            input_tensor, affine_transformation=Affine(translation_matrix)
        )
        self.assertEqual(masked_tensor.counter, 0)
        middle_generated_values = masked_tensor.generate_values(
            torch_device("cpu"), translation_matrix.dtype
        )
        assert_close(middle_generated_values, 3 + input_tensor)
        self.assertEqual(masked_tensor.counter, 1)
        masked_tensor.generate_values(torch_device("cpu"), translation_matrix.dtype)
        self.assertEqual(masked_tensor.counter, 2)
        cached_masked_tensor = masked_tensor.cache()
        cached_masked_tensor.generate_values(
            torch_device("cpu"), translation_matrix.dtype
        )
        self.assertEqual(masked_tensor.counter, 3)
        cached_masked_tensor.generate_values(
            torch_device("cpu"), translation_matrix.dtype
        )
        self.assertEqual(masked_tensor.counter, 3)
        affine_matrix = eye(3)
        affine_matrix[0, 0] = 2
        affine_matrix[1, 1] = 2
        affine_transformed = cached_masked_tensor.apply_affine(Affine(affine_matrix))
        generated_values = affine_transformed.generate_values(
            torch_device("cpu"), translation_matrix.dtype
        )
        self.assertEqual(masked_tensor.counter, 3)
        assert_close(generated_values, 2 * (3 + input_tensor))

    def test_slice_generation_for_voxel_grids(self) -> None:
        """Test that correct slices are generated"""
        voxel_grid_1 = VoxelCoordinateGrid(
            (10, 20),
            affine_transformation=CPUComposableAffine(
                tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
            ),
        )
        self.assertEqual(
            voxel_grid_1.reduce_to_slice((30, 30)),
            (..., slice(2, 12, 1), slice(3, 23, 1)),
        )
        voxel_grid_2 = VoxelCoordinateGrid(
            (10, 20),
            affine_transformation=CPUComposableAffine(
                tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            ),
        )
        self.assertEqual(
            voxel_grid_2.reduce_to_slice((10, 20)),
            (..., slice(0, 10, 1), slice(0, 20, 1)),
        )


class ComposableFactoryTests(TestCase):
    """Test different mappings together"""

    def test_coordinate_transformed_grid_volume(self) -> None:
        """Test coordinate transformed grid volume"""
        data = tensor(
            [
                [[1.0, 0.0, 5.0], [0.0, -2.0, 0.0], [2.0, -3.0, 4.0], [2.0, 0.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 4.0, 0.0], [-1.0, 5.0, 3.0], [-2.0, 0.0, 1.0]],
            ]
        )[None]
        mask = tensor(
            [[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
        )[None]
        interpolator = _CountingInterpolator(padding_mode="border")
        coordinate_system = CoordinateSystemFactory.centered_normalized(
            original_grid_shape=(4, 3), voxel_size=(1.0, 2.0)
        )
        volume = ComposableFactory.create_volume(
            data=data,
            coordinate_system=coordinate_system,
            mask=mask,
            grid_mapping_args=GridMappingArgs(
                interpolator=interpolator, mask_outside_fov=True, mask_threshold=1.0
            ),
            n_channel_dims=1,
        )
        input_points = (
            tensor([-1 / 2, 0.0]),
            tensor([-1 / 6, -1 / 3])[None, ..., None, None],
        )
        output_points = (
            tensor([0.0, 1.0])[None],
            tensor([-1.0, 2.0])[None, ..., None, None],
        )
        output_masks = (tensor([1.0])[None], tensor([0.0])[None, ..., None, None])
        for input_point, expected_output, expected_mask in zip(
            input_points, output_points, output_masks
        ):
            output = volume(MaskedTensor(input_point))
            assert_close(
                output.generate_values(data.device, data.dtype), expected_output
            )
            assert_close(output.mask, expected_mask)
        middle_coordinate_system = CoordinateSystemFactory.centered_normalized(
            original_grid_shape=(4, 3), voxel_size=(1.0, 2.0), grid_shape=(2, 3)
        )
        count_before = interpolator.counter
        assert_close(
            data[:, :, 1:-1],
            volume(middle_coordinate_system.grid).generate_values(
                data.device, data.dtype
            ),
        )
        self.assertEqual(interpolator.counter, count_before)

    def test_coordinate_transformed_grid_mapping(self) -> None:
        """Test that grid mappings work correctly"""
        data = tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [-2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0],
                ],
            ]
        )[None]
        mask = tensor(
            [[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
        )[None]
        interpolator = _CountingInterpolator(padding_mode="border")
        coordinate_system = CoordinateSystemFactory.centered_normalized(
            original_grid_shape=(4, 3), voxel_size=(1.0, 2.0)
        )
        mapping = ComposableFactory.create_dense_mapping(
            displacement_field=data,
            coordinate_system=coordinate_system,
            mask=mask,
            grid_mapping_args=GridMappingArgs(
                interpolator=interpolator, mask_outside_fov=True, mask_threshold=1.0
            ),
        )
        input_points = (
            tensor([-1 / 2, 0.0]),
            tensor([-1 / 2 + 1e-4, 0.0]),
            tensor([1 / 6, 1 / 3])[None, ..., None, None],
        )
        output_points = (
            tensor([-1 / 2, 0.0])[None],
            tensor([-1 / 2 + 1e-4, 0.0])[None],
            tensor([1 / 2, -1])[None, ..., None, None],
        )
        output_masks = (
            tensor([1.0])[None],
            tensor([0.0])[None],
            tensor([1.0])[None, ..., None, None],
        )
        for input_point, expected_output, expected_mask in zip(
            input_points, output_points, output_masks
        ):
            output = mapping(MaskedTensor(input_point))
            assert_close(
                output.generate_values(data.device, data.dtype), expected_output
            )
            assert_close(output.mask, expected_mask)
        middle_coordinate_system = CoordinateSystemFactory.centered_normalized(
            original_grid_shape=(4, 3), voxel_size=(1.0, 2.0), grid_shape=(2, 3)
        )
        count_before = interpolator.counter
        grid_values = mapping(middle_coordinate_system.grid)
        assert_close(
            (data + generate_voxel_coordinate_grid((4, 3), data.device))[..., 1:-1, :],
            coordinate_system.to_voxel_coordinates(grid_values).generate_values(
                data.device, data.dtype
            ),
        )
        self.assertEqual(interpolator.counter, count_before)

    def test_centered_coordinate_system_consistency(self) -> None:
        """Test that coordinate system is consistent"""
        shape = (5, 6)
        coordinate_systems = [
            CoordinateSystemFactory.centered_normalized(
                original_grid_shape=(13, 4), voxel_size=(3.0, 2.05), grid_shape=shape
            ),
            CoordinateSystemFactory.centered_normalized(
                original_grid_shape=(13, 4),
                voxel_size=(3.0, 2.05),
                grid_shape=shape,
                downsampling_factor=(1.2, 0.54, 17.3),
            ),
            CoordinateSystemFactory.centered_normalized(
                original_grid_shape=shape, voxel_size=(1.0, 2.05)
            ),
        ]
        for coordinate_system in coordinate_systems:
            voxel_grid = generate_voxel_coordinate_grid(shape, torch_device("cpu"))
            assert_close(
                coordinate_system.to_voxel_coordinates(
                    coordinate_system.grid
                ).generate_values(torch_device("cpu"), voxel_grid.dtype),
                voxel_grid,
            )
            assert_close(
                coordinate_system.to_voxel_coordinates(coordinate_system.grid)
                .cache()
                .generate_values(torch_device("cpu"), voxel_grid.dtype),
                voxel_grid,
            )
            assert_close(
                coordinate_system.from_voxel_coordinates(
                    coordinate_system.to_voxel_coordinates(coordinate_system.grid)
                ).generate_values(torch_device("cpu"), voxel_grid.dtype),
                coordinate_system.grid.generate_values(
                    torch_device("cpu"), voxel_grid.dtype
                ),
            )
            assert_close(
                coordinate_system.from_voxel_coordinates(
                    coordinate_system.to_voxel_coordinates(
                        coordinate_system.grid
                    ).cache()
                )
                .cache()
                .generate_values(torch_device("cpu"), voxel_grid.dtype),
                coordinate_system.grid.generate_values(
                    torch_device("cpu"), voxel_grid.dtype
                ),
            )

    def test_top_left_aligned_coordinate_system_consistency(self) -> None:
        """Test that coordinate system is consistent"""
        shape = (5, 6)
        original_grid_shape = (16, 18)
        coordinate_systems = [
            CoordinateSystemFactory.top_left_aligned_normalized(
                original_grid_shape=original_grid_shape,
                grid_shape=shape,
                voxel_size=(3.0, 2.05),
                downsampling_factor=[1.0, 0.7],
            ),
            CoordinateSystemFactory.top_left_aligned_normalized(
                original_grid_shape=original_grid_shape,
                grid_shape=shape,
                voxel_size=(1.0, 2.05),
                downsampling_factor=[2.0, 4.0],
            ),
        ]
        for coordinate_system in coordinate_systems:
            voxel_grid = generate_voxel_coordinate_grid(shape, torch_device("cpu"))
            assert_close(
                coordinate_system.to_voxel_coordinates(
                    coordinate_system.grid
                ).generate_values(torch_device("cpu"), voxel_grid.dtype),
                voxel_grid,
            )
            assert_close(
                coordinate_system.to_voxel_coordinates(coordinate_system.grid)
                .cache()
                .generate_values(torch_device("cpu"), voxel_grid.dtype),
                voxel_grid,
            )
            assert_close(
                coordinate_system.from_voxel_coordinates(
                    coordinate_system.to_voxel_coordinates(coordinate_system.grid)
                ).generate_values(torch_device("cpu"), voxel_grid.dtype),
                coordinate_system.grid.generate_values(
                    torch_device("cpu"), voxel_grid.dtype
                ),
            )
            assert_close(
                coordinate_system.from_voxel_coordinates(
                    coordinate_system.to_voxel_coordinates(
                        coordinate_system.grid
                    ).cache()
                )
                .cache()
                .generate_values(torch_device("cpu"), voxel_grid.dtype),
                coordinate_system.grid.generate_values(
                    torch_device("cpu"), voxel_grid.dtype
                ),
            )

    def test_top_left_aligned_and_centered_coordinate_system_consistency(self) -> None:
        """Test that coordinate systems are consistent between themselves"""
        centered_coordinate_systems = [
            CoordinateSystemFactory.centered_normalized(
                original_grid_shape=(17, 14), voxel_size=(1.2, 2.05)
            ),
            CoordinateSystemFactory.centered_normalized(
                original_grid_shape=(8, 6), voxel_size=(3.1, 2.7)
            ),
            CoordinateSystemFactory.centered_normalized(
                original_grid_shape=(2, 3), voxel_size=(3.0, 1.0)
            ),
        ]
        top_left_aligned_coordinate_systems = [
            CoordinateSystemFactory.top_left_aligned_normalized(
                original_grid_shape=(17, 14),
                grid_shape=(17, 14),
                voxel_size=(1.2, 2.05),
                downsampling_factor=[1.0, 1.0],
            ),
            CoordinateSystemFactory.top_left_aligned_normalized(
                original_grid_shape=(16, 12),
                grid_shape=(8, 6),
                voxel_size=(3.1, 2.7),
                downsampling_factor=[2.0, 2.0],
            ),
            CoordinateSystemFactory.top_left_aligned_normalized(
                original_grid_shape=(6, 6),
                grid_shape=(2, 3),
                voxel_size=(2.0, 1.0),
                downsampling_factor=[3.0, 2.0],
            ),
        ]
        for centered_coordinate_system, top_left_aligned_coordinate_system in zip(
            centered_coordinate_systems, top_left_aligned_coordinate_systems
        ):
            centered = centered_coordinate_system.grid.generate_values(
                torch_device("cpu")
            )
            top_left_aligned = top_left_aligned_coordinate_system.grid.generate_values(
                torch_device("cpu")
            )
            assert_close(centered, top_left_aligned)

    def test_top_left_aligned_coordinate_system(self) -> None:
        """Test that coordinate system is correct"""
        coordinate_systems = [
            CoordinateSystemFactory.top_left_aligned_normalized(
                original_grid_shape=(4, 3),
                grid_shape=(3, 2),
                voxel_size=(3.0, 2.0),
                downsampling_factor=[1.0, 1.0],
            ),
            CoordinateSystemFactory.top_left_aligned_normalized(
                original_grid_shape=(4, 3),
                grid_shape=(4, 3),
                voxel_size=(3.0, 2.0),
                downsampling_factor=[3.0, 2.0],
            ),
        ]
        correct_grids = [
            tensor(
                [
                    [[-3 / 4, -3 / 4], [-1 / 4, -1 / 4], [1 / 4, 1 / 4]],
                    [[-1 / 3, 0], [-1 / 3, 0], [-1 / 3, 0]],
                ]
            ),
            tensor(
                [
                    [
                        [-1 / 4, -1 / 4, -1 / 4],
                        [5 / 4, 5 / 4, 5 / 4],
                        [11 / 4, 11 / 4, 11 / 4],
                        [17 / 4, 17 / 4, 17 / 4],
                    ],
                    [
                        [-1 / 6, 1 / 2, 7 / 6],
                        [-1 / 6, 1 / 2, 7 / 6],
                        [-1 / 6, 1 / 2, 7 / 6],
                        [-1 / 6, 1 / 2, 7 / 6],
                    ],
                ]
            ),
        ]
        for coordinate_system, grid in zip(coordinate_systems, correct_grids):
            generated_grid = coordinate_system.grid.generate_values(torch_device("cpu"))
            assert_close(generated_grid, grid[None])
