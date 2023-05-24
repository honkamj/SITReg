"""Tests for affine transformation primitive functions"""

from math import pi
from typing import Sequence
from unittest import TestCase

from torch import diag, exp, eye, tensor, zeros
from torch.testing import assert_close

from algorithm.affine_transformation import (
    AffineTransformationTypeDefinition, calculate_n_dims,
    calculate_n_parameters, compose_affine_transformation_matrices,
    convert_to_homogenous_coordinates, embed_transformation,
    generate_affine_transformation_matrix, generate_rotation_matrix,
    generate_scale_and_shear_matrix, generate_scale_matrix,
    generate_translation_matrix)
from tests.shape_test_util import BroadcastShapeTestingUtil


class AffineSpaceDimensionalityTests(TestCase):
    """Tests for affine space dimensionality"""
    def _test_calculate_n_parameters_and_n_dims(
            self,
            affine_type: AffineTransformationTypeDefinition,
            n_dims_sequence: Sequence[int],
            n_parameters_sequence: Sequence[int]) -> None:
        for n_dims, n_parameters in zip(n_dims_sequence, n_parameters_sequence):
            self.assertEqual(
                calculate_n_dims(n_parameters, affine_type),
                n_dims
            )
            self.assertEqual(
                calculate_n_parameters(n_dims, affine_type),
                n_parameters
            )

    def test_calculate_n_parameters_and_n_dims(self) -> None:
        """Test correct dimensionalities"""
        self._test_calculate_n_parameters_and_n_dims(
            AffineTransformationTypeDefinition.full(),
            [1, 2, 3, 4],
            [2, 6, 12, 20]
        )
        self._test_calculate_n_parameters_and_n_dims(
            AffineTransformationTypeDefinition.only_rotation(),
            [1, 2, 3, 4],
            [0, 1, 3, 6]
        )
        self._test_calculate_n_parameters_and_n_dims(
            AffineTransformationTypeDefinition.only_scale(),
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        )
        self._test_calculate_n_parameters_and_n_dims(
            AffineTransformationTypeDefinition.only_shear(),
            [1, 2, 3, 4],
            [0, 1, 3, 6]
        )
        self._test_calculate_n_parameters_and_n_dims(
            AffineTransformationTypeDefinition.only_translation(),
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        )
        self._test_calculate_n_parameters_and_n_dims(
            AffineTransformationTypeDefinition(False, False, True, True),
            [1, 2, 3, 4],
            [1, 3, 6, 10]
        )


class MatrixEmbeddingTests(TestCase):
    """Tests matrix embedding"""
    INPUTS = [
        (
            tensor(
                [[1.0, 2], [3, 4]]
            ),
            (4, 4)
        ),
        (
            tensor(
                [[1.0, 2], [3, 4]]
            ),
            (4, 3)
        ),
        (
            tensor(
                [[1.0, 2], [3, 4]]
            ),
            (3, 4)
        )
    ]
    OUTPUTS = [
        tensor(
            [[1.0, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        tensor(
            [[1.0, 2, 0], [3, 4, 0], [0, 0, 1], [0, 0, 0]]
        ),
        tensor(
            [[1.0, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 0]]
        )
    ]

    def test_embedding(self) -> None:
        """Test that matrices are embedded correctly"""
        for (input_matrix, target_shape), output_matrix in zip(self.INPUTS, self.OUTPUTS):
            for input_matrix, output_matrix in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        input_matrix,
                        output_matrix):
                assert_close(
                    embed_transformation(input_matrix, target_shape),
                    output_matrix)


class HomogenousCoordinateTests(TestCase):
    """Tests for homogenous coordinate conversions"""
    INPUTS = [
        tensor(
            [1.0, 2]
        ),
        tensor(
            [1.0]
        )
    ]
    OUTPUTS = [
        tensor(
            [1.0, 2, 1]
        ),
        tensor(
            [1.0, 1]
        )
    ]

    def test_generation(self) -> None:
        """Test that vectors are embedded correctly"""
        for input_vector, output_vector in zip(self.INPUTS, self.OUTPUTS):
            for input_vector, output_vector in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        input_vector,
                        output_vector):
                assert_close(
                    convert_to_homogenous_coordinates(input_vector),
                    output_vector)


class TranslationMatrixTests(TestCase):
    """Tests for translation matrix generation"""
    INPUTS = [
        tensor(
            [1.0, 2]
        ),
        tensor(
            [1.0]
        )
    ]
    OUTPUTS = [
        tensor(
            [[1.0, 0, 1], [0, 1, 2], [0, 0, 1]]
        ),
        tensor(
            [[1.0, 1.0], [0, 1]]
        )
    ]

    def test_generation(self) -> None:
        """Test that matrices are generated correctly"""
        for input_parameters, output_matrix in zip(self.INPUTS, self.OUTPUTS):
            for input_parameters, output_matrix in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        input_parameters,
                        output_matrix):
                assert_close(
                    generate_translation_matrix(input_parameters),
                    output_matrix)


class RotationMatrixTests(TestCase):
    """Tests for rotation matrix generation"""
    INPUTS = [
        tensor(
            [0.0, 0, 0]
        ),
        tensor(
            [pi / 2]
        )
    ]
    OUTPUTS = [
        tensor(
            [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]
        ),
        tensor(
            [[0.0, 1], [-1, 0]]
        )
    ]

    def test_generation(self) -> None:
        """Test that matrices are converted correctly"""
        for input_parameters, output_matrix in zip(self.INPUTS, self.OUTPUTS):
            for input_parameters, output_matrix in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        input_parameters,
                        output_matrix):
                assert_close(
                    generate_rotation_matrix(input_parameters),
                    output_matrix)


class ScaleAndShearMatrixTests(TestCase):
    """Tests for scale and shear matrix generation"""
    INPUTS = [
        tensor(
            [2.0, 2.0, 0.0]
        ),
        tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
    ]
    OUTPUTS = [
        diag(exp(tensor([2.0, 2.0]))),
        eye(3)
    ]

    def test_generation(self) -> None:
        """Test that matrices are generated correctly"""
        for input_parameters, output_matrix in zip(self.INPUTS, self.OUTPUTS):
            for input_parameters, output_matrix in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        input_parameters,
                        output_matrix):
                assert_close(
                    generate_scale_and_shear_matrix(input_parameters),
                    output_matrix)


class ScaleMatrixTests(TestCase):
    """Tests for scale matrix generation"""
    INPUTS = [
        tensor(
            [2.0, 2.0]
        ),
        tensor(
            [0.0, 0.0, 1.2]
        )
    ]
    OUTPUTS = [
        diag(exp(tensor([2.0, 2.0]))),
        diag(exp(tensor([0.0, 0.0, 1.2])))
    ]

    def test_generation(self) -> None:
        """Test that matrices are generated correctly"""
        for input_parameters, output_matrix in zip(self.INPUTS, self.OUTPUTS):
            for input_parameters, output_matrix in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        input_parameters,
                        output_matrix):
                assert_close(
                    generate_scale_matrix(input_parameters.exp()),
                    output_matrix)


class AffineMatrixTests(TestCase):
    """Tests for generic affine matrix generation"""
    INPUTS = [
        (
            zeros(12),
            AffineTransformationTypeDefinition.full()
        ),
        (
            zeros(6),
            AffineTransformationTypeDefinition.full()
        )
    ]
    OUTPUTS = [
        eye(4),
        eye(3)
    ]

    def test_generation(self) -> None:
        """Test that matrices are generated correctly"""
        for (input_parameters, affine_type), output_matrix in zip(self.INPUTS, self.OUTPUTS):
            for input_parameters, output_matrix in\
                    BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                        input_parameters,
                        output_matrix):
                assert_close(
                    generate_affine_transformation_matrix(input_parameters, affine_type),
                    output_matrix)


class AffineTransformationCompositionTests(TestCase):
    """Tests for affine transformation composition"""
    INPUT_1 = tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0]
        ]
    )
    INPUT_2 = tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0]
        ]
    )
    OUTPUT = tensor(
        [
            [1.0, 0.0, 4.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0]
        ]
    )

    def test_composition(self) -> None:
        """Test that matrices are composed correctly"""
        for input_1, input_2, output in\
                BroadcastShapeTestingUtil.expand_tensor_shapes_for_testing(
                    self.INPUT_1,
                    self.INPUT_2,
                    self.OUTPUT):
            assert_close(
                compose_affine_transformation_matrices(input_1, input_2),
                output)
