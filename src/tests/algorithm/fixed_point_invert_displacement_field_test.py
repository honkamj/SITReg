"""Tests for fixed point inversion of displacement field"""

from functools import partial
from unittest import TestCase

from torch import Generator, rand, zeros_like
from torch.autograd import gradcheck
from torch.testing import assert_close

from algorithm.dense_deformation import generate_voxel_coordinate_grid
from algorithm.fixed_point_invert_displacement_field import (
    DisplacementFieldInversionArguments,
    fixed_point_invert_displacement_field)
from algorithm.fixed_point_solver import (AndersonSolver,
                                          MaxElementWiseAbsStopCriterion)
from algorithm.interpolator import LinearInterpolator


class FixedPointInversionTests(TestCase):
    """Tests for fixed point inversion"""

    def test_composition_zero(self) -> None:
        """Test that composition is zero"""
        generator = Generator().manual_seed(1337)
        test_input = (2 * rand(1, 2, 5, 5, generator=generator) - 1) * 0.25
        interpolator = LinearInterpolator()
        inverted = fixed_point_invert_displacement_field(
            test_input,
            arguments=DisplacementFieldInversionArguments(
                interpolator=LinearInterpolator(),
                forward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(max_error=1e-6)),
                backward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(max_error=1e-6))
            )
        )
        composition = interpolator(
            test_input,
            generate_voxel_coordinate_grid((5, 5), test_input.device) + inverted
        ) + inverted
        assert_close(
            composition,
            zeros_like(composition)
        )

    def test_grad(self) -> None:
        """Test that gradients are correct"""
        generator = Generator().manual_seed(1337)
        test_input = (2 * rand(1, 2, 5, 5, generator=generator) - 1) * 0.25
        fixed_point_invert_displacement_field_ = partial(
            fixed_point_invert_displacement_field,
            arguments=DisplacementFieldInversionArguments(
                interpolator=LinearInterpolator(),
                forward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(max_error=1e-6)
                ),
                backward_solver=AndersonSolver(
                    stop_criterion=MaxElementWiseAbsStopCriterion(max_error=1e-6)
                )
            )
        )
        self.assertTrue(
            gradcheck(
                fixed_point_invert_displacement_field_,
                test_input.double().requires_grad_(),
                eps=1e-5,
                atol=1e-5,
                check_undefined_grad=False
            )
        )
