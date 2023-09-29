"""Tests for fixed point inversion of displacement field"""

from unittest import TestCase

from torch import Generator, Tensor, float64, rand
from torch.autograd import gradcheck
from torch.linalg import svdvals
from torch.testing import assert_close

from algorithm.fixed_point_solver import NaiveSolver
from algorithm.spectral_norm import (
    SpectralNormMaxElementWiseAbsStopCriterion,
    calculate_spectral_norm_with_power_iteration,
)


class SpectralNormPowerIterationTests(TestCase):
    """Tests for spectral norm computation using power iteration"""

    def test_forward(self) -> None:
        """Test that correct value is returned"""
        generator = Generator().manual_seed(1337)
        test_input = 2 * rand(8, 3, 3, generator=generator, dtype=float64) - 1
        spectral_norm = calculate_spectral_norm_with_power_iteration(
            matrix=test_input,
            solver=NaiveSolver(
                stop_criterion=SpectralNormMaxElementWiseAbsStopCriterion(
                    n_rows=3,
                    max_iterations=200,
                    min_iterations=1,
                    max_error=1e-4,
                    check_convergence_every_nth_iteration=10,
                )
            ),
        )
        correct_spectral_norm = svdvals(test_input)[..., 0, None]
        assert_close(spectral_norm, correct_spectral_norm)

    def test_backward(self) -> None:
        """Test that gradients are correct"""
        generator = Generator().manual_seed(1337)
        test_input = 2 * rand(8, 3, 3, generator=generator, dtype=float64) - 1

        def spectral_norm_power_iteration_(matrix: Tensor) -> Tensor:
            return calculate_spectral_norm_with_power_iteration(
                matrix=matrix,
                solver=NaiveSolver(
                    stop_criterion=SpectralNormMaxElementWiseAbsStopCriterion(
                        n_rows=3,
                        max_iterations=200,
                        min_iterations=1,
                        max_error=1e-7,
                        check_convergence_every_nth_iteration=10,
                    )
                ),
            )

        self.assertTrue(
            gradcheck(
                spectral_norm_power_iteration_,
                test_input.requires_grad_(),
                eps=1e-9,
                atol=1e-4,
                check_undefined_grad=False,
            )
        )
