"""Tests for cubic spline upsampling"""

from unittest import TestCase

from torch import Generator, rand
from torch.autograd import gradcheck

from algorithm.cubic_spline_upsampling import cubic_spline_coefficients


class CubicSplineCoefficientsTests(TestCase):
    """Tests for cubic spline coefficients"""
    def test_cubic_spline_coefficients_grad(self) -> None:
        """Test that gradients are correct for coefficient generation function"""
        generator = Generator().manual_seed(42)
        random_data = rand((2, 5, 5, 6, 7), requires_grad=True, generator=generator)
        self.assertTrue(
            gradcheck(
                func=cubic_spline_coefficients,
                inputs=random_data.double()
            )
        )
