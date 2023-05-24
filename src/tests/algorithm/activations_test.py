"""Tests for custom activation function implementations"""

from unittest import TestCase

from torch import rand
from torch.autograd import gradcheck

from algorithm.activations import (memory_efficient_leaky_relu,
                                   memory_efficient_relu)


class ActivationTests(TestCase):
    """Tests for custom activation implementations"""
    def test_relu_grad_correct(self):
        """Test that gradients are correct for custom relu implementation"""
        random_volume = (2 * rand(5, 6, 7, requires_grad=True) - 1).double()
        gradcheck(
            memory_efficient_relu,
            inputs=random_volume,
            eps=1e-6
        )

    def test_leaky_relu_grad_correct(self):
        """Test that gradients are correct for custom leaky relu implementation"""
        random_volume = (2 * rand(5, 6, 7, requires_grad=True) - 1).double()
        gradcheck(
            memory_efficient_leaky_relu,
            inputs=random_volume,
            eps=1e-6
        )
