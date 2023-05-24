"""Counting parameters of modules"""

from torch.nn import Module


def count_module_trainable_parameters(module: Module):
    """Counts trainable parameters of a module"""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
