"""NDimensional helpers for building NDimensional models"""


import torch.nn


def conv_nd(n_dims: int):
    """Return any dimensional convolution module"""
    return getattr(torch.nn, f"Conv{n_dims}d")


def conv_nd_function(n_dims: int):
    """Return any dimensional convolution function"""
    return getattr(torch.nn.functional, f"conv{n_dims}d")


def conv_transpose_nd(n_dims: int):
    """Return any dimensional tranposed convolution module"""
    return getattr(torch.nn, f"ConvTranspose{n_dims}d")


def conv_transpose_nd_function(n_dims: int):
    """Return any dimensional tranposed convolution function"""
    return getattr(torch.nn.functional, f"conv_transpose{n_dims}d")


def avg_pool_nd(n_dims: int):
    """Return any dimensional average pooling module"""
    return getattr(torch.nn, f"AvgPool{n_dims}d")


def avg_pool_nd_function(n_dims: int):
    """Return any dimensional average pooling function"""
    return getattr(torch.nn.functional, f"avg_pool{n_dims}d")


def max_pool_nd(n_dims: int):
    """Return any dimensional max pooling module"""
    return getattr(torch.nn, f"MaxPool{n_dims}d")


def max_pool_nd_function(n_dims: int):
    """Return any dimensional max pooling function"""
    return getattr(torch.nn.functional, f"max_pool{n_dims}d")


def instance_norm_nd(n_dims: int):
    """Return any dimensional instance normalization module"""
    return getattr(torch.nn, f"InstanceNorm{n_dims}d")
