"""Compute non-diffeomorphic volume (NDV) from a given deformation field."""

from typing import Mapping

from torch import Tensor, cat, stack, tensor
from torch.linalg import det
from torch.nn.functional import conv3d, pad

from algorithm.dense_deformation import generate_voxel_coordinate_grid
from util.optional import optional_add


def calculate_jacobian_determinants(ddf: Tensor) -> Mapping[str, Tensor]:
    """Calculate jacobian determinants of the given displacement field.

    Modified from https://github.com/yihao6/digital_diffeomorphism

    Liu, Yihao, et al. "On finite difference jacobian computation in deformable image registration."
    International Journal of Computer Vision (2024): 1-11."""
    if ddf.size(1) != 3 or ddf.ndim != 5:
        raise ValueError(
            "Currently only supports 3D deformation fields. "
            "The input shape must be (batch_size, 3, dim_1, dim_2, dim_3)."
        )

    grid = generate_voxel_coordinate_grid(ddf.shape[2:], dtype=ddf.dtype, device=ddf.device)
    trans = ddf + grid

    kernels = {}
    kwargs = {"dtype": trans.dtype, "device": trans.device}
    kernels["D0x"] = tensor([-0.5, 0, 0.5], **kwargs).view(1, 1, 3, 1, 1)
    kernels["D+x"] = tensor([0, -1, 1], **kwargs).view(1, 1, 3, 1, 1)
    kernels["D-x"] = tensor([-1, 1, 0], **kwargs).view(1, 1, 3, 1, 1)

    kernels["D0y"] = tensor([-0.5, 0, 0.5], **kwargs).view(1, 1, 1, 3, 1)
    kernels["D+y"] = tensor([0, -1, 1], **kwargs).view(1, 1, 1, 3, 1)
    kernels["D-y"] = tensor([-1, 1, 0], **kwargs).view(1, 1, 1, 3, 1)

    kernels["D0z"] = tensor([-0.5, 0, 0.5], **kwargs).view(1, 1, 1, 1, 3)
    kernels["D+z"] = tensor([0, -1, 1], **kwargs).view(1, 1, 1, 1, 3)
    kernels["D-z"] = tensor([-1, 1, 0], **kwargs).view(1, 1, 1, 1, 3)

    # J1*
    kernels["1*xy"] = tensor(
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        **kwargs,
    ).reshape(1, 1, 3, 3, 1)
    kernels["1*xz"] = tensor(
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        **kwargs,
    ).reshape(1, 1, 3, 1, 3)
    kernels["1*yz"] = tensor(
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        **kwargs,
    ).reshape(1, 1, 1, 3, 3)

    # J2*
    kernels["2*xy"] = tensor(
        [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
        **kwargs,
    ).reshape(1, 1, 3, 3, 1)
    kernels["2*xz"] = tensor(
        [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
        **kwargs,
    ).reshape(1, 1, 3, 1, 3)
    kernels["2*yz"] = tensor(
        [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
        **kwargs,
    ).reshape(1, 1, 1, 3, 3)

    # combine kernels with the same sizes
    weights = {
        "x": cat([kernels[key] for key in ["D0x", "D+x", "D-x"]] * 3, dim=0),
        "y": cat([kernels[key] for key in ["D0y", "D+y", "D-y"]] * 3, dim=0),
        "z": cat([kernels[key] for key in ["D0z", "D+z", "D-z"]] * 3, dim=0),
        "*xy": cat([kernels[key] for key in ["1*xy", "2*xy"]] * 3, dim=0),
        "*xz": cat([kernels[key] for key in ["1*xz", "2*xz"]] * 3, dim=0),
        "*yz": cat([kernels[key] for key in ["1*yz", "2*yz"]] * 3, dim=0),
    }

    partials = {
        "x": conv3d(trans, weights["x"], groups=3)[:, :, :, 1:-1, 1:-1],
        "y": conv3d(trans, weights["y"], groups=3)[:, :, 1:-1, :, 1:-1],
        "z": conv3d(trans, weights["z"], groups=3)[:, :, 1:-1, 1:-1, :],
        "*xy": conv3d(trans, weights["*xy"], groups=3)[:, :, :, :, 1:-1],
        "*xz": conv3d(trans, weights["*xz"], groups=3)[:, :, :, 1:-1, :],
        "*yz": conv3d(trans, weights["*yz"], groups=3)[:, :, 1:-1, :, :],
    }

    jacobians = {
        "000": stack(
            (
                partials["x"][:, ::3, ...],
                partials["y"][:, ::3, ...],
                partials["z"][:, ::3, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "+++": stack(
            (
                partials["x"][:, 1::3, ...],
                partials["y"][:, 1::3, ...],
                partials["z"][:, 1::3, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "++-": stack(
            (
                partials["x"][:, 1::3, ...],
                partials["y"][:, 1::3, ...],
                partials["z"][:, 2::3, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "+-+": stack(
            (
                partials["x"][:, 1::3, ...],
                partials["y"][:, 2::3, ...],
                partials["z"][:, 1::3, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "+--": stack(
            (
                partials["x"][:, 1::3, ...],
                partials["y"][:, 2::3, ...],
                partials["z"][:, 2::3, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "-++": stack(
            (
                partials["x"][:, 2::3, ...],
                partials["y"][:, 1::3, ...],
                partials["z"][:, 1::3, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "-+-": stack(
            (
                partials["x"][:, 2::3, ...],
                partials["y"][:, 1::3, ...],
                partials["z"][:, 2::3, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "--+": stack(
            (
                partials["x"][:, 2::3, ...],
                partials["y"][:, 2::3, ...],
                partials["z"][:, 1::3, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "---": stack(
            (
                partials["x"][:, 2::3, ...],
                partials["y"][:, 2::3, ...],
                partials["z"][:, 2::3, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "j1*": stack(
            (
                partials["*xy"][:, 0::2, ...],
                partials["*xz"][:, 0::2, ...],
                partials["*yz"][:, 0::2, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
        "j2*": stack(
            (
                partials["*xy"][:, 1::2, ...],
                partials["*yz"][:, 1::2, ...],
                partials["*xz"][:, 1::2, ...],
            ),
            dim=-1,
        ).permute(0, 2, 3, 4, 1, 5),
    }

    jacdets = {key: det(value) for key, value in jacobians.items()}

    return jacdets


def calculate_non_diffeomorphic_volume(
    jacobian_determinants: Mapping[str, Tensor], mask: Tensor | None = None, threshold=0.0
) -> Tensor:
    """Calculates the non-diffeomorphic volume using the given jacobian determinants

    Modified from https://github.com/yihao6/digital_diffeomorphism

    Liu, Yihao, et al. "On finite difference jacobian computation in deformable image registration."
    International Journal of Computer Vision (2024): 1-11.
    """

    return calculate_non_diffeomorphic_volume_map(
        jacobian_determinants=jacobian_determinants, mask=mask, threshold=threshold
    ).sum()


def calculate_non_diffeomorphic_volume_map(
    jacobian_determinants: Mapping[str, Tensor], mask: Tensor | None = None, threshold=0.0
) -> Tensor:
    """Calculates the non-diffeomorphic volume map using the given jacobian determinants

    Modified from https://github.com/yihao6/digital_diffeomorphism

    Liu, Yihao, et al. "On finite difference jacobian computation in deformable image registration."
    International Journal of Computer Vision (2024): 1-11.
    """
    if mask is not None:
        mask = mask[:, 1:-1, 1:-1, 1:-1]
    non_diff_volume_map: Tensor | None = None
    for diff_direction in ["+++", "++-", "+-+", "+--", "j1*", "j2*", "-++", "-+-", "--+", "---"]:
        volume_map = -0.5 * jacobian_determinants[diff_direction].clamp(min=None, max=threshold) / 6
        if mask is not None:
            volume_map = volume_map * mask
        non_diff_volume_map = optional_add(
            non_diff_volume_map,
            volume_map,
        )

    assert non_diff_volume_map is not None

    return non_diff_volume_map


def calculate_squared_non_diffeomorphic_volume(
    jacobian_determinants: Mapping[str, Tensor], mask: Tensor | None = None, threshold=0.0
) -> Tensor:
    """Calculates the non-diffeomorphic volume map using the given jacobian determinants

    Modified from https://github.com/yihao6/digital_diffeomorphism

    Liu, Yihao, et al. "On finite difference jacobian computation in deformable image registration."
    International Journal of Computer Vision (2024): 1-11.
    """
    if mask is not None:
        mask = mask[:, 1:-1, 1:-1, 1:-1]
    squared_volume: Tensor | None = None
    for diff_direction in ["+++", "++-", "+-+", "+--", "j1*", "j2*", "-++", "-+-", "--+", "---"]:
        volume_map = -0.5 * jacobian_determinants[diff_direction].clamp(min=None, max=threshold) / 6
        if mask is not None:
            volume_map = volume_map * mask
        squared_volume = optional_add(
            squared_volume,
            volume_map.square().sum(),
        )

    assert squared_volume is not None

    return squared_volume
