"""Algorithms for computing Lipschitz-coefficient of deformations"""

from torch import Tensor, empty
from torch.linalg import svdvals

from algorithm.interface import IFixedPointSolver

from .finite_difference import estimate_spatial_jacobian_matrices
from .spectral_norm import calculate_spectral_norm_with_power_iteration


def calculate_local_lipschitz_constants(
    deformation: Tensor,
    method: str = "power_iteration",
    fixed_point_solver: IFixedPointSolver | None = None,
    initial_left_singular_vector: Tensor | None = None,
    initial_right_singular_vector: Tensor | None = None,
) -> Tensor:
    """Calculate local squared Lipschitz coefficients

    Assumes linear interpolation between grid values and voxel coordinates.
    Lipschitz constant is returned for each grid region.

    Args:
        deformation: Displacement field Tensor with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        method: Method for computing the spectral norm, either "power_iteration", "svd",
            or "fro". "fro" only gives an upper bound on spectral norm but is
            computationally very effecient.
        fixed_point_solver: Solver for solving power iteration, has no effect when
            method is "svd" or "fro".
        initial_left_singular_vector: Initial left singular vector for power
            iteration, has no effect when method is "svd" or "fro"..
        initial_right_singular_vector: Initial right singular vector for power
            iteration, has no effect when method is "svd" or "fro"..
    Returns:
        Tensor with shape (batch_size, n_channels, dim_1 - 1, ..., dim_{n_dims} - 1)
    """
    batch_size = deformation.size(0)
    n_dims = deformation.size(1)
    jacobians = empty(
        (batch_size, n_dims, n_dims, 2**n_dims)
        + tuple(dim_size - 1 for dim_size in deformation.shape[2:]),
        dtype=deformation.dtype,
        device=deformation.device,
    )
    estimate_spatial_jacobian_matrices(
        volume=deformation,
        other_dims="crop_both",
        out=jacobians,
    )
    if method == "power_iteration":
        if fixed_point_solver is None:
            raise ValueError(f'Fixed point solver needs to be specified when method == "{method}"')
        spectral_norms = (
            calculate_spectral_norm_with_power_iteration(
                jacobians,
                solver=fixed_point_solver,
                initial_left_singular_vector=initial_left_singular_vector,
                initial_right_singular_vector=initial_right_singular_vector,
            )
            .max(dim=1)
            .values
        )
    elif method == "svd":
        spectral_norms = (
            svdvals(jacobians.movedim((1, 2), (-2, -1)))[..., 0].max(dim=1).values
        )
    elif method == "fro":
        spectral_norms = jacobians.square().sum(dim=(1, 2)).max(dim=1).values.sqrt()
    else:
        raise ValueError(f'Unknown method: "{method}"')
    return spectral_norms
