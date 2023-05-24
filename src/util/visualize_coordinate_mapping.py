"""Deformation related utility functions"""

from typing import Any, Iterable, Mapping, Optional

from matplotlib.pyplot import axis, plot  # type: ignore
from torch import device as torch_device
from torch import dtype as torch_dtype

from algorithm.composable_mapping.interface import IComposableMapping, IMaskedTensor


def visualize_coordinate_mapping_2d(
    mapping: IComposableMapping,
    grid: IMaskedTensor,
    device: Optional[torch_device] = None,
    dtype: Optional[torch_dtype] = None,
    batch_index: int = 0,
    emphasize_every_nth_line: None | tuple[int, int] = None,
    visualize_mask: bool = False,
    inside_mask_color: Any = "gray",
    outside_mask_color: Any = "red",
    **kwargs
) -> None:
    """Visualizes a mapping as deformed grid

    Args:
        mapping: Coordinate mapping to visualize
        grid: Visualize at the grid locations
        device: Device to use for visualization
        dtype: DType to use for visualization
        batch_index: Batch index to visualize
        emphasize_every_nth_line: If given, emphasize every
            emphasize_every_nth_line[0]:th line starting from line number
            emphasize_every_nth_line[1]. Does not work with mask visualization.
        visualize_mask: Visualize mask
        inside_mask_color: Color inside mask
        outside_mask_color: Color outside mask
    """
    transformed_grid = mapping(grid)
    values = (
        transformed_grid.generate_values(device=device, dtype=dtype).detach().cpu()[batch_index]
    )
    dims = [dim for dim, dim_size in enumerate(values.shape[1:]) if dim_size > 1]
    if len(dims) != 2:
        raise ValueError("Exactly two dimensions should have size different to 1.")
    values = values.squeeze()[dims]

    def _get_kwargs(
        indices: Iterable[int],
        inside_mask: bool = True,
    ) -> Mapping[str, Any]:
        if inside_mask:
            default_kwargs = {"color": inside_mask_color} | kwargs
        else:
            default_kwargs = {"color": outside_mask_color} | kwargs
        if emphasize_every_nth_line is None:
            return default_kwargs
        if any(
            (index + emphasize_every_nth_line[1]) % emphasize_every_nth_line[0] == 0
            for index in indices
        ):
            return {"alpha": 0.6, "linewidth": 2.0} | default_kwargs
        return {"alpha": 0.2, "linewidth": 1.0} | default_kwargs

    axis("equal")
    if not visualize_mask:
        for row_index in range(values.size(1)):
            plot(values[1, row_index, :], values[0, row_index, :], **_get_kwargs([row_index]))
        for col_index in range(values.size(2)):
            plot(values[1, :, col_index], values[0, :, col_index], **_get_kwargs([col_index]))
    else:
        mask = transformed_grid.generate_mask(device).detach().cpu()[batch_index]
        for row_index in range(values.size(1)):
            is_inside_mask = bool(mask[0, row_index, 0] == 1.0)
            previous_change_index = 0
            for col_index in range(values.size(2) - 1):
                is_next_inside_mask = bool(mask[0, row_index, col_index + 1] == 1.0)
                if is_next_inside_mask != is_inside_mask:
                    if col_index + 1 - previous_change_index > 1:
                        plot(
                            values[1, row_index, previous_change_index : col_index + 1],
                            values[0, row_index, previous_change_index : col_index + 1],
                            **_get_kwargs([row_index, col_index], inside_mask=is_inside_mask),
                        )
                    plot(
                        values[1, row_index, col_index : col_index + 2],
                        values[0, row_index, col_index : col_index + 2],
                        **_get_kwargs([row_index, col_index], inside_mask=False),
                    )
                    previous_change_index = col_index + 1
                    is_inside_mask = is_next_inside_mask
            if values.size(2) - previous_change_index > 1:
                plot(
                    values[1, row_index, previous_change_index:],
                    values[0, row_index, previous_change_index:],
                    **_get_kwargs([row_index, col_index], inside_mask=is_inside_mask),
                )
        for col_index in range(values.size(2)):
            is_inside_mask = bool(mask[0, 0, col_index] == 1.0)
            previous_change_index = 0
            for row_index in range(values.size(1) - 1):
                is_next_inside_mask = bool(mask[0, row_index + 1, col_index] == 1.0)
                if is_next_inside_mask != is_inside_mask:
                    if row_index + 1 - previous_change_index > 1:
                        plot(
                            values[1, previous_change_index : row_index + 1, col_index],
                            values[0, previous_change_index : row_index + 1, col_index],
                            **_get_kwargs([row_index, col_index], inside_mask=is_inside_mask),
                        )
                    plot(
                        values[1, row_index : row_index + 2, col_index],
                        values[0, row_index : row_index + 2, col_index],
                        **_get_kwargs([row_index, col_index], inside_mask=False),
                    )
                    previous_change_index = row_index + 1
                    is_inside_mask = is_next_inside_mask
            if values.size(1) - previous_change_index > 1:
                plot(
                    values[1, previous_change_index:, col_index],
                    values[0, previous_change_index:, col_index],
                    **_get_kwargs([row_index, col_index], inside_mask=is_inside_mask),
                )
