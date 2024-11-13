"""Script for measuring memory use with respect to standard SVF framework"""

from argparse import ArgumentParser

from composable_mapping import (
    CoordinateSystem,
    DataFormat,
    LinearInterpolator,
    samplable_volume,
)
from deformation_inversion_layer import fixed_point_invert_deformation
from deformation_inversion_layer.fixed_point_invert_deformation import (
    DeformationInversionArguments,
)
from deformation_inversion_layer.fixed_point_iteration import (
    AndersonSolver,
    AndersonSolverArguments,
    MaxElementWiseAbsStopCriterion,
    RelativeL2ErrorStopCriterion,
)
from torch import Generator, Tensor, arange, device, meshgrid, rand, stack
from torch.cuda import (
    Event,
    current_stream,
    max_memory_allocated,
    memory_allocated,
    reset_peak_memory_stats,
    synchronize,
)
from torch.nn import Module
from torch.nn.functional import grid_sample

DEVICE = device("cuda:0")
SHAPE = (144, 192, 160)


class SpatialTransformer(Module):
    """3D Spatial Transformer"""

    def __init__(self, shape, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.register_buffer(
            "grid",
            stack(meshgrid([arange(0, dim_size) for dim_size in shape], indexing="ij")).unsqueeze(
                0
            ),
            persistent=False,
        )

    def forward(self, src, flow):
        """Perform interpolation"""
        new_locs = self.grid + flow
        new_locs = new_locs.permute(0, 2, 3, 4, 1).flip(-1)
        return grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class SVFIntegration(Module):
    """Integrates a vector field via scaling and squaring."""

    def __init__(self, shape, n_steps):
        super().__init__()
        self.n_steps = n_steps
        self.transformer = SpatialTransformer(shape)

    def forward(self, svf):
        """Perform SVF integration"""
        scale = 1.0 / (2**self.n_steps)
        svf = svf * scale
        for _ in range(self.n_steps):
            svf = svf + self.transformer(svf, svf)
        return svf


def _generate_volume():
    generator = Generator().manual_seed(123456789)
    return rand((1, 3) + SHAPE, generator=generator).to(DEVICE)


def _measure_svf_memory_single():
    svf = _generate_volume().requires_grad_(True)
    svf_integration = SVFIntegration(shape=SHAPE, n_steps=7)
    svf_integration.to(DEVICE)
    memory_before = memory_allocated(DEVICE)
    integrated = svf_integration(svf)
    memory_after = memory_allocated(DEVICE)
    integrated.mean().backward()
    peak_memory_usage = max_memory_allocated(DEVICE)
    reset_peak_memory_stats(DEVICE)
    print("SVF:")
    print(f"Memory required for backpropagation: {(memory_after - memory_before) / 2**30}")
    print(f"Peak memory usage: {peak_memory_usage / 2**30}")


def _measure_svf_memory_full():
    forward_start_event = Event(enable_timing=True)
    forward_end_event = Event(enable_timing=True)
    backward_end_event = Event(enable_timing=True)
    cuda_stream = current_stream(device=DEVICE)
    svf_1 = _generate_volume().requires_grad_(True)
    svf_2 = _generate_volume().requires_grad_(True)
    svf_integration = SVFIntegration(shape=SHAPE, n_steps=7)
    svf_integration.to(DEVICE)
    memory_before = memory_allocated(DEVICE)
    forward_start_event.record(cuda_stream)
    integrated = svf_integration(svf_1 - svf_2)
    forward_end_event.record(cuda_stream)
    memory_after = memory_allocated(DEVICE)
    integrated.mean().backward()
    backward_end_event.record(cuda_stream)
    peak_memory_usage = max_memory_allocated(DEVICE)
    reset_peak_memory_stats(DEVICE)
    print("SVF:")
    print(f"Memory required for backpropagation: {(memory_after - memory_before) / 2**30}")
    print(f"Peak memory usage: {peak_memory_usage / 2**30}")
    synchronize(DEVICE)
    print(f"Forward pass time elapsed: {forward_start_event.elapsed_time(forward_end_event)}")
    print(f"Full time elapsed: {forward_start_event.elapsed_time(backward_end_event)}")


def _fixed_point_composition(displacement_field_1: Tensor, displacement_field_2: Tensor) -> Tensor:
    interpolator = LinearInterpolator()
    inverse_displacement_field = fixed_point_invert_deformation(
        displacement_field=displacement_field_2,
        arguments=DeformationInversionArguments(
            interpolator=interpolator.sample_values,
            forward_solver=AndersonSolver(
                stop_criterion=MaxElementWiseAbsStopCriterion(
                    min_iterations=2, max_iterations=50, threshold=1e-2
                ),
                arguments=AndersonSolverArguments(memory_length=4),
            ),
            backward_solver=AndersonSolver(
                stop_criterion=RelativeL2ErrorStopCriterion(
                    min_iterations=2, max_iterations=50, threshold=1e-2
                ),
                arguments=AndersonSolverArguments(memory_length=4),
            ),
        ),
    )
    coordinate_system = CoordinateSystem.voxel(
        SHAPE, dtype=displacement_field_1.dtype, device=displacement_field_1.device
    )
    mapping = samplable_volume(
        displacement_field_1,
        coordinate_system=coordinate_system,
        data_format=DataFormat.voxel_displacements(),
        sampler=LinearInterpolator(mask_extrapolated_regions_for_empty_volume_mask=False),
    )
    inverse_mapping = samplable_volume(
        inverse_displacement_field,
        coordinate_system=coordinate_system,
        data_format=DataFormat.voxel_displacements(),
        sampler=LinearInterpolator(mask_extrapolated_regions_for_empty_volume_mask=False),
    )
    return (mapping @ inverse_mapping).sample().generate_values()


def _measure_fixed_point_memory_single():
    displacement_field = _generate_volume().requires_grad_(True)
    memory_before = memory_allocated(DEVICE)
    inverse_displacement_field = fixed_point_invert_deformation(
        displacement_field=displacement_field,
        arguments=DeformationInversionArguments(
            interpolator=LinearInterpolator(),
            forward_solver=AndersonSolver(
                stop_criterion=MaxElementWiseAbsStopCriterion(
                    min_iterations=2, max_iterations=50, threshold=1e-2
                ),
                arguments=AndersonSolverArguments(memory_length=4),
            ),
            backward_solver=AndersonSolver(
                stop_criterion=RelativeL2ErrorStopCriterion(
                    min_iterations=2, max_iterations=50, threshold=1e-2
                ),
                arguments=AndersonSolverArguments(memory_length=4),
            ),
        ),
    )
    memory_after = memory_allocated(DEVICE)
    inverse_displacement_field.mean().backward()
    peak_memory_usage = max_memory_allocated(DEVICE)
    reset_peak_memory_stats(DEVICE)
    print("Fixed point inversion:")
    print(f"Memory required for backpropagation: {(memory_after - memory_before) / 2**30}")
    print(f"Peak memory usage: {peak_memory_usage / 2**30}")


def _measure_fixed_point_memory_full():
    forward_start_event = Event(enable_timing=True)
    forward_end_event = Event(enable_timing=True)
    backward_end_event = Event(enable_timing=True)
    cuda_stream = current_stream(device=DEVICE)
    displacement_field_1 = _generate_volume().requires_grad_(True) / 10
    displacement_field_2 = _generate_volume().requires_grad_(True) / 10
    memory_before = memory_allocated(DEVICE)
    forward_start_event.record(cuda_stream)
    composition_displacement_field = _fixed_point_composition(
        displacement_field_1=displacement_field_1, displacement_field_2=displacement_field_2
    )
    forward_end_event.record(cuda_stream)
    memory_after = memory_allocated(DEVICE)
    composition_displacement_field.mean().backward()
    backward_end_event.record(cuda_stream)
    peak_memory_usage = max_memory_allocated(DEVICE)
    reset_peak_memory_stats(DEVICE)
    print("Fixed point inversion:")
    print(f"Memory required for backpropagation: {(memory_after - memory_before) / 2**30}")
    print(f"Peak memory usage: {peak_memory_usage / 2**30}")
    synchronize(DEVICE)
    print(f"Forward pass time elapsed: {forward_start_event.elapsed_time(forward_end_event)}")
    print(f"Full time elapsed: {forward_start_event.elapsed_time(backward_end_event)}")


def _main() -> None:
    """Parse arguments for training and train the model"""
    parser = ArgumentParser()
    parser.add_argument("--fixed-point-full", action="store_true")
    parser.add_argument("--svf-full", action="store_true")
    parser.add_argument("--fixed-point-single", action="store_true")
    parser.add_argument("--svf-single", action="store_true")
    parser.add_argument("--iterations", type=int, required=False, default=1)
    args = parser.parse_args()
    for iteration in range(args.iterations):
        print(f"Starting iteration {iteration + 1}")
        if args.fixed_point_full:
            _measure_fixed_point_memory_full()
        if args.svf_full:
            _measure_svf_memory_full()
        if args.fixed_point_single:
            _measure_fixed_point_memory_single()
        if args.svf_single:
            _measure_svf_memory_single()
        print("-----------------------")


if __name__ == "__main__":
    _main()
