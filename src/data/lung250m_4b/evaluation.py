"""Lung250M-4B dataset evaluation"""

from logging import getLogger
from typing import Any, Mapping, Sequence

from composable_mapping import (
    CoordinateSystem,
    DataFormat,
    LinearInterpolator,
    mappable,
    samplable_volume,
)
from torch import Tensor, tensor

from algorithm.cubic_spline_upsampling import CubicSplineUpsampling
from data.base import BaseVolumetricRegistrationEvaluator

logger = getLogger(__name__)


class Lung250M4BEvaluator(BaseVolumetricRegistrationEvaluator):
    """Lung250M-4B evaluator"""

    def __init__(
        self,
        source_landmarks: Tensor,
        target_landmarks: Tensor,
        metrics_to_compute: Sequence[str],
        n_jacobian_samples: int | None = None,
        jacobian_sampling_seed: int | None = None,
        evaluation_prefix: str = "",
        upsampling_factor: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            metrics_to_compute=metrics_to_compute,
            n_jacobian_samples=n_jacobian_samples,
            jacobian_sampling_seed=jacobian_sampling_seed,
            evaluation_prefix=evaluation_prefix,
        )
        self._source_landmarks = source_landmarks
        self._target_landmarks = target_landmarks
        self._upsampling_factor = upsampling_factor

    def _upsample_displacement_fields(
        self, inference_outputs: Mapping[str, Any]
    ) -> Mapping[str, Sequence[Tensor | None]]:
        upsampled_ddfs: dict[str, Sequence[Tensor | None]] = {}
        for ddf_key in ["forward_displacement_field", "inverse_displacement_field"]:
            updated_ddfs: list[Tensor | None] = []
            for ddf in inference_outputs[ddf_key]:
                if ddf is None:
                    updated_ddfs.append(None)
                    continue
                if self._upsampling_factor is None:
                    upsampling_factor = (1,) * (ddf.ndim - 1)
                else:
                    upsampling_factor = self._upsampling_factor
                scaling_factor = tensor(upsampling_factor, device=ddf.device, dtype=ddf.dtype)[
                    (...,) + (None,) * (ddf.ndim - 1)
                ]
                upsampler = CubicSplineUpsampling(
                    upsampling_factor=upsampling_factor, dtype=ddf.dtype
                )
                upsampler.to(ddf.device)
                upsampled_ddf = (
                    upsampler(ddf[None], apply_prefiltering=True, prefilter_inplace=True)[0]
                    * scaling_factor
                )
                updated_ddfs.append(upsampled_ddf)
            upsampled_ddfs[ddf_key] = updated_ddfs
        return upsampled_ddfs

    def __call__(self, inference_outputs: Mapping[str, Any]) -> Mapping[str, float]:
        upsampled_ddfs = self._upsample_displacement_fields(inference_outputs)
        metrics: dict[str, int | float] = {}
        for index, forward_displacement_field in enumerate(
            upsampled_ddfs["forward_displacement_field"]
        ):
            if forward_displacement_field is not None:
                metrics.update(
                    self._compute_landmark_metrics(
                        displacement_field=forward_displacement_field,
                        source_landmarks=self._source_landmarks,
                        target_landmarks=self._target_landmarks,
                        prefix=f"{self._evaluation_prefix}input_order_{index}_forward_",
                    )
                )
        for index, inverse_displacement_field in enumerate(
            upsampled_ddfs["inverse_displacement_field"]
        ):
            if inverse_displacement_field is not None:
                metrics.update(
                    self._compute_landmark_metrics(
                        displacement_field=inverse_displacement_field,
                        source_landmarks=self._target_landmarks,
                        target_landmarks=self._source_landmarks,
                        prefix=f"{self._evaluation_prefix}input_order_{index}_inverse_",
                    )
                )
        return dict(super().__call__(inference_outputs)) | metrics

    def _compute_landmark_metrics(
        self,
        displacement_field: Tensor,
        source_landmarks: Tensor,
        target_landmarks: Tensor,
        prefix: str,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if "landmark" not in self._metrics_to_compute:
            return {}
        source_landmarks = source_landmarks.to(displacement_field.device)
        target_landmarks = target_landmarks.to(displacement_field.device)
        shape_tensor = tensor(displacement_field.shape[1:], device=displacement_field.device)[
            :, None
        ]
        landmark_source_fov_mask = (source_landmarks >= 0).all(dim=0) & (
            source_landmarks <= shape_tensor - 1
        ).all(dim=0)
        landmark_target_fov_mask = (target_landmarks >= 0).all(dim=0) & (
            target_landmarks <= shape_tensor - 1
        ).all(dim=0)
        combined_landmark_fov_mask = landmark_source_fov_mask & landmark_target_fov_mask
        n_landmarks_outside_fov = (~combined_landmark_fov_mask).sum().item()
        if n_landmarks_outside_fov > 0:
            logger.warning(
                "Total of %d landmark(s) are outside the field of view "
                "and will be ignored for evaluation.",
                n_landmarks_outside_fov,
            )
        source_landmarks = source_landmarks[:, combined_landmark_fov_mask]
        target_landmarks = target_landmarks[:, combined_landmark_fov_mask]
        displacement_field = displacement_field[None]
        coordinate_system = CoordinateSystem.voxel(
            displacement_field.shape[2:],
            voxel_size=(1.0, 1.0, 1.0),
            dtype=displacement_field.dtype,
            device=displacement_field.device,
        )
        displacement_field_mapping = samplable_volume(
            displacement_field,
            coordinate_system=coordinate_system,
            data_format=DataFormat.voxel_displacements(),
            sampler=LinearInterpolator(mask_extrapolated_regions_for_empty_volume_mask=False),
        )
        transformed_target_landmarks = displacement_field_mapping(
            mappable(target_landmarks[None])
        ).generate_values()[0]
        distances = (transformed_target_landmarks - source_landmarks).norm(p=2, dim=0)
        metrics[f"{prefix}landmark_distance"] = distances.mean().item()
        metrics[f"{prefix}original_landmark_distance"] = (
            (target_landmarks - source_landmarks).norm(p=2, dim=0).mean().item()
        )
        return metrics
