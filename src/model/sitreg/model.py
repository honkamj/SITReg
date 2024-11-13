"""Implementation of SITReg model"""

from abc import abstractmethod
from itertools import count
from logging import getLogger
from typing import NamedTuple, Optional, Sequence, cast

from composable_mapping import (
    CoordinateSystem,
    CubicSplineSampler,
    DataFormat,
    GridComposableMapping,
    Identity,
    LinearInterpolator,
    OriginalFOV,
    Start,
    affine,
    default_sampler,
    samplable_volume,
)
from deformation_inversion_layer.interface import FixedPointSolver
from numpy import prod as np_prod
from torch import Tensor, cat, chunk
from torch import device as torch_device
from torch import float64, long, tanh
from torch.nn import Linear, Module, ModuleList

from algorithm.affine_transformation import (
    AffineTransformationTypeDefinition,
    calculate_n_parameters,
    generate_affine_transformation_matrix,
)
from algorithm.cubic_b_spline_control_point_upper_bound import (
    compute_max_control_point_value,
)
from model.components import ConvBlockNd, ConvNd
from model.interface import IActivationFactory, INormalizerFactory
from model.normalizer import get_normalizer_factory

from .interface import FeatureExtractor

logger = getLogger(__name__)


class MappingPair(NamedTuple):
    """Mapping pair containing both forward and inverse deformation"""

    forward_mapping: GridComposableMapping
    inverse_mapping: GridComposableMapping


class SITReg(Module):
    """SITReg is a deep learning intra-modality image registration arhitecture
    fulfilling strict symmetry properties

    The implementation is dimensionality agnostic but PyTorch linear interpolation
    supports only 2 and 3 dimensional volumes.

    Arguments:
        feature_extractor: Multi-resolution feature extractor
        n_transformation_features_per_resolution: Defines how many features to
            use for extracting transformation in anti-symmetric update for each
            resolution. If None is given for some resolution, no deformation is
            extracted for that.
        n_transformation_convolutions_per_resolution: Defines how many convolutions to
            use for extracting transformation in anti-symmetric update for each
            resolution. If None is given for some resolution, no deformation is
            extracted for that.
        affine_transformation_type: Defines which type of affine transformation
            to predict. If None, no affine transformation is predicted.
        input_voxel_size: Voxel size of the inputs images
        input_shape: Shape of the input images
        transformation_downsampling_factor: Downsampling factor for each
            dimension for the final deformation, e.g., providing [1.0, 1.0, 1.0] means
            no downsampling for three dimensional inputs.
        forward_fixed_point_solver: Defines fixed point solver for the forward pass
            of the deformation inversion layers.
        backward_fixed_point_solver: Defines fixed point solver for the backward pass
            of the deformation inversion layers.
        max_control_point_multiplier: Optimal maximum control point values are
            multiplied with this value to ensure that the individual
            deformations are invertible even after numerical errors. This should
            be some value just below 1, e.g. 0.99.
        activation_factory: Activation function to use.
        normalizer_factory: Normalizer factory to use. If None, no normalization
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        n_transformation_features_per_resolution: Sequence[int],
        n_transformation_convolutions_per_resolution: Sequence[int],
        affine_transformation_type: Optional[AffineTransformationTypeDefinition],
        input_voxel_size: Sequence[float],
        input_shape: Sequence[int],
        transformation_downsampling_factor: Sequence[float],
        forward_fixed_point_solver: FixedPointSolver,
        backward_fixed_point_solver: FixedPointSolver,
        max_control_point_multiplier: float,
        activation_factory: IActivationFactory,
        normalizer_factory: INormalizerFactory | None,
    ) -> None:
        super().__init__()
        normalizer_factory = get_normalizer_factory(normalizer_factory)
        feature_shapes = feature_extractor.get_shapes()
        n_blocks = len(feature_shapes)
        if n_blocks != len(n_transformation_features_per_resolution) or n_blocks != len(
            n_transformation_convolutions_per_resolution
        ):
            raise ValueError(
                "Lists specifying blocks have to have the same length as the number of "
                "feature levels produced by the feature extractor"
            )
        predict_dense_for_stages = [
            n_transformation_convolutions is not None
            for n_transformation_convolutions in n_transformation_convolutions_per_resolution
        ]
        for (
            n_transformation_features,
            predict_dense_for_stage,
        ) in zip(
            n_transformation_features_per_resolution,
            predict_dense_for_stages,
        ):
            if (predict_dense_for_stage and n_transformation_features is None) or (
                not predict_dense_for_stage and n_transformation_features is not None
            ):
                raise ValueError(
                    "Either both, number of tranformation convolutions and number of "
                    "transformation features must be provided for a resolution, or neither of them."
                )
        n_dims = len(input_voxel_size)
        self._feature_coordinate_systems = ModuleList(
            [
                CoordinateSystem.centered_normalized(
                    spatial_shape=input_shape,
                    voxel_size=input_voxel_size,
                ).reformat(
                    downsampling_factor=downsampling_factor,
                    spatial_shape=feature_shape[1:],
                    reference=Start(),
                )
                for (predict_deformation, feature_shape, downsampling_factor) in zip(
                    predict_dense_for_stages,
                    feature_shapes,
                    feature_extractor.get_downsampling_factors(),
                )
                if predict_deformation
            ]
        )
        self._transformation_coordinate_system = CoordinateSystem.centered_normalized(
            spatial_shape=input_shape,
            voxel_size=input_voxel_size,
        ).reformat(
            downsampling_factor=transformation_downsampling_factor,
            spatial_shape=OriginalFOV(fitting_method="ceil"),
        )
        self._image_coordinate_system = CoordinateSystem.centered_normalized(
            spatial_shape=input_shape,
            voxel_size=input_voxel_size,
        )
        counter = count()
        self._dense_prediction_indices = [
            next(counter) if predict_deformation else None
            for predict_deformation in predict_dense_for_stages
        ]
        # variable name for backward compatibility
        self._not_none_dense_extration_networks = ModuleList(
            [
                _DenseExtractionNetwork(
                    n_input_features=feature_shape[0],
                    n_features=n_transformation_features,
                    n_convolutions=n_transformation_convolutions,
                    feature_coordinate_system=cast(
                        CoordinateSystem,
                        self._feature_coordinate_systems[dense_prediction_index],
                    ),
                    transformation_coordinate_system=self._transformation_coordinate_system,
                    forward_fixed_point_solver=forward_fixed_point_solver,
                    backward_fixed_point_solver=backward_fixed_point_solver,
                    max_control_point_multiplier=max_control_point_multiplier,
                    activation_factory=activation_factory,
                    normalizer_factory=normalizer_factory,
                )
                for (
                    dense_prediction_index,
                    feature_shape,
                    n_transformation_features,
                    n_transformation_convolutions,
                ) in zip(
                    self._dense_prediction_indices,
                    feature_shapes,
                    n_transformation_features_per_resolution,
                    n_transformation_convolutions_per_resolution,
                )
                if dense_prediction_index is not None
            ]
        )
        self._affine_extraction_network = (
            _AffineExtractionNetwork(
                n_dims=n_dims,
                volume_shape=feature_shapes[-1][1:],
                n_features=feature_shapes[-1][0],
                activation_factory=activation_factory,
                transformation_coordinate_system=self._transformation_coordinate_system,
                affine_transformation_type=affine_transformation_type,
            )
            if affine_transformation_type is not None
            else None
        )
        self._predict_dense_for_stages = predict_dense_for_stages
        self._feature_extractor = feature_extractor

    @property
    def image_coordinate_system(self) -> CoordinateSystem:
        """Coordinate system used by the network for the inputs."""
        return self._image_coordinate_system

    @property
    def transformation_coordinate_system(self) -> CoordinateSystem:
        """Coordinate system used by the predicted transformations."""
        return self._transformation_coordinate_system

    def _extract_affine(
        self,
        batch_combined_features: Tensor,
    ) -> tuple[GridComposableMapping, GridComposableMapping]:
        if self._affine_extraction_network is None:
            return (
                Identity().assign_coordinates(self._transformation_coordinate_system),
                Identity().assign_coordinates(self._transformation_coordinate_system),
            )
        logger.debug("Starting affine transformation extraction")
        features_1, features_2 = chunk(batch_combined_features, 2)
        return self._affine_extraction_network(features_1, features_2)

    def _extract_dense(
        self,
        batch_combined_features: Tensor,
        coordinate_system: CoordinateSystem,
        mapping_builder: "_MappingBuilder",
        dense_extraction_network: Module,
    ) -> tuple[GridComposableMapping, GridComposableMapping]:
        features_1, features_2 = chunk(batch_combined_features, 2)
        transformed_features_1 = (
            (
                samplable_volume(
                    data=features_1,
                    coordinate_system=coordinate_system,
                )
                @ mapping_builder.left_forward()
            )
            .sample()
            .generate_values()
        )
        transformed_features_2 = (
            (
                samplable_volume(
                    data=features_2,
                    coordinate_system=coordinate_system,
                )
                @ mapping_builder.right_forward()
            )
            .sample()
            .generate_values()
        )
        return dense_extraction_network(
            transformed_features_1,
            transformed_features_2,
        )

    def _to_level_index(self, low_first_level_index: int) -> int:
        """Obtain high resolution first level index

        Args:
            low_first_level_index: Low resolution first index, -1 refers to affine level
        """
        n_dense_levels = sum(self._predict_dense_for_stages)
        return n_dense_levels - 1 - low_first_level_index

    def forward(
        self,
        image_1: Tensor,
        image_2: Tensor,
        mappings_for_levels: Sequence[tuple[int, bool]] = ((0, True),),
        resample_when_composing: bool = True,
    ) -> list[MappingPair]:
        """Generate deformations

        Args:
            image_1: Image registered to image_2, Tensor with shape
                (batch_size, n_channels, dim_1, ..., dim_{n_dims})
            image_2: Image registered to image_1, Tensor with shape
                (batch_size, n_channels, dim_1, ..., dim_{n_dims})
            deformations_for_levels: Defines for which resolution levels the
                deformations are returned. The first element of each tuple is
                the index of the level and the second element indicates whether
                to include affine transformation in the deformation. Indexing
                starts from the highest resolution. Default value is [(0, True)]
                which corresponds to returning the full deformation at the
                highest resolution.

        Returns:
            List of MappingPairs in the order given by the input argument
            "mappings_for_levels"
        """
        with default_sampler(
            LinearInterpolator(mask_extrapolated_regions_for_empty_volume_mask=False)
        ):
            mappings_for_levels_set = set(mappings_for_levels)
            batch_combined_features_list: list[Tensor] = self._feature_extractor(
                (image_1, image_2),
            )
            predict_affine = self._affine_extraction_network is not None
            forward_affine, inverse_affine = self._extract_affine(
                batch_combined_features_list[-1],
            )
            mapping_builder = _MappingBuilder(
                forward_affine=forward_affine,
                inverse_affine=inverse_affine,
                resample_when_composing=resample_when_composing,
            )
            output_mappings: dict[tuple[int, bool], MappingPair] = {}
            affine_level_index = self._to_level_index(-1)
            for include_affine in (False, True):
                if (affine_level_index, include_affine) in mappings_for_levels_set:
                    output_mappings[(affine_level_index, include_affine)] = (
                        mapping_builder.as_mapping_pair(include_affine)
                    )
                    if not predict_affine:
                        logger.warning(
                            "Deformation requested after affine transformation "
                            "but no affine transformation is predicted."
                        )
                    if not include_affine:
                        logger.warning(
                            "Deformation before any dense deformations requested "
                            "but no affine transformation is predicted."
                        )
            for low_first_level_index, (
                batch_combined_features,
                dense_prediction_index,
            ) in enumerate(
                zip(
                    reversed(batch_combined_features_list),
                    reversed(self._dense_prediction_indices),
                )
            ):
                if dense_prediction_index is not None:
                    if predict_affine and all(
                        dim_size == 1 for dim_size in batch_combined_features.shape[2:]
                    ):
                        logger.warning(
                            "Predicting dense deformation together with affine transformation "
                            "from features with only one dimension is not advisable."
                        )
                    logger.debug(
                        "Starting deformation extraction from features with shape %s",
                        tuple(batch_combined_features.shape),
                    )
                    forward_dense, inverse_dense = self._extract_dense(
                        batch_combined_features=batch_combined_features,
                        coordinate_system=cast(
                            CoordinateSystem,
                            self._feature_coordinate_systems[dense_prediction_index],
                        ),
                        mapping_builder=mapping_builder,
                        dense_extraction_network=self._not_none_dense_extration_networks[
                            dense_prediction_index
                        ],
                    )
                    mapping_builder.update(
                        forward_dense=forward_dense,
                        inverse_dense=inverse_dense,
                    )
                level_index = self._to_level_index(low_first_level_index)
                for include_affine in (False, True):
                    if (level_index, include_affine) in mappings_for_levels_set:
                        output_mappings[(level_index, include_affine)] = (
                            mapping_builder.as_mapping_pair(include_affine)
                        )
                        if dense_prediction_index is None:
                            logger.warning(
                                "Intermediate mapping requested for level %d "
                                "but no deformation is predicted on that level.",
                                level_index,
                            )
                del batch_combined_features_list[-1]
            return [
                output_mappings[mapping_index] for mapping_index in mappings_for_levels
            ]


class _BaseTransformationExtractionNetwork(Module):
    """Base class for generating exactly inverse consistent transformations"""

    @abstractmethod
    def _extract_atomic_transformation(
        self,
        combined_input: Tensor,
    ) -> GridComposableMapping:
        """Extract the smallest unit transformation"""

    @abstractmethod
    def _invert_mapping(
        self, mapping: GridComposableMapping, device: torch_device
    ) -> GridComposableMapping:
        """Invert transformation"""

    def _modify_input(self, input_tensor: Tensor) -> Tensor:
        return input_tensor

    def _extract_atomic_transformations(
        self,
        features_1: Tensor,
        features_2: Tensor,
    ) -> tuple[GridComposableMapping, GridComposableMapping]:
        input_1_modified = self._modify_input(features_1)
        input_2_modified = self._modify_input(features_2)
        forward_combined_input = cat(
            (input_1_modified - input_2_modified, input_1_modified + input_2_modified),
            dim=1,
        )
        reverse_combined_input = cat(
            (input_2_modified - input_1_modified, input_1_modified + input_2_modified),
            dim=1,
        )
        forward_atomic = self._extract_atomic_transformation(
            forward_combined_input,
        )
        reverse_atomic = self._extract_atomic_transformation(
            reverse_combined_input,
        )
        return forward_atomic, reverse_atomic

    def forward(
        self,
        features_1: Tensor,
        features_2: Tensor,
    ) -> tuple[GridComposableMapping, GridComposableMapping]:
        """Generate affine transformation parameters

        Args:
            features_1: Tensor with shape (batch_size, n_features, *volume_shape)
            features_2: Tensor with shape (batch_size, n_features, *volume_shape)

        Returns:
            Forward mapping
            Inverse mapping
            Optional regularization
        """
        forward_atomic, reverse_atomic = self._extract_atomic_transformations(
            features_1=features_1,
            features_2=features_2,
        )
        device = features_1.device
        inverse_forward_atomic = self._invert_mapping(forward_atomic, device=device)
        inverse_reverse_atomic = self._invert_mapping(reverse_atomic, device=device)
        forward_transformation = forward_atomic @ inverse_reverse_atomic
        inverse_transformation = reverse_atomic @ inverse_forward_atomic
        return forward_transformation, inverse_transformation


class _AffineExtractionNetwork(_BaseTransformationExtractionNetwork):
    def __init__(
        self,
        n_dims: int,
        volume_shape: Sequence[int],
        n_features: int,
        activation_factory: IActivationFactory,
        transformation_coordinate_system: CoordinateSystem,
        affine_transformation_type: AffineTransformationTypeDefinition,
    ) -> None:
        super().__init__()
        n_affine_extraction_dimensions = 2 * int(np_prod(volume_shape)) * n_features
        n_affine_parameters = calculate_n_parameters(n_dims, affine_transformation_type)
        self._transformation_coordinate_system = transformation_coordinate_system
        self._affine_transformation_type = affine_transformation_type
        n_linears = 2
        self.linears = ModuleList(
            [
                Linear(
                    n_affine_extraction_dimensions,
                    n_affine_extraction_dimensions,
                    bias=True,
                )
                for _ in range(n_linears)
            ]
        )
        self.activations = ModuleList(
            [activation_factory.build() for _ in range(n_linears)]
        )
        self.final_linear = Linear(
            n_affine_extraction_dimensions, n_affine_parameters, bias=True
        )
        self._n_dims = n_dims

    def _extract_atomic_transformation(
        self,
        combined_input: Tensor,
    ) -> GridComposableMapping:
        output = combined_input
        for linear, activation in zip(self.linears, self.activations):
            output = linear(combined_input)
            output = activation(output)
        output = self.final_linear(output)
        affine_matrix = generate_affine_transformation_matrix(
            output, self._affine_transformation_type
        )
        return affine(affine_matrix).assign_coordinates(
            self._transformation_coordinate_system
        )

    def _modify_input(self, input_tensor: Tensor) -> Tensor:
        return input_tensor.view(input_tensor.size(0), -1)

    def _invert_mapping(
        self, mapping: GridComposableMapping, device: torch_device
    ) -> GridComposableMapping:
        return mapping.invert()


class _DenseExtractionNetwork(_BaseTransformationExtractionNetwork):
    def __init__(
        self,
        n_input_features: int,
        n_features: int,
        n_convolutions: int,
        feature_coordinate_system: CoordinateSystem,
        transformation_coordinate_system: CoordinateSystem,
        forward_fixed_point_solver: FixedPointSolver,
        backward_fixed_point_solver: FixedPointSolver,
        max_control_point_multiplier: float,
        activation_factory: IActivationFactory,
        normalizer_factory: INormalizerFactory,
    ) -> None:
        super().__init__()
        self._n_dims = len(transformation_coordinate_system.spatial_shape)
        upsampling_factor_float = (
            feature_coordinate_system.grid_spacing_cpu()
            / transformation_coordinate_system.grid_spacing_cpu()
        )
        upsampling_factor = upsampling_factor_float.round().to(dtype=long).tolist()
        self.convolutions = ConvBlockNd(
            n_convolutions=n_convolutions,
            n_input_channels=2 * n_input_features,
            n_output_channels=n_features,
            kernel_size=(3,) * self._n_dims,
            padding=1,
            activation_factory=activation_factory,
            normalizer_factory=normalizer_factory,
        )
        self.final_convolution = ConvNd(
            n_input_channels=n_features,
            n_output_channels=self._n_dims,
            kernel_size=(1,) * self._n_dims,
            padding=0,
            bias=True,
        )
        self._feature_coordinate_system = feature_coordinate_system
        self._transformation_coordinate_system = transformation_coordinate_system
        self._forward_fixed_point_solver = forward_fixed_point_solver
        self._backward_fixed_point_solver = backward_fixed_point_solver
        self._max_control_point_value = (
            max_control_point_multiplier
            * self._get_control_point_upper_bound(
                upsampling_factor,
            )
        )

    def _get_control_point_upper_bound(self, upsampling_factor: Sequence[int]) -> float:
        if tuple(upsampling_factor) in self.CONTROL_POINT_UPPER_BOUNDS_LOOKUP:
            return self.CONTROL_POINT_UPPER_BOUNDS_LOOKUP[tuple(upsampling_factor)]
        logger.info(
            "Computing cubic b-spline control point upper bound for upsampling factor %s "
            "which is not found in the lookup table. This might take a while. If you need "
            "this often, consider adding the upsampling factor to the lookup table.",
            tuple(upsampling_factor),
        )
        return compute_max_control_point_value(
            upsampling_factors=upsampling_factor,
            dtype=float64,
        ).item()

    # Precalculated upper bounds for the most common cases
    CONTROL_POINT_UPPER_BOUNDS_LOOKUP = {
        (1, 1, 1): 0.36,
        (2, 2, 2): 0.3854469526363808,
        (4, 4, 4): 0.39803114051999844,
        (8, 8, 8): 0.4007250760586097,
        (16, 16, 16): 0.40175329749498856,
        (32, 32, 32): 0.40253633025170044,
        (64, 64, 64): 0.4029187201371043,
        (128, 128, 128): 0.4031077300385704,
        (256, 256, 256): 0.4032092148874895,
        (512, 512, 512): 0.4032603042953307,
        (1, 1): 0.44999999999999996,
        (2, 2): 0.4923274169638207,
        (4, 4): 0.47980320571640545,
        (8, 8): 0.4845970764531083,
        (16, 16): 0.48612715818186963,
        (32, 32): 0.48728730908780815,
        (64, 64): 0.4879642422739664,
        (128, 128): 0.48831189411920906,
        (256, 256): 0.48848723983607806,
        (512, 512): 0.48857585860314345,
        (1024, 1024): 0.48862023712123526,
        (2048, 2048): 0.48864249909355595,
        (4096, 4096): 0.4886536264200244,
        (8192, 8192): 0.48865919697612337,
    }

    def _extract_atomic_transformation(
        self,
        combined_input: Tensor,
    ) -> GridComposableMapping:
        output = self.convolutions(combined_input)
        output = self.final_convolution(output)
        output = self._max_control_point_value * tanh(output)
        return samplable_volume(
            output,
            coordinate_system=self._feature_coordinate_system,
            data_format=DataFormat.voxel_displacements(),
            sampler=CubicSplineSampler(
                prefilter=False, mask_extrapolated_regions_for_empty_volume_mask=False
            ),
        ).resample_to(self._transformation_coordinate_system)

    def _invert_mapping(
        self, mapping: GridComposableMapping, device: torch_device
    ) -> GridComposableMapping:
        return mapping.invert(
            fixed_point_inversion_arguments={
                "forward_solver": self._forward_fixed_point_solver,
                "backward_solver": self._backward_fixed_point_solver,
            }
        ).resample()


class _MappingBuilder:
    """Builder peforming the anti-symmetric deformation updates"""

    def __init__(
        self,
        forward_affine: GridComposableMapping,
        inverse_affine: GridComposableMapping,
        resample_when_composing: bool,
    ) -> None:
        self._resample_when_composing = resample_when_composing
        self.forward_affine = forward_affine
        self.inverse_affine = inverse_affine
        self.left_forward_dense = Identity(
            device=forward_affine.device, dtype=forward_affine.dtype
        ).assign_coordinates(forward_affine)
        self.right_forward_dense = Identity(
            device=forward_affine.device, dtype=forward_affine.dtype
        ).assign_coordinates(forward_affine)
        self.left_inverse_dense = Identity(
            device=forward_affine.device, dtype=forward_affine.dtype
        ).assign_coordinates(forward_affine)
        self.right_inverse_dense = Identity(
            device=forward_affine.device, dtype=forward_affine.dtype
        ).assign_coordinates(forward_affine)

    def left_forward(self) -> GridComposableMapping:
        """Return full left forward mapping"""
        return self.forward_affine @ self.left_forward_dense

    def right_forward(self) -> GridComposableMapping:
        """Return full right forward mapping"""
        return self.inverse_affine @ self.right_forward_dense

    def left_inverse(self) -> GridComposableMapping:
        """Return full left inverse mapping"""
        return self.left_inverse_dense @ self.inverse_affine

    def right_inverse(self) -> GridComposableMapping:
        """Return full right inverse mapping"""
        return self.right_inverse_dense @ self.forward_affine

    def _resample(
        self,
        mapping: GridComposableMapping,
    ) -> GridComposableMapping:
        if self._resample_when_composing:
            return mapping.resample()
        return mapping

    def update(
        self,
        forward_dense: GridComposableMapping,
        inverse_dense: GridComposableMapping,
    ) -> None:
        """Update with mappings from new stage"""
        self.left_forward_dense = self._resample(
            self.left_forward_dense @ forward_dense
        )
        self.right_forward_dense = self._resample(
            self.right_forward_dense @ inverse_dense
        )
        self.left_inverse_dense = self._resample(
            inverse_dense @ self.left_inverse_dense
        )
        self.right_inverse_dense = self._resample(
            forward_dense @ self.right_inverse_dense
        )

    def as_mapping_pair(self, include_affine: bool = True) -> MappingPair:
        """Get current mapping as mapping pair"""
        if include_affine:
            forward = self._resample(
                self.left_forward() @ self.right_inverse(),
            )
            inverse = self._resample(
                self.right_forward() @ self.left_inverse(),
            )
        else:
            forward = self._resample(
                self.left_forward_dense @ self.right_inverse_dense,
            )
            inverse = self._resample(self.right_forward_dense @ self.left_inverse_dense)
        return MappingPair(forward, inverse)
