"""Implementation of SITReg model"""


from abc import abstractmethod
from logging import getLogger
from math import ceil
from typing import Callable, NamedTuple, Optional, Sequence

from attr import Factory, define
from numpy import prod as np_prod
from torch import Tensor, cat, chunk, float64
from torch import device as torch_device
from torch import tanh, tensor
from torch.nn import Linear, Module, ModuleList

from algorithm.affine_transformation import (
    AffineTransformationTypeDefinition,
    calculate_n_parameters,
)
from algorithm.composable_mapping.factory import ComposableFactory, CoordinateSystemFactory
from algorithm.composable_mapping.grid_mapping import GridMappingArgs
from algorithm.composable_mapping.interface import IComposableMapping, VoxelCoordinateSystem
from algorithm.cubic_b_spline_control_point_upper_bound import compute_max_control_point_value
from algorithm.cubic_spline_upsampling import CubicSplineUpsampling
from algorithm.interface import IFixedPointSolver
from model.components import ConvolutionBlockNd
from util.ndimensional_operators import conv_nd

from .interface import IFeatureExtractor

logger = getLogger(__name__)


class MappingPair(NamedTuple):
    """Mapping pair containing both forward and inverse deformation"""

    forward_mapping: IComposableMapping
    inverse_mapping: IComposableMapping


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
        transformation_mapping_args: Defines interpolation arguments for
            transformations.
        volume_mapping_args: Defines interpolation arguments for feature
            volumes.
        forward_fixed_point_solver: Defines fixed point solver for the forward pass
            of the deformation inversion layers.
        backward_fixed_point_solver: Defines fixed point solver for the backward pass
            of the deformation inversion layers.
        max_control_point_multiplier: Optimal maximum control point values are
            multiplied with this value to ensure that the individual
            deformations are invertible even after numerical errors. This should
            be some value just below 1, e.g. 0.99.
        activation: Activation function to use.
    """

    def __init__(
        self,
        feature_extractor: IFeatureExtractor,
        n_transformation_features_per_resolution: Sequence[int],
        n_transformation_convolutions_per_resolution: Sequence[int],
        affine_transformation_type: Optional[AffineTransformationTypeDefinition],
        input_voxel_size: Sequence[float],
        input_shape: Sequence[int],
        transformation_downsampling_factor: Sequence[float],
        transformation_mapping_args: GridMappingArgs,
        volume_mapping_args: GridMappingArgs,
        forward_fixed_point_solver: IFixedPointSolver,
        backward_fixed_point_solver: IFixedPointSolver,
        max_control_point_multiplier: float,
        activation: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
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
        self._affine_extraction_network = (
            _AffineExtractionNetwork(
                n_dims=n_dims,
                volume_shape=feature_shapes[-1][1:],
                n_features=feature_shapes[-1][0],
                activation=activation,
                affine_transformation_type=affine_transformation_type,
            )
            if affine_transformation_type is not None
            else None
        )
        self._feature_coordinate_systems = [
            CoordinateSystemFactory.top_left_aligned_normalized(
                original_grid_shape=input_shape,
                grid_shape=feature_shape[1:],
                voxel_size=input_voxel_size,
                downsampling_factor=downsampling_factor,
            )
            if predict_deformation
            else None
            for (predict_deformation, feature_shape, downsampling_factor) in zip(
                predict_dense_for_stages,
                feature_shapes,
                feature_extractor.get_downsampling_factors(),
            )
        ]
        transformation_shape = [
            ceil(input_dim_size / dim_downsampling_factor)
            for input_dim_size, dim_downsampling_factor in zip(
                input_shape, transformation_downsampling_factor
            )
        ]
        self._transformation_coordinate_system = (
            CoordinateSystemFactory.top_left_aligned_normalized(
                original_grid_shape=input_shape,
                grid_shape=transformation_shape,
                voxel_size=input_voxel_size,
                downsampling_factor=transformation_downsampling_factor,
            )
        )
        self._image_coordinate_system = CoordinateSystemFactory.top_left_aligned_normalized(
            original_grid_shape=input_shape,
            voxel_size=input_voxel_size,
            downsampling_factor=[1.0] * len(input_shape),
            grid_shape=input_shape,
        )
        feature_downsampling_factors = feature_extractor.get_downsampling_factors(
            transformation_downsampling_factor
        )
        self._dense_extraction_networks = [
            _DenseExtractionNetwork(
                n_input_features=feature_shape[0],
                n_features=n_transformation_features,
                n_convolutions=n_transformation_convolutions,
                upsampling_factor=self._downsampling_factor_to_integer(upsampling_factor),
                transformation_shape=transformation_shape,
                transformation_coordinate_system=self._transformation_coordinate_system,
                transformation_mapping_args=transformation_mapping_args,
                forward_fixed_point_solver=forward_fixed_point_solver,
                backward_fixed_point_solver=backward_fixed_point_solver,
                max_control_point_multiplier=max_control_point_multiplier,
                activation=activation,
            )
            if predict_deformation
            else None
            for (
                predict_deformation,
                feature_shape,
                n_transformation_features,
                n_transformation_convolutions,
                upsampling_factor,
            ) in zip(
                predict_dense_for_stages,
                feature_shapes,
                n_transformation_features_per_resolution,
                n_transformation_convolutions_per_resolution,
                feature_downsampling_factors,
            )
        ]
        self._not_none_dense_extration_networks = ModuleList(
            [network for network in self._dense_extraction_networks if network is not None]
        )
        self._volume_mapping_args = volume_mapping_args
        self._transformation_mapping_args = transformation_mapping_args
        self._predict_dense_for_stages = predict_dense_for_stages
        self._feature_extractor = feature_extractor
        self._backward_fixed_point_solver = backward_fixed_point_solver
        self._output_downsampling_factors = [
            self._downsampling_factor_to_integer(downsampling_factor)
            for (
                predict_deformation,
                downsampling_factor,
            ) in zip(
                predict_dense_for_stages,
                feature_downsampling_factors,
            )
            if predict_deformation
        ]

    @property
    def image_coordinate_system(self) -> VoxelCoordinateSystem:
        """Return coordinate system used by the network for the inputs"""
        return self._image_coordinate_system

    @property
    def transformation_coordinate_system(self) -> VoxelCoordinateSystem:
        """Return coordinate system used by the predicted transformations"""
        return self._transformation_coordinate_system

    @property
    def output_downsampling_factors(self) -> Sequence[Sequence[int]]:
        """Return prediction downsampling factors for each output intermediate deformation"""
        return self._output_downsampling_factors

    @staticmethod
    def _downsampling_factor_to_integer(downsampling_factor: Sequence[float]) -> Sequence[int]:
        int_downsampling_factor = []
        for dim_downsampling_factor in downsampling_factor:
            if not dim_downsampling_factor.is_integer():
                raise ValueError("Only integer downsampling factors are supported")
            int_downsampling_factor.append(int(dim_downsampling_factor))
        return int_downsampling_factor

    def _extract_affine(
        self, batch_combined_features: Tensor
    ) -> tuple[IComposableMapping, IComposableMapping]:
        if self._affine_extraction_network is None:
            return ComposableFactory.create_identity(), ComposableFactory.create_identity()
        logger.debug("Starting affine transformation extraction")
        features_1, features_2 = chunk(batch_combined_features, 2)
        return self._affine_extraction_network(features_1, features_2)

    def _extract_dense(
        self,
        batch_combined_features: Tensor,
        coordinate_system: VoxelCoordinateSystem,
        mapping_builder: "_MappingBuilder",
        dense_extraction_network: "_DenseExtractionNetwork",
    ) -> tuple[IComposableMapping, IComposableMapping]:
        features_1, features_2 = chunk(batch_combined_features, 2)
        left_forward_ddf = mapping_builder.left_forward()(coordinate_system.grid)
        right_forward_ddf = mapping_builder.right_forward()(coordinate_system.grid)
        transformed_features_1 = ComposableFactory.create_volume(
            data=features_1,
            coordinate_system=coordinate_system,
            grid_mapping_args=self._volume_mapping_args,
        )(left_forward_ddf).generate_values()
        transformed_features_2 = ComposableFactory.create_volume(
            data=features_2,
            coordinate_system=coordinate_system,
            grid_mapping_args=self._volume_mapping_args,
        )(right_forward_ddf).generate_values()
        return dense_extraction_network(
            transformed_features_1,
            transformed_features_2,
        )

    def _resample_mapping(self, mapping: IComposableMapping) -> IComposableMapping:
        return ComposableFactory.resample_to_dense_mapping(
            mapping=mapping,
            coordinate_system=self._transformation_coordinate_system,
            grid_mapping_args=self._transformation_mapping_args,
        )

    def _to_level_index(self, low_first_level_index: int) -> int:
        """Obtain high resolution first level index

        Args:
            low_first_level_index: Low resolution first index, -1 refers to affine level
        """
        n_dense_levels = len(self._dense_extraction_networks)
        return n_dense_levels - 1 - low_first_level_index

    def forward(
        self,
        image_1: Tensor,
        image_2: Tensor,
        mappings_for_levels: Sequence[tuple[int, bool]] = ((0, True),),
        resample_when_composing: bool = True
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
        mappings_for_levels_set = set(mappings_for_levels)
        device = image_1.device
        batch_combined_inputs = cat((image_1, image_2), dim=0)
        batch_combined_features_list = self._feature_extractor(batch_combined_inputs)
        predict_affine = self._affine_extraction_network is not None
        forward_affine, inverse_affine = self._extract_affine(batch_combined_features_list[-1])
        mapping_builder = _MappingBuilder(
            forward_affine=forward_affine,
            inverse_affine=inverse_affine,
            transformation_coordinate_system=self._transformation_coordinate_system,
            transformation_mapping_args=self._transformation_mapping_args,
            resample_when_composing=resample_when_composing,
        )
        output_mappings: dict[tuple[int, bool], MappingPair] = {}
        affine_level_index = self._to_level_index(-1)
        for include_affine in (False, True):
            if (affine_level_index, include_affine) in mappings_for_levels_set:
                output_mappings[
                    (affine_level_index, include_affine)
                ] = mapping_builder.as_mapping_pair(device, include_affine)
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
            predict_deformation,
            batch_combined_features,
            dense_extraction_network,
            coordinate_system,
        ) in enumerate(
            zip(
                reversed(self._predict_dense_for_stages),
                reversed(batch_combined_features_list),
                reversed(self._dense_extraction_networks),
                reversed(self._feature_coordinate_systems),
            )
        ):
            if predict_deformation:
                assert coordinate_system is not None
                assert dense_extraction_network is not None
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
                    coordinate_system=coordinate_system,
                    mapping_builder=mapping_builder,
                    dense_extraction_network=dense_extraction_network,
                )
                mapping_builder.update(
                    forward_dense=forward_dense, inverse_dense=inverse_dense, device=device
                )
            level_index = self._to_level_index(low_first_level_index)
            for include_affine in (False, True):
                if (level_index, include_affine) in mappings_for_levels_set:
                    output_mappings[
                        (level_index, include_affine)
                    ] = mapping_builder.as_mapping_pair(device, include_affine)
                    if not predict_deformation:
                        logger.warning(
                            "Intermediate mapping requested for level %d "
                            "but no deformation is predicted on that level.",
                            level_index,
                        )
            del batch_combined_features_list[-1]
        return [output_mappings[mapping_index] for mapping_index in mappings_for_levels]


class _BaseTransformationExtractionNetwork(Module):
    """Base class for generating exactly inverse consistent transformations"""

    @abstractmethod
    def _extract_atomic_transformation(self, combined_input: Tensor) -> IComposableMapping:
        """Extract the smallest unit transformation"""

    @abstractmethod
    def _invert_mapping(self, mapping: IComposableMapping) -> IComposableMapping:
        """Invert transformation"""

    def _modify_input(self, input_tensor: Tensor) -> Tensor:
        return input_tensor

    def _extract_atomic_transformations(
        self, features_1: Tensor, features_2: Tensor
    ) -> tuple[IComposableMapping, IComposableMapping]:
        input_1_modified = self._modify_input(features_1)
        input_2_modified = self._modify_input(features_2)
        forward_combined_input = cat(
            (input_1_modified - input_2_modified, input_1_modified + input_2_modified), dim=1
        )
        reverse_combined_input = cat(
            (input_2_modified - input_1_modified, input_1_modified + input_2_modified), dim=1
        )
        forward_atomic = self._extract_atomic_transformation(forward_combined_input)
        reverse_atomic = self._extract_atomic_transformation(reverse_combined_input)
        return forward_atomic, reverse_atomic

    def forward(
        self, features_1: Tensor, features_2: Tensor
    ) -> tuple[IComposableMapping, IComposableMapping]:
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
            features_1=features_1, features_2=features_2
        )
        inverse_forward_atomic = self._invert_mapping(forward_atomic)
        inverse_reverse_atomic = self._invert_mapping(reverse_atomic)
        forward_transformation = forward_atomic.compose(inverse_reverse_atomic)
        inverse_transformation = reverse_atomic.compose(inverse_forward_atomic)
        return forward_transformation, inverse_transformation


class _AffineExtractionNetwork(_BaseTransformationExtractionNetwork):
    def __init__(
        self,
        n_dims: int,
        volume_shape: Sequence[int],
        n_features: int,
        activation: Callable[[Tensor], Tensor],
        affine_transformation_type: AffineTransformationTypeDefinition,
    ) -> None:
        super().__init__()
        n_affine_extraction_dimensions = 2 * int(np_prod(volume_shape)) * n_features
        n_affine_parameters = calculate_n_parameters(n_dims, affine_transformation_type)
        self._affine_transformation_type = affine_transformation_type
        self._activation = activation
        self._linear_1 = Linear(n_affine_extraction_dimensions, n_affine_extraction_dimensions)
        self._linear_2 = Linear(n_affine_extraction_dimensions, n_affine_extraction_dimensions)
        self._final_linear = Linear(n_affine_extraction_dimensions, n_affine_parameters)
        self._n_dims = n_dims

    def _extract_atomic_transformation(self, combined_input: Tensor) -> IComposableMapping:
        output = self._activation(self._linear_1(combined_input))
        output = self._activation(self._linear_2(output))
        output = self._final_linear(output)
        return ComposableFactory.create_affine_from_parameters(
            output, self._affine_transformation_type
        )

    def _modify_input(self, input_tensor: Tensor) -> Tensor:
        return input_tensor.view(input_tensor.size(0), -1)

    def _invert_mapping(self, mapping: IComposableMapping) -> IComposableMapping:
        return mapping.invert()


class _DenseExtractionNetwork(_BaseTransformationExtractionNetwork):
    def __init__(
        self,
        n_input_features: int,
        n_features: int,
        n_convolutions: int,
        upsampling_factor: Sequence[int],
        transformation_shape: Sequence[int],
        transformation_coordinate_system: VoxelCoordinateSystem,
        transformation_mapping_args: GridMappingArgs,
        forward_fixed_point_solver: IFixedPointSolver,
        backward_fixed_point_solver: IFixedPointSolver,
        max_control_point_multiplier: float,
        activation: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self._n_dims = len(upsampling_factor)
        self._convolution_block = ConvolutionBlockNd(
            n_dims=self._n_dims,
            n_convolutions=n_convolutions,
            n_input_channels=2 * n_input_features,
            n_output_channels=n_features,
            activation=activation,
        )
        self._final_conv = conv_nd(self._n_dims)(
            in_channels=n_features, out_channels=self._n_dims, kernel_size=1
        )
        self._transformation_coordinate_system = transformation_coordinate_system
        self._transformation_mapping_args = transformation_mapping_args
        self._forward_fixed_point_solver = forward_fixed_point_solver
        self._backward_fixed_point_solver = backward_fixed_point_solver
        self._cropping_tuple = (...,) + tuple(
            slice(None, transformation_dim_size) for transformation_dim_size in transformation_shape
        )
        self._upsampler = CubicSplineUpsampling(upsampling_factor)
        self._upsampling_factor = upsampling_factor
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

    def _obtain_displacement_field(self, combined_input: Tensor) -> Tensor:
        output = self._convolution_block(combined_input)
        output = self._final_conv(output)
        output = self._max_control_point_value * tanh(output)
        upsampling_factor_tensor = tensor(
            self._upsampling_factor, device=combined_input.device, dtype=combined_input.dtype
        ).view(-1, *(1,) * self._n_dims)
        upsampled_output = (
            self._upsampler(output, apply_prefiltering=False)[self._cropping_tuple]
            * upsampling_factor_tensor
        )
        return upsampled_output

    def _extract_atomic_transformation(self, combined_input: Tensor) -> IComposableMapping:
        displacement_field = self._obtain_displacement_field(combined_input)
        high_res_coordinate_mapping = ComposableFactory.create_dense_mapping(
            displacement_field=displacement_field,
            coordinate_system=self._transformation_coordinate_system,
            grid_mapping_args=self._transformation_mapping_args,
        )
        return high_res_coordinate_mapping

    def _invert_mapping(self, mapping: IComposableMapping) -> IComposableMapping:
        return mapping.invert(
            forward_fixed_point_solver=self._forward_fixed_point_solver,
            backward_fixed_point_solver=self._backward_fixed_point_solver,
        )


@define
class _MappingBuilder:
    """Builder peforming the anti-symmetric deformation updates"""

    transformation_coordinate_system: VoxelCoordinateSystem
    transformation_mapping_args: GridMappingArgs
    resample_when_composing: bool
    forward_affine: IComposableMapping = Factory(ComposableFactory.create_identity)
    inverse_affine: IComposableMapping = Factory(ComposableFactory.create_identity)
    left_forward_dense: IComposableMapping = Factory(ComposableFactory.create_identity)
    right_forward_dense: IComposableMapping = Factory(ComposableFactory.create_identity)
    left_inverse_dense: IComposableMapping = Factory(ComposableFactory.create_identity)
    right_inverse_dense: IComposableMapping = Factory(ComposableFactory.create_identity) 

    def left_forward(self) -> IComposableMapping:
        """Return full left forward mapping"""
        # pylint bug
        return self.forward_affine.compose(self.left_forward_dense)  # pylint: disable=no-member

    def right_forward(self) -> IComposableMapping:
        """Return full right forward mapping"""
        # pylint bug
        return self.inverse_affine.compose(self.right_forward_dense)  # pylint: disable=no-member

    def left_inverse(self) -> IComposableMapping:
        """Return full left inverse mapping"""
        return self.left_inverse_dense.compose(self.inverse_affine)

    def right_inverse(self) -> IComposableMapping:
        """Return full right inverse mapping"""
        return self.right_inverse_dense.compose(self.forward_affine)

    def _resample_mapping(
        self, mapping: IComposableMapping, device: Optional[torch_device] = None
    ) -> IComposableMapping:
        if self.resample_when_composing:
            return ComposableFactory.resample_to_dense_mapping(
                mapping=mapping,
                coordinate_system=self.transformation_coordinate_system,
                grid_mapping_args=self.transformation_mapping_args,
                device=device,
            )
        return mapping

    def update(
        self,
        forward_dense: IComposableMapping,
        inverse_dense: IComposableMapping,
        device: Optional[torch_device] = None,
    ) -> None:
        """Update with mappings from new stage"""
        self.left_forward_dense = self._resample_mapping(
            self.left_forward_dense.compose(forward_dense), device=device
        )
        self.right_forward_dense = self._resample_mapping(
            self.right_forward_dense.compose(inverse_dense), device=device
        )
        self.left_inverse_dense = self._resample_mapping(
            inverse_dense.compose(self.left_inverse_dense), device=device
        )
        self.right_inverse_dense = self._resample_mapping(
            forward_dense.compose(self.right_inverse_dense), device=device
        )

    def as_mapping_pair(
        self, device: Optional[torch_device] = None, include_affine: bool = True
    ) -> MappingPair:
        """Get current mapping as mapping pair"""
        if include_affine:
            forward = self._resample_mapping(
                self.left_forward().compose(self.right_inverse()),
                device=device,
            )
            inverse = self._resample_mapping(
                self.right_forward().compose(self.left_inverse()),
                device=device,
            )
        else:
            forward = self._resample_mapping(
                self.left_forward_dense.compose(self.right_inverse_dense),
                device=device,
            )
            inverse = self._resample_mapping(
                self.right_forward_dense.compose(self.left_inverse_dense), device=device
            )
        return MappingPair(forward, inverse)
