"""SITReg registration applications interface"""

from logging import getLogger
from typing import Any, Mapping

from deformation_inversion_layer.fixed_point_iteration import (
    AndersonSolver,
    AndersonSolverArguments,
)
from torch import device as torch_device

from algorithm.affine_transformation import AffineTransformationTypeDefinition
from application.interface import (
    IInferenceDefinition,
    ITrainingDefinition,
    TrainingDefinitionArgs,
)
from application.sitreg.consistency_training import (
    SITRegDistributedConsistencyTraining,
)
from application.sitreg.inference import (
    SITRegInference,
)
from application.sitreg.training import (
    SITRegSegmentationTraining,
    SITRegTraining,
)
from model.sitreg import SITReg
from model.sitreg.feature_extractor import EncoderFeatureExtractor
from util.count_parameters import count_module_trainable_parameters
from util.import_util import import_object

logger = getLogger(__name__)


def create_model(
    model_config: Mapping[str, Any],
    device: torch_device,
) -> SITReg:
    """Create SITReg model from config"""
    activation_factory = import_object(f'model.activation.{model_config["activation"]["type"]}')(
        **model_config["activation"].get("args", {})
    )
    normalizer_factory = import_object(f'model.normalizer.{model_config["normalizer"]["type"]}')(
        **model_config["normalizer"].get("args", {})
    )
    feature_extractor = EncoderFeatureExtractor(
        n_input_channels=model_config["n_input_channels"],
        activation_factory=activation_factory,
        n_features_per_resolution=model_config["n_features_per_resolution"],
        n_convolutions_per_resolution=model_config["n_feature_convolutions_per_resolution"],
        input_shape=model_config["input_shape"],
        normalizer_factory=normalizer_factory,
    )
    displacement_field_inversion_config = model_config["displacement_field_inversion"]
    network = SITReg(
        feature_extractor=feature_extractor,  # type: ignore
        n_transformation_features_per_resolution=model_config[
            "n_transformation_features_per_resolution"
        ],
        n_transformation_convolutions_per_resolution=model_config[
            "n_transformation_convolutions_per_resolution"
        ],
        affine_transformation_type=(
            AffineTransformationTypeDefinition.full() if model_config["predict_affine"] else None
        ),
        input_voxel_size=model_config["voxel_size"],
        input_shape=model_config["input_shape"],
        transformation_downsampling_factor=model_config["transformation_downsampling_factor"],
        forward_fixed_point_solver=AndersonSolver(
            stop_criterion=import_object(
                displacement_field_inversion_config["forward_fixed_point_solver"]["stop_criterion"][
                    "type"
                ]
            )(
                **displacement_field_inversion_config["forward_fixed_point_solver"][
                    "stop_criterion"
                ]["args"]
            ),
            arguments=AndersonSolverArguments(
                **displacement_field_inversion_config["forward_fixed_point_solver"]["arguments"]
            ),
        ),
        backward_fixed_point_solver=AndersonSolver(
            stop_criterion=import_object(
                displacement_field_inversion_config["backward_fixed_point_solver"][
                    "stop_criterion"
                ]["type"]
            )(
                **displacement_field_inversion_config["backward_fixed_point_solver"][
                    "stop_criterion"
                ]["args"]
            ),
            arguments=AndersonSolverArguments(
                **displacement_field_inversion_config["backward_fixed_point_solver"]["arguments"]
            ),
        ),
        max_control_point_multiplier=model_config["max_control_point_multiplier"],
        activation_factory=activation_factory,
        normalizer_factory=normalizer_factory,
    )
    logger.info(
        "Initiated SITReg model with %d parameters",
        count_module_trainable_parameters(network),
    )
    network.to(device)
    return network


def create_training_definition(
    application_config: Mapping[str, Any], args: TrainingDefinitionArgs
) -> ITrainingDefinition:
    """Create training definition based on config"""
    model = create_model(application_config["model"], device=args.devices[0])
    return SITRegTraining(
        model=model,
        application_config=application_config,
        args=args,
    )


def create_distributed_consistency_training_definition(
    application_config: Mapping[str, Any], args: TrainingDefinitionArgs
) -> ITrainingDefinition:
    """Create distributed consistency training definition based on config"""
    model = create_model(application_config["model"], device=args.devices[0])
    return SITRegDistributedConsistencyTraining(
        model=model,
        application_config=application_config,
        args=args,
    )


def create_segmentation_training_definition(
    application_config: Mapping[str, Any], args: TrainingDefinitionArgs
) -> ITrainingDefinition:
    """Create training definition based on config"""
    model = create_model(application_config["model"], device=args.devices[0])
    return SITRegSegmentationTraining(
        model=model,
        application_config=application_config,
        args=args,
    )


def create_inference_definition(
    application_config: Mapping[str, Any], device: torch_device
) -> IInferenceDefinition:
    """Create training definition based on config"""
    model = create_model(application_config["model"], device=device)
    return SITRegInference(
        model=model,
        application_config=application_config,
        device=device,
    )
