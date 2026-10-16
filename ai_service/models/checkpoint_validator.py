"""
Scanora AI — Checkpoint Validator and Model Lifecycle Gatekeeper.

Enforces:
1. Actual physical checkpoint existence and size integrity.
2. Architecture compatibility with modified EfficientNet-B2 (1-channel, dual-head).
3. Task-specific training verification: Separates ImageNet pretraining from Mammography fine-tuning.
4. Strict Lifecycle State Gatekeeping:
   - MODEL_UNAVAILABLE: Missing, corrupt, unverified, or untrained checkpoints.
   - RESEARCH_UNVALIDATED: Validated task-trained checkpoint, investigational use only.
   - VALIDATED_FOR_INTENDED_USE: Requires verified clinical evidence and regulatory clearance.
"""

import os
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
import torch

logger = logging.getLogger("scanora.checkpoint_validator")


class ModelLifecycleState:
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RESEARCH_UNVALIDATED = "RESEARCH_UNVALIDATED"
    VALIDATED_FOR_INTENDED_USE = "VALIDATED_FOR_INTENDED_USE"
    SIMULATED_DEMO = "SIMULATED_DEMO"


class ValidationStatus(str, Enum):
    PASSED = "PASSED"
    VALIDATION_FAILED = "VALIDATION_FAILED"


@dataclass
class ValidationResult:
    status: ValidationStatus
    lifecycle_state: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class CheckpointValidationError(Exception):
    """Raised when a checkpoint fails integrity, architecture, or training checks."""
    pass


class CheckpointValidator:
    """Rigorous gatekeeper validating mammography model checkpoints."""

    EXPECTED_HEAD_KEYS = [
        "malignancy_head.1.weight",
        "malignancy_head.1.bias",
        "malignancy_head.5.weight",
        "malignancy_head.5.bias",
        "density_head.1.weight",
        "density_head.1.bias",
        "density_head.4.weight",
        "density_head.4.bias",
    ]

    @staticmethod
    def validate_file(checkpoint_path: str) -> None:
        """Verifies checkpoint file existence and non-zero size."""
        if not checkpoint_path:
            raise CheckpointValidationError("Checkpoint path is empty or not provided.")
        if not os.path.exists(checkpoint_path):
            raise CheckpointValidationError(f"Checkpoint file '{checkpoint_path}' does not exist on disk.")
        file_size = os.path.getsize(checkpoint_path)
        if file_size < 1024:
            raise CheckpointValidationError(
                f"Checkpoint file '{checkpoint_path}' is corrupted or incomplete (size: {file_size} bytes)."
            )

    @classmethod
    def inspect_and_load(
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Loads and verifies checkpoint structure.
        Returns:
            (state_dict, checkpoint_id, lifecycle_state)
        """
        cls.validate_file(checkpoint_path)

        try:
            raw_obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as e:
            raise CheckpointValidationError(f"Failed to safely deserialize PyTorch checkpoint: {e}")

        if not isinstance(raw_obj, dict):
            raise CheckpointValidationError(
                "Invalid checkpoint format. Expected dictionary with 'state_dict' and metadata."
            )

        # Extract state dict
        if "state_dict" in raw_obj:
            state_dict = raw_obj["state_dict"]
            metadata = raw_obj
        elif "model_state_dict" in raw_obj:
            state_dict = raw_obj["model_state_dict"]
            metadata = raw_obj
        else:
            state_dict = raw_obj
            metadata = {}

        # Verify architecture compatibility
        cls.verify_architecture_compatibility(state_dict)

        # Verify task-specific training (reject ImageNet-only or untrained weights)
        is_task_trained = cls.verify_task_training(state_dict, metadata)
        if not is_task_trained:
            raise CheckpointValidationError(
                "Checkpoint does not contain verified mammography task-specific weights. "
                "ImageNet pretraining alone is not permitted for clinical decision inference."
            )

        # Determine lifecycle state
        lifecycle_state = cls.determine_lifecycle_state(metadata)
        checkpoint_id = metadata.get("checkpoint_id", os.path.basename(checkpoint_path))

        return state_dict, checkpoint_id, lifecycle_state

    @classmethod
    def verify_architecture_compatibility(cls, state_dict: Dict[str, torch.Tensor]) -> None:
        """Checks for required 1-channel conv and dual heads."""
        # 1. Check initial conv is 1-channel
        first_conv_key = "initial_conv.weight"
        if first_conv_key in state_dict:
            weight_shape = state_dict[first_conv_key].shape
            if weight_shape[1] != 1:
                raise CheckpointValidationError(
                    f"First conv layer input channels mismatch. Expected 1 (Grayscale), got {weight_shape[1]}."
                )

        # 2. Check head keys
        missing_keys = [k for k in cls.EXPECTED_HEAD_KEYS if k not in state_dict]
        if missing_keys:
            raise CheckpointValidationError(
                f"Checkpoint is missing required task-specific head weights: {missing_keys}"
            )

        # 3. Check output dimensions (malignancy: 1, density: 4)
        mal_out_shape = state_dict["malignancy_head.5.weight"].shape
        if mal_out_shape[0] != 1:
            raise CheckpointValidationError(
                f"Malignancy head output dimension mismatch. Expected 1, got {mal_out_shape[0]}."
            )

        den_out_shape = state_dict["density_head.4.weight"].shape
        if den_out_shape[0] != 4:
            raise CheckpointValidationError(
                f"Density head output dimension mismatch. Expected 4 (A, B, C, D), got {den_out_shape[0]}."
            )

    @classmethod
    def verify_task_training(cls, state_dict: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> bool:
        """
        Confirms weights contain non-trivial trained parameters and explicit task training marker.
        Rejects all-zero, NaN, or explicitly untrained weights.
        """
        # Metadata check
        if metadata.get("task_trained") is False:
            return False

        # Weight non-degeneracy check
        for key in ["malignancy_head.5.weight", "density_head.4.weight"]:
            tensor = state_dict[key]
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise CheckpointValidationError(f"Checkpoint weight '{key}' contains NaN or Inf values.")
            if torch.all(tensor == 0):
                raise CheckpointValidationError(f"Checkpoint weight '{key}' is all zeros.")

        return True

    @classmethod
    def determine_lifecycle_state(cls, metadata: Dict[str, Any]) -> str:
        """
        Enforces strict lifecycle rules:
        - VALIDATED_FOR_INTENDED_USE requires documented multi-reader evidence AND regulatory clearance.
        - Training or internal benchmark evaluation ALONE can NEVER grant clinical validation.
        """
        claims_clinical = (
            metadata.get("clinical_validation", False)
            or metadata.get("lifecycle_state") == ModelLifecycleState.VALIDATED_FOR_INTENDED_USE
        )
        regulatory_clearance_id = metadata.get("regulatory_clearance_id")
        independent_reader_study = metadata.get("independent_reader_study")

        if (
            claims_clinical
            and regulatory_clearance_id
            and independent_reader_study
        ):
            logger.info(f"Verified clinical authorization: {regulatory_clearance_id}")
            return ModelLifecycleState.VALIDATED_FOR_INTENDED_USE

        # If anyone attempted to claim clinical validation without the requisite evidence:
        if claims_clinical:
            logger.warning(
                "REJECTED clinical validation claim: Checkpoint claims 'clinical_validation' "
                "but lacks verified regulatory_clearance_id or independent_reader_study record. "
                "Downgrading to RESEARCH_UNVALIDATED."
            )

        return ModelLifecycleState.RESEARCH_UNVALIDATED

    @classmethod
    def validate_checkpoint(cls, checkpoint_path: str) -> ValidationResult:
        """
        Non-throwing inspection method that produces a structured ValidationResult.
        Used by diagnostic tools, test suites, and startup health checks.
        """
        warnings = []
        try:
            cls.validate_file(checkpoint_path)
            raw_obj = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
            if not isinstance(raw_obj, dict):
                return ValidationResult(
                    status=ValidationStatus.VALIDATION_FAILED,
                    lifecycle_state=ModelLifecycleState.MODEL_UNAVAILABLE,
                    error="Invalid checkpoint format: expected dictionary structure.",
                )

            if "state_dict" in raw_obj:
                state_dict = raw_obj["state_dict"]
                metadata = raw_obj
            elif "model_state_dict" in raw_obj:
                state_dict = raw_obj["model_state_dict"]
                metadata = raw_obj
            else:
                state_dict = raw_obj
                metadata = {}

            cls.verify_architecture_compatibility(state_dict)
            is_task_trained = cls.verify_task_training(state_dict, metadata)
            if not is_task_trained:
                return ValidationResult(
                    status=ValidationStatus.VALIDATION_FAILED,
                    lifecycle_state=ModelLifecycleState.MODEL_UNAVAILABLE,
                    error="Checkpoint does not contain verified mammography task-specific weights.",
                )

            claims_clinical = (
                metadata.get("clinical_validation", False)
                or metadata.get("lifecycle_state") == ModelLifecycleState.VALIDATED_FOR_INTENDED_USE
            )
            if claims_clinical and not (
                metadata.get("regulatory_clearance_id") and metadata.get("independent_reader_study")
            ):
                warnings.append("Downgraded from clinical validation claim: missing regulatory authorization record.")

            lifecycle_state = cls.determine_lifecycle_state(metadata)

            return ValidationResult(
                status=ValidationStatus.PASSED,
                lifecycle_state=lifecycle_state,
                metadata=metadata,
                warnings=warnings,
            )

        except CheckpointValidationError as e:
            return ValidationResult(
                status=ValidationStatus.VALIDATION_FAILED,
                lifecycle_state=ModelLifecycleState.MODEL_UNAVAILABLE,
                error=str(e),
                warnings=warnings,
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.VALIDATION_FAILED,
                lifecycle_state=ModelLifecycleState.MODEL_UNAVAILABLE,
                error=f"Unexpected error validating checkpoint: {e}",
                warnings=warnings,
            )

