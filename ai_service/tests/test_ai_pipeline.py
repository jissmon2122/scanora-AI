"""
Scanora AI — Complete Python Unit & Integration Test Suite.
Phase 2: Model Readiness, Safety Gates & Verification.
"""

import io
import os
import base64
import tempfile
import pytest
import numpy as np
from PIL import Image
import torch
from fastapi.testclient import TestClient

from ai_service.preprocessing.image_validator import ImageValidator, SupportedImageFormat, ImageValidationError
from ai_service.preprocessing.mammography_preprocessor import MammographyPreprocessor
from ai_service.models.mammography_classifier import MammographyClassifier
from ai_service.models.lesion_detector import MammographyLesionDetector
from ai_service.models.multi_view_aggregator import MultiViewAggregator
from ai_service.models.checkpoint_validator import CheckpointValidator, ValidationStatus, ModelLifecycleState
from ai_service.training.evaluate import compute_metrics, compute_bootstrap_ci, aggregate_patient_predictions
import ai_service.main as main_module
from ai_service.main import app, SCANORA_AI_API_KEY


@pytest.fixture
def sample_pil_image():
    """Generates a synthetic grayscale image mimicking breast tissue."""
    arr = np.zeros((300, 200), dtype=np.uint8)
    y, x = np.ogrid[:300, :200]
    mask = ((x - 80) ** 2) / 60**2 + ((y - 150) ** 2) / 120**2 <= 1
    arr[mask] = 180
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def sample_png_bytes(sample_pil_image):
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_task_trained_checkpoint(tmp_path):
    """Creates a temporary valid synthetic research checkpoint with task_trained=True."""
    model = MammographyClassifier(pretrained_backbone=False)
    ckpt_path = tmp_path / "valid_research_checkpoint.pt"
    ckpt_data = {
        "model_state_dict": model.state_dict(),
        "task_trained": True,
        "lifecycle_state": "RESEARCH_UNVALIDATED",
        "dataset_source": "SYNTHETIC_TEST_FIXTURE",
        "architecture": "MammographyClassifier",
        "epoch": 5,
        "val_auc": 0.82,
    }
    torch.save(ckpt_data, ckpt_path)
    return str(ckpt_path)


# ---------------------------------------------------------------------------
# Preprocessing and Image Validation Tests
# ---------------------------------------------------------------------------

def test_image_validator_magic_bytes(sample_png_bytes):
    fmt = ImageValidator.detect_format(sample_png_bytes)
    assert fmt == SupportedImageFormat.PNG

    fake_bytes = b"NOT_AN_IMAGE_HEADER_AT_ALL_1234567890"
    assert ImageValidator.detect_format(fake_bytes) == SupportedImageFormat.UNSUPPORTED


def test_image_validator_empty_file():
    with pytest.raises(ImageValidationError):
        ImageValidator.validate_and_load_raster(b"")


def test_mammography_preprocessor(sample_pil_image):
    preprocessor = MammographyPreprocessor(target_size=512)
    prep = preprocessor.preprocess(sample_pil_image)

    assert prep.tensor.shape == (1, 1, 512, 512)
    assert prep.original_width == 200
    assert prep.original_height == 300

    mapped = prep.map_box_to_original(0.2, 0.2, 0.4, 0.4)
    assert "x" in mapped and "y" in mapped
    assert "width" in mapped and "height" in mapped
    assert 0.0 <= mapped["x"] <= 100.0


# ---------------------------------------------------------------------------
# Model Architecture & Safety Gate Tests
# ---------------------------------------------------------------------------

def test_mammography_classifier_raw_forward():
    model = MammographyClassifier(pretrained_backbone=False)
    dummy_input = torch.randn(2, 1, 512, 512)
    mal_logits, den_logits = model(dummy_input)

    assert mal_logits.shape == (2, 1)
    assert den_logits.shape == (2, 4)
    assert model.is_trained is False
    assert model.lifecycle_state == ModelLifecycleState.MODEL_UNAVAILABLE


def test_mammography_classifier_untrained_safety_gate():
    """Verify that an untrained model strictly refuses to generate cancer risk scores."""
    model = MammographyClassifier(pretrained_backbone=False)
    dummy_input = torch.randn(1, 1, 512, 512)

    with pytest.raises(RuntimeError) as exc_info:
        model.predict_single(dummy_input)
    assert "Inference blocked" in str(exc_info.value)
    assert "Untrained random linear heads are strictly prohibited" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Checkpoint Validator Tests
# ---------------------------------------------------------------------------

def test_checkpoint_validator_missing_file():
    res = CheckpointValidator.validate_checkpoint("non_existent_file.pt")
    assert res.status == ValidationStatus.VALIDATION_FAILED
    assert res.lifecycle_state == ModelLifecycleState.MODEL_UNAVAILABLE
    assert "not exist" in res.error.lower() or "not found" in res.error.lower()


def test_checkpoint_validator_corrupt_or_truncated_file(tmp_path):
    small_file = tmp_path / "corrupt.pt"
    small_file.write_bytes(b"PK\x03\x04short")
    res = CheckpointValidator.validate_checkpoint(str(small_file))
    assert res.status == ValidationStatus.VALIDATION_FAILED
    assert "too small" in res.error.lower() or "corrupted" in res.error.lower()


def test_checkpoint_validator_incompatible_architecture(tmp_path):
    incompatible_file = tmp_path / "incompatible.pt"
    fake_weights = {
        "model_state_dict": {
            "unrelated_layer.weight": torch.randn(10, 10),
            "unrelated_layer.bias": torch.randn(10),
        },
        "task_trained": True,
    }
    torch.save(fake_weights, incompatible_file)
    res = CheckpointValidator.validate_checkpoint(str(incompatible_file))
    assert res.status == ValidationStatus.VALIDATION_FAILED
    assert "missing" in res.error.lower() and "head" in res.error.lower()



def test_checkpoint_validator_unauthorized_clinical_claim(tmp_path):
    """Model claiming VALIDATED_FOR_INTENDED_USE without clinical auth token must be downgraded."""
    model = MammographyClassifier(pretrained_backbone=False)
    ckpt_path = tmp_path / "unauthorized_clinical.pt"
    ckpt_data = {
        "model_state_dict": model.state_dict(),
        "task_trained": True,
        "lifecycle_state": "VALIDATED_FOR_INTENDED_USE",
    }
    torch.save(ckpt_data, ckpt_path)

    res = CheckpointValidator.validate_checkpoint(str(ckpt_path))
    assert res.status == ValidationStatus.PASSED
    assert res.lifecycle_state == ModelLifecycleState.RESEARCH_UNVALIDATED
    assert any("downgraded" in w.lower() for w in res.warnings)


def test_checkpoint_validator_valid_research_checkpoint(sample_task_trained_checkpoint):
    res = CheckpointValidator.validate_checkpoint(sample_task_trained_checkpoint)
    assert res.status == ValidationStatus.PASSED
    assert res.lifecycle_state == ModelLifecycleState.RESEARCH_UNVALIDATED
    assert res.metadata["task_trained"] is True


def test_classifier_loads_valid_checkpoint(sample_task_trained_checkpoint):
    model = MammographyClassifier(pretrained_backbone=False)
    success = model.load_checkpoint(sample_task_trained_checkpoint)
    assert success is True
    assert model.is_trained is True
    assert model.lifecycle_state == ModelLifecycleState.RESEARCH_UNVALIDATED

    dummy_input = torch.randn(1, 1, 512, 512)
    pred = model.predict_single(dummy_input)
    assert "malignancy_score" in pred
    assert "density" in pred
    assert pred["density"] in ["A", "B", "C", "D"]
    assert 0.0 <= pred["malignancy_score"] <= 100.0
    assert "prediction_entropy" in pred



# ---------------------------------------------------------------------------
# Lesion Detector Tests
# ---------------------------------------------------------------------------

def test_lesion_detector_unconfigured():
    detector = MammographyLesionDetector(pretrained_backbone=False)
    dummy_input = torch.randn(1, 1, 512, 512)
    res = detector.detect(dummy_input)

    assert res["detector_status"] == "UNAVAILABLE"
    assert len(res["findings"]) == 0


def test_lesion_detector_quadrant_mapping():
    assert MammographyLesionDetector.infer_quadrant(0.8, 0.2, "right") == "Upper Outer Quadrant"
    assert MammographyLesionDetector.infer_quadrant(0.2, 0.2, "right") == "Upper Inner Quadrant"
    assert MammographyLesionDetector.infer_quadrant(0.5, 0.5, "right") == "Retroareolar"


# ---------------------------------------------------------------------------
# Multi-View Aggregator Tests
# ---------------------------------------------------------------------------

def test_multi_view_aggregator():
    views_data = [
        {
            "view": "CC",
            "calibrated_malignancy_probability": 0.45,
            "findings": [{"type": "Mass"}],
        },
        {
            "view": "MLO",
            "calibrated_malignancy_probability": 0.50,
            "findings": [{"type": "Mass"}],
        },
    ]

    agg = MultiViewAggregator.aggregate_breast_views(views_data, "right")
    assert agg["is_complete_study"] is True
    assert agg["has_orthogonal_concordance"] is True
    assert agg["breast_malignancy_prob"] >= 0.50


# ---------------------------------------------------------------------------
# FastAPI Endpoints & Safety Gate Tests
# ---------------------------------------------------------------------------

def test_fastapi_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "lifecycle_state" in data
    assert "classifier_ready" in data


def test_fastapi_model_info_endpoint():
    client = TestClient(app)
    response = client.get("/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "lifecycle_state" in data
    assert "is_trained" in data
    assert "regulatory_notice" in data



def test_fastapi_infer_untrained_safety_gate_503(sample_png_bytes):
    """When no trained checkpoint is loaded, inference must return HTTP 503 SERVICE_UNAVAILABLE."""
    # Ensure classifier is in untrained state
    main_module.classifier.is_trained = False
    main_module.classifier.lifecycle_state = ModelLifecycleState.MODEL_UNAVAILABLE

    client = TestClient(app)
    b64_str = base64.b64encode(sample_png_bytes).decode("utf-8")

    payload = {
        "scan_id": "test-untrained-scan",
        "file_name": "test_mammo.png",
        "file_base64": b64_str,
        "patient_id": "PAT-TEST-1",
        "patient_age": 50,
        "laterality": "right",
        "view": "MLO",
    }

    # Request without token should be 401
    unauth_resp = client.post("/v1/infer", json=payload)
    assert unauth_resp.status_code == 401

    # Request with valid bearer token must return 503
    auth_resp = client.post(
        "/v1/infer",
        json=payload,
        headers={"Authorization": f"Bearer {SCANORA_AI_API_KEY}"},
    )
    assert auth_resp.status_code == 503
    err_body = auth_resp.json()
    assert "detail" in err_body
    assert "Model Unavailable" in str(err_body["detail"])


def test_fastapi_infer_with_trained_research_checkpoint(sample_png_bytes, sample_task_trained_checkpoint):
    """When a legitimate research checkpoint is loaded, inference returns HTTP 200."""
    main_module.classifier.load_checkpoint(sample_task_trained_checkpoint)

    client = TestClient(app)
    b64_str = base64.b64encode(sample_png_bytes).decode("utf-8")

    payload = {
        "scan_id": "test-trained-scan",
        "file_name": "test_mammo.png",
        "file_base64": b64_str,
        "patient_id": "PAT-TEST-1",
        "patient_age": 50,
        "laterality": "right",
        "view": "MLO",
    }

    auth_resp = client.post(
        "/v1/infer",
        json=payload,
        headers={"Authorization": f"Bearer {SCANORA_AI_API_KEY}"},
    )
    assert auth_resp.status_code == 200
    res_data = auth_resp.json()

    assert res_data["scan_id"] == "test-trained-scan"
    assert "malignancy_score" in res_data
    assert "density" in res_data
    assert res_data["density"] in ["A", "B", "C", "D"]
    assert "inference_latency_ms" in res_data
    assert res_data["model_status"] == "RESEARCH_UNVALIDATED"
    assert res_data.get("disclaimer") is not None



def test_fastapi_infer_malformed_image():
    """Sending corrupt/non-image bytes returns HTTP 400."""
    client = TestClient(app)
    b64_garbage = base64.b64encode(b"THIS_IS_NOT_AN_IMAGE_FILE").decode("utf-8")

    payload = {
        "scan_id": "test-bad-scan",
        "file_name": "corrupt.png",
        "file_base64": b64_garbage,
        "patient_id": "PAT-TEST-BAD",
        "patient_age": 45,
        "laterality": "left",
        "view": "CC",
    }

    resp = client.post(
        "/v1/infer",
        json=payload,
        headers={"Authorization": f"Bearer {SCANORA_AI_API_KEY}"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Evaluation Pipeline Tests (FNR, Bootstrap CI, Density, Max-Pooling)
# ---------------------------------------------------------------------------

def test_evaluation_pipeline_fnr_and_bootstrap_ci():
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    y_probs = np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.4, 0.85, 0.05, 0.75, 0.15])

    metrics = compute_metrics(y_true, y_probs, threshold=0.5)

    assert "false_negative_rate_fnr" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert "expected_calibration_error_ece" in metrics

    # 1 of 5 positive cases (y_probs[2]=0.2) is below 0.5 threshold -> FNR = 1/5 = 0.2
    assert abs(metrics["false_negative_rate_fnr"] - 0.2) < 1e-4

    # Bootstrap 95% CI
    ci_lower, ci_upper = compute_bootstrap_ci(y_true, y_probs, n_bootstraps=200, seed=42)
    assert 0.0 <= ci_lower <= ci_upper <= 1.0


def test_evaluation_patient_level_aggregation():
    predictions = [
        {"patient_id": "P1", "y_true": 1, "calibrated_probability": 0.3, "density": "B"},
        {"patient_id": "P1", "y_true": 1, "calibrated_probability": 0.8, "density": "B"},
        {"patient_id": "P2", "y_true": 0, "calibrated_probability": 0.1, "density": "C"},
        {"patient_id": "P2", "y_true": 0, "calibrated_probability": 0.2, "density": "C"},
    ]

    p_true, p_probs = aggregate_patient_predictions(predictions)
    assert len(p_true) == 2
    assert len(p_probs) == 2
    # P1 max pooled is 0.8
    assert abs(p_probs[0] - 0.8) < 1e-4
    # P2 max pooled is 0.2
    assert abs(p_probs[1] - 0.2) < 1e-4
