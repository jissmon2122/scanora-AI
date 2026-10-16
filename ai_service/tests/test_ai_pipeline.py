"""
Scanora AI — Complete Python Unit & Integration Test Suite.
"""

import io
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
from ai_service.main import app, SCANORA_AI_API_KEY


@pytest.fixture
def sample_pil_image():
    """Generates a synthetic grayscale image mimicking breast tissue."""
    arr = np.zeros((300, 200), dtype=np.uint8)
    # Draw simple gradient/ellipse mimicking tissue
    y, x = np.ogrid[:300, :200]
    mask = ((x - 80) ** 2) / 60**2 + ((y - 150) ** 2) / 120**2 <= 1
    arr[mask] = 180
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def sample_png_bytes(sample_pil_image):
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="PNG")
    return buf.getvalue()


def test_image_validator_magic_bytes(sample_png_bytes):
    fmt = ImageValidator.detect_format(sample_png_bytes)
    assert fmt == SupportedImageFormat.PNG

    # Test corrupted bytes
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

    # Test coordinate mapping
    mapped = prep.map_box_to_original(0.2, 0.2, 0.4, 0.4)
    assert "x" in mapped and "y" in mapped
    assert "width" in mapped and "height" in mapped
    assert 0.0 <= mapped["x"] <= 100.0


def test_mammography_classifier_forward():
    model = MammographyClassifier(pretrained_backbone=False)
    dummy_input = torch.randn(2, 1, 512, 512)
    mal_logits, den_logits = model(dummy_input)

    assert mal_logits.shape == (2, 1)
    assert den_logits.shape == (2, 4)

    # Test single-image prediction
    pred = model.predict_single(dummy_input[:1])
    assert "malignancy_score" in pred
    assert "density" in pred
    assert pred["density"] in ["A", "B", "C", "D"]
    assert 0.0 <= pred["malignancy_score"] <= 100.0


def test_lesion_detector_unconfigured():
    detector = MammographyLesionDetector(pretrained_backbone=False)
    dummy_input = torch.randn(1, 1, 512, 512)
    res = detector.detect(dummy_input)

    # Must return UNAVAILABLE without fabricating boxes
    assert res["detector_status"] == "UNAVAILABLE"
    assert len(res["findings"]) == 0


def test_lesion_detector_quadrant_mapping():
    assert MammographyLesionDetector.infer_quadrant(0.8, 0.2, "right") == "Upper Outer Quadrant"
    assert MammographyLesionDetector.infer_quadrant(0.2, 0.2, "right") == "Upper Inner Quadrant"
    assert MammographyLesionDetector.infer_quadrant(0.5, 0.5, "right") == "Retroareolar"


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
    # Concordance should elevate confidence
    assert agg["breast_malignancy_prob"] >= 0.50


def test_fastapi_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "classifier_ready" in data


def test_fastapi_infer_endpoint(sample_png_bytes):
    import base64

    client = TestClient(app)
    b64_str = base64.b64encode(sample_png_bytes).decode("utf-8")

    payload = {
        "scan_id": "test-scan-001",
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

    # Request with valid bearer token
    auth_resp = client.post(
        "/v1/infer",
        json=payload,
        headers={"Authorization": f"Bearer {SCANORA_AI_API_KEY}"},
    )
    assert auth_resp.status_code == 200
    res_data = auth_resp.json()

    assert res_data["scan_id"] == "test-scan-001"
    assert "malignancy_score" in res_data
    assert "density" in res_data
    assert res_data["density"] in ["A", "B", "C", "D"]
    assert "inference_latency_ms" in res_data
    assert res_data["model_status"] in ["TRAINED_MODEL", "RESEARCH_PROTOTYPE"]
