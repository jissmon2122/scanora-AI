"""
Scanora AI — FastAPI Clinical Mammography Inference Microservice.

Provides:
- GET /health: Health & Model Readiness probe
- GET /v1/model-info: Checkpoint and architecture provenance
- POST /v1/infer: Real PyTorch inference (classification + lesion detection)
"""

import os
import time
import base64
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from ai_service.preprocessing.image_validator import ImageValidator, ImageValidationError, SupportedImageFormat
from ai_service.preprocessing.dicom_handler import DicomHandler, DicomProcessingError
from ai_service.preprocessing.mammography_preprocessor import MammographyPreprocessor
from ai_service.models.mammography_classifier import MammographyClassifier
from ai_service.models.lesion_detector import MammographyLesionDetector
from ai_service.models.checkpoint_validator import ModelLifecycleState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scanora.inference_service")

# Security configuration
SCANORA_AI_API_KEY = os.getenv("SCANORA_AI_API_KEY", "scanora-research-token-secure-2026")

app = FastAPI(
    title="Scanora AI Inference Service",
    description="Research-Grade Computer-Aided Detection Microservice for Mammography",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessor = MammographyPreprocessor(target_size=512)

classifier = MammographyClassifier(pretrained_backbone=True).to(device)
detector = MammographyLesionDetector(pretrained_backbone=True).to(device)

# Attempt to load trained checkpoints if present
CHECKPOINTS_DIR = os.getenv("SCANORA_CHECKPOINTS_DIR", "./ai_service/checkpoints")
classifier_path = os.path.join(CHECKPOINTS_DIR, "best_mammography_classifier.pt")
detector_path = os.path.join(CHECKPOINTS_DIR, "best_lesion_detector.pt")

if os.path.exists(classifier_path):
    try:
        classifier.load_checkpoint(classifier_path, device=device)
        logger.info(f"Loaded classifier checkpoint: {classifier_path}")
    except Exception as e:
        logger.warning(f"Could not load classifier checkpoint: {e}")

if os.path.exists(detector_path):
    try:
        detector.load_checkpoint(detector_path, device=device)
        logger.info(f"Loaded detector checkpoint: {detector_path}")
    except Exception as e:
        logger.warning(f"Could not load detector checkpoint: {e}")


# Request & Response Schemas
class InferenceRequest(BaseModel):
    scan_id: str
    file_name: str
    file_base64: str
    patient_id: Optional[str] = None
    patient_age: Optional[int] = Field(default=None, ge=18, le=120)
    laterality: Optional[str] = Field(default="right")
    view: Optional[str] = Field(default="MLO")


class LesionItem(BaseModel):
    id: str
    type: str
    location: str
    confidencePercent: float
    riskLevel: str
    coordinates: Dict[str, float]


class InferenceResponse(BaseModel):
    scan_id: str
    model_status: str  # "TRAINED_MODEL", "RESEARCH_PROTOTYPE", "DETECTOR_UNAVAILABLE"
    model_architecture: str
    checkpoint_id: str
    calibrated_malignancy_probability: float
    malignancy_score: float
    benign_score: float
    density: str
    density_description: str
    density_confidence: float
    is_indeterminate: bool
    findings: List[LesionItem]
    view_evaluated: str
    laterality_evaluated: str
    inference_latency_ms: float
    disclaimer: str


def verify_auth_token(authorization: Optional[str] = Header(None)):
    """Verifies service-to-service Bearer authentication token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != SCANORA_AI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired service authentication token.",
        )


@app.get("/health")
def health():
    """Health & Readiness probe. Distinguishes service status from model readiness."""
    return {
        "status": "healthy",
        "service": "scanora-ai-inference",
        "version": "1.0.0",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "classifier_ready": classifier.is_trained,
        "classifier_trained": classifier.is_trained,
        "classifier_checkpoint": classifier.checkpoint_id,
        "lifecycle_state": classifier.lifecycle_state,
        "detector_ready": detector.is_trained,
        "detector_trained": detector.is_trained,
        "detector_checkpoint": detector.checkpoint_id,
    }


@app.get("/v1/model-info")
def model_info():
    """Returns architecture, lifecycle status, and training dataset provenance."""
    return {
        "service": "Scanora Mammography AI Service",
        "version": "1.0.0",
        "lifecycle_state": classifier.lifecycle_state,
        "is_trained": classifier.is_trained,
        "classifier": {
            "backbone": classifier.backbone_name,
            "trained": classifier.is_trained,
            "checkpoint": classifier.checkpoint_id,
            "lifecycle_state": classifier.lifecycle_state,
            "dataset_provenance": "VinDr-Mammo (FFDM) / CBIS-DDSM",
        },
        "detector": {
            "backbone": "Faster R-CNN (ResNet-50-FPN)",
            "trained": detector.is_trained,
            "checkpoint": detector.checkpoint_id,
            "classes": ["Mass", "Suspicious Calcification", "Architectural Distortion", "Asymmetry"],
        },
        "regulatory_notice": (
            "Investigational research prototype. Not approved for autonomous clinical diagnosis."
        ),
    }


@app.post("/v1/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest, authorization: Optional[str] = Header(None)):
    """Executes validated PyTorch computer vision inference."""
    verify_auth_token(authorization)
    start_time = time.perf_counter()

    # Safety Gate: Prevent untrained random linear heads from generating clinical predictions
    if not classifier.is_trained or classifier.lifecycle_state == ModelLifecycleState.MODEL_UNAVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "AI Model Unavailable: No verified mammography task-trained checkpoint is loaded. "
                "Untrained random linear heads are strictly prohibited from generating clinical cancer risk scores."
            ),
        )

    # Decode base64 payload
    try:
        raw_base64 = req.file_base64
        if "," in raw_base64:
            raw_base64 = raw_base64.split(",")[1]
        file_bytes = base64.b64decode(raw_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")

    # Validate file format & integrity
    fmt = ImageValidator.detect_format(file_bytes)
    if fmt == SupportedImageFormat.UNSUPPORTED:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Valid DICOM, PNG, JPEG, or WebP files required.",
        )

    # Ingest and decode image
    try:
        if fmt == SupportedImageFormat.DICOM:
            pixel_array, dicom_meta = DicomHandler.parse_dicom(file_bytes)
            prep_res = preprocessor.preprocess(pixel_array)
            laterality = req.laterality or dicom_meta.get("laterality", "right")
            view = req.view or dicom_meta.get("view_position", "MLO")
        else:
            image, meta = ImageValidator.validate_and_load_raster(file_bytes)
            prep_res = preprocessor.preprocess(image)
            laterality = req.laterality or "right"
            view = req.view or "MLO"
    except (ImageValidationError, DicomProcessingError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Image decompression error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image processing failure: {e}")

    # Run PyTorch Classifier
    classification = classifier.predict_single(prep_res.tensor, device=device)

    # Run PyTorch Detector
    detection = detector.detect(prep_res.tensor, device=device, laterality=laterality)

    # Map detector boxes back to original image space
    mapped_findings: List[LesionItem] = []
    for f in detection.get("findings", []):
        raw_box = f["raw_model_box"]
        coords = prep_res.map_box_to_original(
            raw_box["xmin"], raw_box["ymin"], raw_box["xmax"], raw_box["ymax"]
        )
        mapped_findings.append(
            LesionItem(
                id=f["id"],
                type=f["type"],
                location=f["location"],
                confidencePercent=f["confidencePercent"],
                riskLevel=f["riskLevel"],
                coordinates=coords,
            )
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    model_status_label = classifier.lifecycle_state

    return InferenceResponse(
        scan_id=req.scan_id,
        model_status=model_status_label,
        model_architecture=classification["model_architecture"],
        checkpoint_id=classification["checkpoint_id"],
        calibrated_malignancy_probability=classification["calibrated_malignancy_probability"],
        malignancy_score=classification["malignancy_score"],
        benign_score=classification["benign_score"],
        density=classification["density"],
        density_description=classification["density_description"],
        density_confidence=classification["density_confidence"],
        is_indeterminate=classification["is_indeterminate_case"],
        findings=mapped_findings,
        view_evaluated=view,
        laterality_evaluated=laterality,
        inference_latency_ms=round(latency_ms, 2),
        disclaimer=(
            "RESEARCH PROTOTYPE: Results generated by deep learning model for investigational use only. "
            "Not for independent primary diagnostic use."
        ),
    )
