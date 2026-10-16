"""
Scanora AI — FastAPI Clinical Mammography Inference Microservice.

Provides:
- GET /health: Health & Model Readiness probe (lightweight, zero-model-load liveness check)
- GET /v1/model-info: Checkpoint and architecture provenance
- POST /v1/infer: Real PyTorch inference with safe thread-safe lazy model loading
"""

import os
import time
import base64
import logging
import threading
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Header, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from ai_service.preprocessing.image_validator import ImageValidator, ImageValidationError, SupportedImageFormat
from ai_service.preprocessing.dicom_handler import DicomHandler, DicomProcessingError
from ai_service.preprocessing.mammography_preprocessor import MammographyPreprocessor
from ai_service.models.mammography_classifier import MammographyClassifier
from ai_service.models.lesion_detector import MammographyLesionDetector
from ai_service.models.checkpoint_validator import (
    ModelLifecycleState,
    CheckpointValidator,
    CheckpointValidationError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scanora.inference_service")

# Security configuration
SCANORA_AI_API_KEY = os.getenv("SCANORA_AI_API_KEY", "scanora-research-token-secure-2026")

app = FastAPI(
    title="Scanora AI Inference Service",
    description="Research-Grade Computer-Aided Detection Microservice for Mammography",
    version="1.0.0",
)

# CORS configuration with explicit origins and Vercel preview regex
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://scanora-ai.vercel.app",
    "https://scanora.vercel.app",
]

env_origins = os.getenv("ALLOWED_ORIGINS", "")
if env_origins:
    for o in env_origins.split(","):
        clean_o = o.strip()
        if clean_o and clean_o not in default_origins:
            default_origins.append(clean_o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=default_origins,
    allow_origin_regex=r"^https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelManager:
    """
    Thread-safe model manager implementing lazy initialization.
    Prevents out-of-memory crashes on memory-constrained deployments (e.g. Render 512MiB)
    by ensuring neural networks are NEVER instantiated at startup/import time, and weights
    are loaded ONLY on genuine inference requests with verified checkpoints.
    """
    _lock = threading.Lock()
    _classifier: Optional[MammographyClassifier] = None
    _detector: Optional[MammographyLesionDetector] = None
    _classifier_checkpoint_path: Optional[str] = None
    _detector_checkpoint_path: Optional[str] = None

    @classmethod
    def get_device(cls) -> torch.device:
        force_cpu = os.getenv("FORCE_CPU", "true").lower() in ("true", "1", "yes")
        if force_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    @classmethod
    def get_checkpoints_dir(cls) -> str:
        checkpoints_dir = os.getenv("SCANORA_CHECKPOINTS_DIR")
        if not checkpoints_dir or not os.path.exists(checkpoints_dir):
            file_relative_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            checkpoints_dir = file_relative_dir if os.path.exists(file_relative_dir) else "./ai_service/checkpoints"
        return checkpoints_dir

    @classmethod
    def find_classifier_checkpoint_on_disk(cls) -> Optional[str]:
        cdir = cls.get_checkpoints_dir()
        candidates = [
            os.path.join(cdir, "best_mammography_classifier.pt"),
            os.path.join(cdir, "mammography_classifier.pt"),
        ]
        for c_path in candidates:
            if os.path.exists(c_path):
                try:
                    if os.path.getsize(c_path) >= 1024:
                        return c_path
                except OSError:
                    continue
        return None

    @classmethod
    def find_detector_checkpoint_on_disk(cls) -> Optional[str]:
        cdir = cls.get_checkpoints_dir()
        candidates = [
            os.path.join(cdir, "best_lesion_detector.pt"),
            os.path.join(cdir, "lesion_detector.pt"),
        ]
        for d_path in candidates:
            if os.path.exists(d_path):
                try:
                    if os.path.getsize(d_path) >= 1024:
                        return d_path
                except OSError:
                    continue
        return None

    @classmethod
    def is_classifier_available(cls) -> bool:
        """Checks if classifier is loaded or a valid candidate checkpoint exists on disk."""
        if cls._classifier is not None:
            return cls._classifier.is_trained and cls._classifier.lifecycle_state != ModelLifecycleState.MODEL_UNAVAILABLE
        return cls.find_classifier_checkpoint_on_disk() is not None

    @classmethod
    def is_classifier_loaded(cls) -> bool:
        return cls._classifier is not None and cls._classifier.is_trained

    @classmethod
    def is_detector_loaded(cls) -> bool:
        return cls._detector is not None and cls._detector.is_trained

    @classmethod
    def get_classifier_checkpoint_id(cls) -> Optional[str]:
        if cls._classifier is not None:
            return cls._classifier.checkpoint_id
        ckpt = cls.find_classifier_checkpoint_on_disk()
        return os.path.basename(ckpt) if ckpt else None

    @classmethod
    def get_detector_checkpoint_id(cls) -> Optional[str]:
        if cls._detector is not None:
            return cls._detector.checkpoint_id
        ckpt = cls.find_detector_checkpoint_on_disk()
        return os.path.basename(ckpt) if ckpt else None

    @classmethod
    def get_classifier_lifecycle_state(cls) -> str:
        if cls._classifier is not None:
            return cls._classifier.lifecycle_state
        return ModelLifecycleState.MODEL_UNAVAILABLE

    @classmethod
    def get_detector_lifecycle_state(cls) -> str:
        if cls._detector is not None:
            return cls._detector.lifecycle_state
        return ModelLifecycleState.MODEL_UNAVAILABLE

    @classmethod
    def get_classifier(cls) -> MammographyClassifier:
        """
        Thread-safe double-checked lazy initialization of MammographyClassifier.
        Ensures pretrained_backbone=False so no weights are downloaded from torchvision URLs.
        """
        if cls._classifier is not None and cls._classifier.is_trained:
            return cls._classifier

        with cls._lock:
            if cls._classifier is not None and cls._classifier.is_trained:
                return cls._classifier

            ckpt_path = cls.find_classifier_checkpoint_on_disk()
            if not ckpt_path:
                raise CheckpointValidationError(
                    "No verified mammography task-trained classifier checkpoint found on disk. "
                    "Untrained random linear heads are strictly prohibited from generating clinical cancer risk scores."
                )

            device = cls.get_device()
            logger.info("Initializing MammographyClassifier (pretrained_backbone=False) on %s from %s", device, ckpt_path)
            model = MammographyClassifier(pretrained_backbone=False)
            model.load_checkpoint(ckpt_path, device=device)
            cls._classifier = model
            cls._classifier_checkpoint_path = ckpt_path
            return cls._classifier

    @classmethod
    def get_detector(cls) -> Optional[MammographyLesionDetector]:
        """
        Thread-safe double-checked lazy initialization of MammographyLesionDetector.
        Returns None without instantiating Faster R-CNN (saving ~205MB) if no checkpoint exists.
        """
        if cls._detector is not None and cls._detector.is_trained:
            return cls._detector

        with cls._lock:
            if cls._detector is not None and cls._detector.is_trained:
                return cls._detector

            ckpt_path = cls.find_detector_checkpoint_on_disk()
            if not ckpt_path:
                logger.info("No detector checkpoint found on disk. Lesion detector will remain unallocated.")
                return None

            device = cls.get_device()
            logger.info("Initializing MammographyLesionDetector (pretrained_backbone=False) on %s from %s", device, ckpt_path)
            model = MammographyLesionDetector(pretrained_backbone=False)
            model.load_checkpoint(ckpt_path, device=device)
            cls._detector = model
            cls._detector_checkpoint_path = ckpt_path
            return cls._detector

    @classmethod
    def set_classifier(cls, classifier: MammographyClassifier) -> None:
        with cls._lock:
            cls._classifier = classifier

    @classmethod
    def set_detector(cls, detector: MammographyLesionDetector) -> None:
        with cls._lock:
            cls._detector = detector

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._classifier = None
            cls._detector = None
            cls._classifier_checkpoint_path = None
            cls._detector_checkpoint_path = None


class ClassifierProxy:
    """Proxy object preserving backward compatibility for test fixtures and direct module access."""
    @property
    def is_trained(self) -> bool:
        if ModelManager._classifier is not None:
            return ModelManager._classifier.is_trained
        return False

    @is_trained.setter
    def is_trained(self, val: bool):
        if ModelManager._classifier is None:
            ModelManager._classifier = MammographyClassifier(pretrained_backbone=False)
        ModelManager._classifier.is_trained = val

    @property
    def lifecycle_state(self) -> str:
        if ModelManager._classifier is not None:
            return ModelManager._classifier.lifecycle_state
        return ModelManager.get_classifier_lifecycle_state()

    @lifecycle_state.setter
    def lifecycle_state(self, val: str):
        if ModelManager._classifier is None:
            ModelManager._classifier = MammographyClassifier(pretrained_backbone=False)
        ModelManager._classifier.lifecycle_state = val

    @property
    def checkpoint_id(self) -> Optional[str]:
        if ModelManager._classifier is not None:
            return ModelManager._classifier.checkpoint_id
        return ModelManager.get_classifier_checkpoint_id()

    @property
    def backbone_name(self) -> str:
        if ModelManager._classifier is not None:
            return ModelManager._classifier.backbone_name
        return "efficientnet_b2"

    def load_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None):
        if device is None:
            device = ModelManager.get_device()
        if ModelManager._classifier is None:
            model = MammographyClassifier(pretrained_backbone=False)
            model.load_checkpoint(checkpoint_path, device=device)
            ModelManager.set_classifier(model)
            return True
        return ModelManager._classifier.load_checkpoint(checkpoint_path, device=device)

    def predict_single(self, *args, **kwargs):
        clf = ModelManager.get_classifier()
        return clf.predict_single(*args, **kwargs)

    def __getattr__(self, name):
        clf = ModelManager.get_classifier()
        return getattr(clf, name)


class DetectorProxy:
    """Proxy object preserving backward compatibility for detector access."""
    @property
    def is_trained(self) -> bool:
        if ModelManager._detector is not None:
            return ModelManager._detector.is_trained
        return False

    @property
    def lifecycle_state(self) -> str:
        if ModelManager._detector is not None:
            return ModelManager._detector.lifecycle_state
        return ModelManager.get_detector_lifecycle_state()

    @property
    def checkpoint_id(self) -> Optional[str]:
        if ModelManager._detector is not None:
            return ModelManager._detector.checkpoint_id
        return ModelManager.get_detector_checkpoint_id()

    def load_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None):
        if device is None:
            device = ModelManager.get_device()
        if ModelManager._detector is None:
            model = MammographyLesionDetector(pretrained_backbone=False)
            model.load_checkpoint(checkpoint_path, device=device)
            ModelManager.set_detector(model)
            return True
        return ModelManager._detector.load_checkpoint(checkpoint_path, device=device)

    def detect(self, *args, **kwargs):
        det = ModelManager.get_detector()
        if det is None:
            return {"findings": [], "detector_status": "DETECTOR_UNAVAILABLE"}
        return det.detect(*args, **kwargs)

    def __getattr__(self, name):
        det = ModelManager.get_detector()
        if det is None:
            raise AttributeError("MammographyLesionDetector is not initialized (no checkpoint available)")
        return getattr(det, name)


# Module-level instances and proxies for backwards compatibility
device = ModelManager.get_device()
preprocessor = MammographyPreprocessor(target_size=512)
classifier = ClassifierProxy()
detector = DetectorProxy()


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
    model_status: str  # "TRAINED_MODEL", "RESEARCH_PROTOTYPE", "MODEL_UNAVAILABLE", etc.
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
    """
    Lightweight Liveness probe (<2ms).
    Distinguishes service availability from model readiness without loading PyTorch weights into memory.
    """
    return {
        "status": "healthy",
        "service": "scanora-ai-inference",
        "version": "1.0.0",
        "device": str(ModelManager.get_device()),
        "cuda_available": (
            False if os.getenv("FORCE_CPU", "true").lower() in ("true", "1", "yes")
            else torch.cuda.is_available()
        ),
        "classifier_ready": ModelManager.is_classifier_loaded(),
        "classifier_trained": ModelManager.is_classifier_loaded(),
        "classifier_checkpoint": ModelManager.get_classifier_checkpoint_id(),
        "lifecycle_state": ModelManager.get_classifier_lifecycle_state(),
        "detector_ready": ModelManager.is_detector_loaded(),
        "detector_trained": ModelManager.is_detector_loaded(),
        "detector_checkpoint": ModelManager.get_detector_checkpoint_id(),
    }


@app.get("/v1/model-info")
def model_info():
    """Returns architecture, lifecycle status, and training dataset provenance without loading weights."""
    clf_loaded = ModelManager.is_classifier_loaded()
    clf_state = ModelManager.get_classifier_lifecycle_state()
    clf_ckpt = ModelManager.get_classifier_checkpoint_id()
    det_loaded = ModelManager.is_detector_loaded()
    det_ckpt = ModelManager.get_detector_checkpoint_id()

    return {
        "service": "Scanora Mammography AI Service",
        "version": "1.0.0",
        "lifecycle_state": clf_state,
        "is_trained": clf_loaded,
        "classifier": {
            "backbone": "efficientnet_b2",
            "trained": clf_loaded,
            "checkpoint": clf_ckpt,
            "lifecycle_state": clf_state,
            "dataset_provenance": "VinDr-Mammo (FFDM) / CBIS-DDSM",
        },
        "detector": {
            "backbone": "Faster R-CNN (ResNet-50-FPN)",
            "trained": det_loaded,
            "checkpoint": det_ckpt,
            "classes": ["Mass", "Suspicious Calcification", "Architectural Distortion", "Asymmetry"],
        },
        "regulatory_notice": (
            "Investigational research prototype. Not approved for autonomous clinical diagnosis."
        ),
    }


@app.post("/v1/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest, authorization: Optional[str] = Header(None)):
    """Executes validated PyTorch computer vision inference with safe lazy model loading."""
    verify_auth_token(authorization)
    start_time = time.perf_counter()

    # Safety Gate: Prevent untrained random linear heads from generating clinical predictions
    if not ModelManager.is_classifier_available():
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

    # Lazily initialize classifier with thread-safety and local weights only
    try:
        active_classifier = ModelManager.get_classifier()
    except (CheckpointValidationError, Exception) as e:
        logger.error(f"Failed to lazily load classifier checkpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI Model Unavailable: Could not load verified checkpoint: {e}",
        )

    device = ModelManager.get_device()

    # Run PyTorch Classifier
    classification = active_classifier.predict_single(prep_res.tensor, device=device)

    # Lazily retrieve detector (remains None if no detector checkpoint on disk, avoiding OOM)
    active_detector = ModelManager.get_detector()
    if active_detector is not None:
        detection = active_detector.detect(prep_res.tensor, device=device, laterality=laterality)
    else:
        detection = {"findings": [], "detector_status": "DETECTOR_UNAVAILABLE"}

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

    model_status_label = active_classifier.lifecycle_state

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


# ---------------------------------------------------------------------------
# Clinical Workstation API Endpoints (Vercel Frontend & Integration Layer)
# ---------------------------------------------------------------------------

DEMO_COOKIE_NAME = "scanora_demo_session"
in_memory_scans: List[Dict[str, Any]] = []


class PatientInfoPayload(BaseModel):
    patientId: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=18, le=120)
    laterality: Optional[str] = "right"
    view: Optional[str] = "MLO"
    notes: Optional[str] = None


class AnalyzeScanRequest(BaseModel):
    scanId: str
    fileName: str
    fileSize: Optional[int] = None
    fileDataUrl: Optional[str] = None
    patientInfo: Optional[PatientInfoPayload] = None
    allowDemoFallback: Optional[bool] = False


def generate_simulated_analysis(
    scan_id: str,
    file_name: str,
    file_size: int = 1000,
    age: Optional[int] = None,
) -> Dict[str, Any]:
    """Generates deterministic clinical benchmark simulation strictly labeled as SIMULATED_DEMO."""
    seed_str = f"{file_name}-{file_size}-{age or 45}"
    h = 0
    for char in seed_str:
        h = ((h << 5) - h + ord(char)) & 0xFFFFFFFF
        if h >= 0x80000000:
            h -= 0x100000000
    abs_hash = abs(h)
    cat_sel = abs_hash % 100

    if cat_sel < 45:
        birads_cat = "BI-RADS 1"
        birads_desc = "Negative — Symmetrical breast tissue without architectural distortion or suspicious masses."
        mal_score = round(1.2 + (abs_hash % 15) / 10.0, 1)
        ben_score = round(100.0 - mal_score, 1)
        rec = "Routine annual screening mammography recommended."
        findings: List[Dict[str, Any]] = []
    elif cat_sel < 75:
        birads_cat = "BI-RADS 2"
        birads_desc = "Benign Findings — Involuting fibroadenoma or typical secretory calcification."
        mal_score = round(2.8 + (abs_hash % 20) / 10.0, 1)
        ben_score = round(100.0 - mal_score, 1)
        rec = "Continue routine screening mammography. No biopsy or short-term follow-up required."
        findings = []
    elif cat_sel < 90:
        birads_cat = "BI-RADS 3"
        birads_desc = "Probably Benign — Well-circumscribed focal density; malignancy risk < 2%."
        mal_score = round(4.5 + (abs_hash % 25) / 10.0, 1)
        ben_score = round(100.0 - mal_score, 1)
        rec = "Initial 6-month short-interval unilateral follow-up recommended."
        findings = []
    else:
        birads_cat = "BI-RADS 4"
        birads_desc = "Suspicious Abnormality — Indeterminate cluster or irregular microcalcifications."
        mal_score = round(32.0 + (abs_hash % 40), 1)
        ben_score = round(100.0 - mal_score, 1)
        rec = "Image-guided tissue biopsy should be strongly considered."
        findings = [
            {
                "id": f"sim-{scan_id}-1",
                "type": "Microcalcification",
                "location": "Upper Outer Quadrant",
                "confidencePercent": 88.5,
                "riskLevel": "Suspicious",
                "coordinates": {"x": 42.0, "y": 30.0, "width": 14.0, "height": 14.0},
            }
        ]

    density_options = [
        ("A", "Entirely fatty tissue (ACR Density A)"),
        ("B", "Scattered fibroglandular densities (ACR Density B)"),
        ("C", "Heterogeneously dense tissue (ACR Density C)"),
        ("D", "Extremely dense tissue (ACR Density D)"),
    ]
    chosen_density = density_options[(abs_hash >> 3) % len(density_options)]

    return {
        "scanId": scan_id,
        "modelStatus": "SIMULATED_DEMO",
        "modelArchitecture": "Synthetic Benchmark Estimator",
        "checkpointId": "simulation-fallback",
        "biradsCategory": birads_cat,
        "biradsDescription": birads_desc,
        "malignancyScore": mal_score,
        "benignScore": ben_score,
        "density": chosen_density[0],
        "densityDescription": chosen_density[1],
        "findings": findings,
        "recommendation": rec,
        "analyzedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inferenceLatencyMs": 45.0,
        "disclaimer": "DEMONSTRATION MODE: Simulated outputs. Connect the Python AI microservice for real deep learning inference.",
    }


@app.post("/api/demo-session")
@app.post("/demo-session")
def create_demo_session(request: Request, response: Response):
    """
    Initiates an authenticated Clinician Demo Session.
    Sets session cookie and returns clinician user profile.
    """
    is_secure = request.url.scheme == "https" or os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1")
    response.set_cookie(
        key=DEMO_COOKIE_NAME,
        value="active",
        max_age=60 * 60 * 24 * 7,
        httponly=True,
        samesite="none" if is_secure else "lax",
        secure=is_secure,
    )
    return {
        "success": True,
        "message": "Demo Clinician session established.",
        "user": {
            "id": "demo-clinician",
            "email": "dr.clinician@scanora.health",
            "google_user_data": {
                "name": "Dr. Alex Taylor, MD",
                "email": "dr.clinician@scanora.health",
                "email_verified": True,
                "sub": "demo-clinician-sub",
            },
            "last_signed_in_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


@app.get("/api/users/me")
@app.get("/users/me")
def get_current_user(request: Request):
    """Returns authenticated clinician or user profile; checks cookie or Authorization header."""
    cookie = request.cookies.get(DEMO_COOKIE_NAME)
    auth_header = request.headers.get("Authorization", "")
    if cookie == "active" or "demo-clinician" in auth_header or "scanora_demo_session" in auth_header:
        return {
            "id": "demo-clinician",
            "email": "dr.clinician@scanora.health",
            "google_user_data": {
                "name": "Dr. Alex Taylor, MD",
                "email": "dr.clinician@scanora.health",
                "email_verified": True,
                "sub": "demo-clinician-sub",
            },
            "last_signed_in_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: No active clinician or user session",
    )


@app.post("/api/logout")
@app.get("/api/logout")
@app.post("/logout")
@app.get("/logout")
def terminate_session(response: Response):
    """Terminates session and clears cookie."""
    response.delete_cookie(key=DEMO_COOKIE_NAME, path="/")
    return {"success": True, "message": "Clinician session terminated."}


@app.get("/api/scans")
@app.get("/scans")
def list_clinical_scans():
    """Lists clinical scan records."""
    return {"scans": in_memory_scans}


@app.post("/api/scans/analyze")
@app.post("/scans/analyze")
def analyze_clinical_scan(req: AnalyzeScanRequest):
    """Executes validated clinical scan analysis (real PyTorch model inference or explicit demo simulation)."""
    start_time = time.perf_counter()
    analysis: Optional[Dict[str, Any]] = None

    # If model is available and image provided, perform real inference
    if ModelManager.is_classifier_available() and req.fileDataUrl:
        try:
            raw_base64 = req.fileDataUrl
            if "," in raw_base64:
                raw_base64 = raw_base64.split(",")[1]
            file_bytes = base64.b64decode(raw_base64)
            fmt = ImageValidator.detect_format(file_bytes)
            if fmt == SupportedImageFormat.UNSUPPORTED:
                raise HTTPException(status_code=400, detail="Unsupported image format.")

            laterality = (req.patientInfo.laterality if req.patientInfo and req.patientInfo.laterality else "right")
            view = (req.patientInfo.view if req.patientInfo and req.patientInfo.view else "MLO")

            if fmt == SupportedImageFormat.DICOM:
                pixel_array, _ = DicomHandler.parse_dicom(file_bytes)
                prep_res = preprocessor.preprocess(pixel_array)
            else:
                image, _ = ImageValidator.validate_and_load_raster(file_bytes)
                prep_res = preprocessor.preprocess(image)

            active_classifier = ModelManager.get_classifier()
            device_target = ModelManager.get_device()
            classification = active_classifier.predict_single(prep_res.tensor, device=device_target)

            # Map to BI-RADS
            prob = classification["calibrated_malignancy_probability"]
            if prob < 0.02:
                birads_cat = "BI-RADS 1"
                birads_desc = "Negative — Symmetrical breast tissue, no suspicious lesion."
                rec = "Routine annual screening mammography recommended."
            elif prob < 0.08:
                birads_cat = "BI-RADS 2"
                birads_desc = "Benign Findings — Involuting fibroadenoma or secretory calcification."
                rec = "Continue routine screening mammography."
            elif prob < 0.20:
                birads_cat = "BI-RADS 3"
                birads_desc = "Probably Benign — Well-circumscribed focal density; low probability of malignancy."
                rec = "Initial 6-month short-interval unilateral mammographic follow-up."
            elif prob < 0.60:
                birads_cat = "BI-RADS 4"
                birads_desc = "Suspicious Abnormality — Indeterminate mass or clustered microcalcifications."
                rec = "Targeted ultrasound and image-guided core biopsy strongly recommended."
            else:
                birads_cat = "BI-RADS 5"
                birads_desc = "Highly Suggestive of Malignancy — High probability lesion (>95% risk)."
                rec = "Urgent diagnostic biopsy and oncology multidisciplinary review."

            # Detector
            active_detector = ModelManager.get_detector()
            if active_detector is not None:
                det = active_detector.detect(prep_res.tensor, device=device_target, laterality=laterality)
                raw_findings = det.get("findings", [])
            else:
                raw_findings = []

            mapped_findings = []
            for f in raw_findings:
                raw_box = f["raw_model_box"]
                coords = prep_res.map_box_to_original(
                    raw_box["xmin"], raw_box["ymin"], raw_box["xmax"], raw_box["ymax"]
                )
                mapped_findings.append({
                    "id": f["id"],
                    "type": f["type"],
                    "location": f["location"],
                    "confidencePercent": f["confidencePercent"],
                    "riskLevel": f["riskLevel"],
                    "coordinates": coords,
                })

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            analysis = {
                "scanId": req.scanId,
                "modelStatus": active_classifier.lifecycle_state,
                "modelArchitecture": classification["model_architecture"],
                "checkpointId": classification["checkpoint_id"],
                "biradsCategory": birads_cat,
                "biradsDescription": birads_desc,
                "malignancyScore": classification["malignancy_score"],
                "benignScore": classification["benign_score"],
                "calibratedProbability": classification["calibrated_malignancy_probability"],
                "isIndeterminate": classification["is_indeterminate_case"],
                "density": classification["density"],
                "densityDescription": classification["density_description"],
                "findings": mapped_findings,
                "recommendation": rec,
                "analyzedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "inferenceLatencyMs": round(latency_ms, 2),
                "disclaimer": "RESEARCH PROTOTYPE: Results generated by deep learning model for investigational use only. Not for independent primary diagnostic use.",
            }
        except Exception as e:
            logger.warning(f"Model inference failed: {e}")
            if not req.allowDemoFallback:
                raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Fallback simulation if model unavailable and allowed
    if analysis is None:
        if req.allowDemoFallback:
            age = req.patientInfo.age if req.patientInfo else None
            analysis = generate_simulated_analysis(req.scanId, req.fileName, req.fileSize or 1000, age)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI Model Unavailable: No verified task-trained checkpoint is loaded and demo fallback was not requested.",
            )

    scan_record = {
        "id": req.scanId,
        "userEmail": "dr.clinician@scanora.health",
        "fileName": req.fileName,
        "fileSize": req.fileSize or 0,
        "previewUrl": req.fileDataUrl,
        "uploadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "patientInfo": req.patientInfo.model_dump() if req.patientInfo else {
            "laterality": "right",
            "view": "MLO",
        },
        "analysis": analysis,
    }
    in_memory_scans.insert(0, scan_record)
    return {"success": True, "scan": scan_record}


@app.delete("/api/scans/{scan_id}")
@app.delete("/scans/{scan_id}")
def delete_clinical_scan(scan_id: str):
    global in_memory_scans
    in_memory_scans = [s for s in in_memory_scans if s.get("id") != scan_id]
    return {"success": True, "message": f"Scan {scan_id} deleted."}


@app.get("/api/oauth/google/redirect_url")
def oauth_google_redirect():
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Google OAuth is not configured on this instance. Please use Clinician Demo Access.",
    )


@app.get("/api/health")
def api_health():
    return health()

