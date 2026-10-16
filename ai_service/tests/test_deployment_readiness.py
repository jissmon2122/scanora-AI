"""
Scanora AI — Render Deployment Readiness & ASGI Verification Suite.

Verifies:
1. Dependency configuration in pyproject.toml explicitly declares uvicorn and runtime dependencies.
2. The authoritative ASGI import path (ai_service.main:app) imports cleanly without side-effects.
3. The FastAPI application instance is properly configured with all routes registered.
4. Health endpoints respond successfully without loading heavy PyTorch models.
5. Lazy initialization protects startup memory (< 512 MiB constraint).
6. Model checkpoint fallback and safety gates remain intact for production cloud environments.
"""

import sys
import tomllib
import time
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_service.main import app, ModelManager, SCANORA_AI_API_KEY
from ai_service.models.checkpoint_validator import ModelLifecycleState


def test_pyproject_contains_required_runtime_dependencies():
    """Verify that pyproject.toml specifies uvicorn, fastapi, and required ML libraries."""
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "pyproject.toml must exist in repository root"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Check project.dependencies (PEP 621)
    deps = data.get("project", {}).get("dependencies", [])
    dep_names = [d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in deps]

    assert "uvicorn" in dep_names, "uvicorn must be listed in pyproject.toml dependencies"
    assert "fastapi" in dep_names, "fastapi must be listed in pyproject.toml dependencies"
    assert "pydantic" in dep_names, "pydantic must be listed in pyproject.toml dependencies"
    assert "torch" in dep_names, "torch must be listed in pyproject.toml dependencies"
    assert "torchvision" in dep_names, "torchvision must be listed in pyproject.toml dependencies"


def test_asgi_import_path_resolves():
    """Verify that 'ai_service.main:app' imports successfully as a FastAPI application."""
    assert isinstance(app, FastAPI), "Imported 'app' must be an instance of FastAPI"
    assert app.title == "Scanora AI Inference Service"


def test_routes_registered_on_asgi_app():
    """Ensure all core clinical API endpoints are properly mounted."""
    route_paths = [route.path for route in app.routes]
    assert "/health" in route_paths, "/health endpoint must be registered"
    assert "/v1/infer" in route_paths, "/v1/infer endpoint must be registered"
    assert "/v1/model-info" in route_paths, "/v1/model-info endpoint must be registered"


def test_health_endpoint_contract():
    """Ensure /health returns 200 OK with expected JSON structure in deployment environment."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "scanora-ai-inference"
    assert "lifecycle_state" in payload


def test_health_endpoint_zero_allocation_speed():
    """Ensure /health executes quickly (<100ms) and does not eagerly instantiate models into RAM."""
    ModelManager.reset()
    client = TestClient(app)

    start = time.perf_counter()
    resp = client.get("/health")
    duration_ms = (time.perf_counter() - start) * 1000.0

    assert resp.status_code == 200
    assert duration_ms < 100.0, f"/health took {duration_ms}ms, expected < 100ms"
    payload = resp.json()
    assert payload["status"] == "healthy"
    # Models must NOT have been instantiated by querying /health
    assert ModelManager._classifier is None
    assert ModelManager._detector is None


def test_uvicorn_config_can_resolve_app():
    """Verify uvicorn.Config can load the target ASGI string 'ai_service.main:app'."""
    import uvicorn

    config = uvicorn.Config(
        "ai_service.main:app",
        host="0.0.0.0",
        port=10000,
        log_level="error",
    )
    config.load()
    assert callable(config.loaded_app)
    assert isinstance(config.loaded_app.app, FastAPI)


def test_inference_safety_gate_without_checkpoint_fails_fast():
    """Verify that missing checkpoint returns HTTP 503 without allocating models."""
    ModelManager.reset()
    client = TestClient(app)

    dummy_payload = {
        "scan_id": "test-unavail-scan",
        "file_name": "mammo.png",
        "file_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    }

    resp = client.post(
        "/v1/infer",
        json=dummy_payload,
        headers={"Authorization": f"Bearer {SCANORA_AI_API_KEY}"},
    )

    assert resp.status_code == 503
    assert "AI Model Unavailable" in resp.json()["detail"]
    # Model instances should remain unallocated
    assert ModelManager._classifier is None
    assert ModelManager._detector is None
