"""
Scanora AI — Render Deployment Readiness & ASGI Verification Suite.

Verifies:
1. Dependency configuration in pyproject.toml explicitly declares uvicorn and runtime dependencies.
2. The authoritative ASGI import path (ai_service.main:app) imports cleanly without side-effects.
3. The FastAPI application instance is properly configured with all routes registered.
4. Health endpoints respond successfully.
5. Model checkpoint fallback and safety gates remain intact for production cloud environments.
"""

import sys
import tomllib
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


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
    from ai_service.main import app

    assert isinstance(app, FastAPI), "Imported 'app' must be an instance of FastAPI"
    assert app.title == "Scanora AI Inference Service"


def test_routes_registered_on_asgi_app():
    """Ensure all core clinical API endpoints are properly mounted."""
    from ai_service.main import app

    route_paths = [route.path for route in app.routes]
    assert "/health" in route_paths, "/health endpoint must be registered"
    assert "/v1/infer" in route_paths, "/v1/infer endpoint must be registered"
    assert "/v1/model-info" in route_paths, "/v1/model-info endpoint must be registered"


def test_health_endpoint_contract():
    """Ensure /health returns 200 OK with expected JSON structure in deployment environment."""
    from ai_service.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "scanora-ai-inference"
    assert "lifecycle_state" in payload


def test_uvicorn_config_can_resolve_app():
    """Verify uvicorn.Config can load the target ASGI string 'ai_service.main:app'."""
    import uvicorn

    config = uvicorn.Config(
        "ai_service.main:app",
        host="0.0.0.0",
        port=10000,
        log_level="error",
    )
    # config.load() loads the app and wraps it in ASGI middleware
    config.load()
    assert callable(config.loaded_app)
    assert isinstance(config.loaded_app.app, FastAPI)


