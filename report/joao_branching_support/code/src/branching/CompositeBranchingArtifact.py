from __future__ import annotations

import copy
import hashlib
import importlib
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn
import pandas as pd
import numpy as np
import scipy

from .CompositeBranchingEngine import CompositeBranchingEngine


ARTIFACT_FORMAT = "joao_composite_branching_v2"
LEGACY_ARTIFACT_FORMAT = "joao_composite_branching_v1"


def export_composite_branching_artifact(
    composite: CompositeBranchingEngine,
    path: str | Path,
    metadata: dict[str, Any],
    artifact_scope: str = "evaluation",
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    engines = copy.deepcopy(composite.engines)
    _remove_training_logs(engines)
    _reset_engine_runtime_state(engines)

    if artifact_scope not in {"evaluation", "deployment"}:
        raise ValueError("artifact_scope must be 'evaluation' or 'deployment'.")

    payload = {
        "format": ARTIFACT_FORMAT,
        "artifact_format": ARTIFACT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_scope": artifact_scope,
        "metadata": {
            **metadata,
            "artifact_format": ARTIFACT_FORMAT,
            "artifact_scope": artifact_scope,
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "runtime_state_persisted": False,
        },
        "seed": composite.seed,
        "engines": engines,
    }

    with path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = dict(payload["metadata"])
    metadata["artifact_sha256"] = artifact_sha256(path)
    return metadata


def load_composite_branching_artifact(
    path: str | Path,
    expected_scope: str | None = None,
    expected_sha256: str | None = None,
) -> CompositeBranchingEngine:
    payload = load_artifact_payload(path, expected_scope=expected_scope, expected_sha256=expected_sha256)
    engines = copy.deepcopy(payload["engines"])
    _reset_engine_runtime_state(engines)
    return CompositeBranchingEngine(
        engines=engines,
        seed=payload.get("seed", 1),
        use_default_hierarchy=False,
        train_on_init=False,
    )


def load_artifact_payload(
    path: str | Path,
    expected_scope: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if expected_sha256 is not None:
        actual = artifact_sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"Branching artifact SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    _prepare_pickle_module_aliases()
    with path.open("rb") as file:
        payload = pickle.load(file)
    artifact_format = payload.get("format")
    if artifact_format not in {ARTIFACT_FORMAT, LEGACY_ARTIFACT_FORMAT}:
        raise ValueError(f"Unsupported branching artifact format: {payload.get('format')}")
    scope = payload.get("artifact_scope") or payload.get("metadata", {}).get("artifact_scope")
    if expected_scope is not None and scope != expected_scope:
        raise ValueError(f"Branching artifact scope mismatch: expected {expected_scope}, got {scope}")
    _warn_on_version_mismatch(payload)
    return payload


def _warn_on_version_mismatch(payload: dict[str, Any]) -> None:
    import warnings

    metadata = payload.get("metadata", {})
    artifact_sklearn = metadata.get("sklearn_version")
    if artifact_sklearn and artifact_sklearn != sklearn.__version__:
        warnings.warn(
            "Branching artifact was created with scikit-learn "
            f"{artifact_sklearn}, current version is {sklearn.__version__}.",
            RuntimeWarning,
            stacklevel=2,
        )


def _prepare_pickle_module_aliases() -> None:
    """
    Keep artifact classes identical when the repo is imported as either
    ``src.*`` or ``joao.src.*`` under different PYTHONPATH layouts.
    """

    package = __package__ or ""
    if package.startswith("src."):
        preferred_root = "src"
        alias_root = "joao.src"
    elif package.startswith("joao.src."):
        preferred_root = "joao.src"
        alias_root = "src"
    else:
        return

    module_names = [
        "AttributeBasedBranchingEngine",
        "AttributeSamplingBranchingEngine",
        "BranchingLogHandler",
        "BranchingUtils",
        "CompositeBranchingEngine",
        "PredictiveBranchingEngine",
        "ProbabilityBranchingEngine",
    ]

    for module_name in module_names:
        preferred_name = f"{preferred_root}.branching.{module_name}"
        alias_name = f"{alias_root}.branching.{module_name}"
        module = importlib.import_module(preferred_name)
        sys.modules[alias_name] = module


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_training_logs(engines: list[Any]) -> None:
    for engine in engines:
        if hasattr(engine, "log"):
            engine.log = None
        for attr in ("fallback_engine", "base_engine"):
            child = getattr(engine, attr, None)
            if child is not None:
                _remove_training_logs([child])


def _reset_engine_runtime_state(engines: list[Any]) -> None:
    for engine in engines:
        if hasattr(engine, "random"):
            engine.random = random.Random(getattr(engine, "seed", 1))
        for attr in (
            "total_predictions",
            "valid_ml_predictions",
            "fallback_count",
            "rule_matches",
            "total_decisions",
            "sampled_attribute_count",
            "derived_attribute_count",
            "modified_attribute_count",
        ):
            if hasattr(engine, attr):
                setattr(engine, attr, 0)
        model = getattr(engine, "model", None)
        if model is not None and hasattr(model, "named_steps"):
            classifier = model.named_steps.get("classifier")
            if classifier is not None and hasattr(classifier, "n_jobs"):
                classifier.n_jobs = 1
        for attr in ("fallback_engine", "base_engine"):
            child = getattr(engine, attr, None)
            if child is not None:
                _reset_engine_runtime_state([child])
