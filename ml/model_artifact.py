from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import io
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


_ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
    # NumPy 1.x used numpy.core while NumPy 2.x uses numpy._core.
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("sklearn.calibration", "CalibratedClassifierCV"),
    ("sklearn.calibration", "_CalibratedClassifier"),
    ("sklearn.calibration", "_SigmoidCalibration"),
    ("sklearn.linear_model._logistic", "LogisticRegression"),
    ("sklearn.pipeline", "Pipeline"),
    ("sklearn.preprocessing._data", "StandardScaler"),
}


class ModelArtifactError(RuntimeError):
    """Raised when a saved model is corrupt, unsupported, or unsafe to deserialize."""


@dataclass(frozen=True)
class LoadedModelArtifact:
    """Validated payload plus identity of the exact bytes deserialized."""

    payload: dict[str, Any]
    model_sha256: str
    sidecar_declared_sha256: str | None
    sidecar_sha256: str | None


class _RestrictedModelUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise ModelArtifactError(
                f"Model artifact references unsupported global {module}.{name}; "
                "only locally trained sklearn logistic artifacts are accepted."
            )
        return super().find_class(module, name)


def _sha256_sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".sha256")


def _validate_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ModelArtifactError("Model payload must be a dictionary.")
    if "model" not in payload or "feature_cols" not in payload:
        raise ModelArtifactError("Model payload must contain model and feature_cols.")

    feature_cols = payload["feature_cols"]
    if (
        not isinstance(feature_cols, (list, tuple))
        or not feature_cols
        or any(not isinstance(column, str) or not column for column in feature_cols)
        or len(set(feature_cols)) != len(feature_cols)
    ):
        raise ModelArtifactError("Model feature_cols must be a non-empty list of unique names.")
    if not callable(getattr(payload["model"], "predict_proba", None)):
        raise ModelArtifactError("Model payload does not provide predict_proba().")
    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ModelArtifactError("Model metadata must be a dictionary when present.")
    return payload


def _deserialize_model_payload(raw: bytes, *, model_path: Path) -> dict[str, Any]:
    stream = io.BytesIO(raw)
    try:
        payload = _RestrictedModelUnpickler(stream).load()
    except ModelArtifactError:
        raise
    except (pickle.PickleError, AttributeError, EOFError, ImportError, IndexError, TypeError, ValueError) as exc:
        raise ModelArtifactError(f"Model artifact could not be deserialized: {model_path}") from exc
    if stream.read(1):
        raise ModelArtifactError(f"Model artifact contains trailing data: {model_path}")
    return _validate_payload(payload)


def load_model_payload_with_evidence(model_path: Path) -> LoadedModelArtifact:
    """
    Load the model and retain identity for the exact byte buffer deserialized.

    A matching ``.sha256`` sidecar is verified when present. Legacy artifacts without a
    sidecar remain readable, but are still constrained by the restricted unpickler.
    """

    model_path = Path(model_path)
    if not model_path.is_file():
        raise ModelArtifactError(f"Model file not found: {model_path}")
    try:
        raw = model_path.read_bytes()
    except OSError as exc:
        raise ModelArtifactError(f"Model file is unreadable: {model_path}") from exc

    model_sha256 = hashlib.sha256(raw).hexdigest()
    sidecar_path = _sha256_sidecar_path(model_path)
    sidecar_declared_sha256: str | None = None
    sidecar_sha256: str | None = None
    if sidecar_path.exists():
        try:
            sidecar_raw = sidecar_path.read_bytes()
            expected = sidecar_raw.decode("ascii").strip().split()[0].lower()
        except (OSError, IndexError, UnicodeError) as exc:
            raise ModelArtifactError(f"Model hash sidecar is unreadable: {sidecar_path}") from exc
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ModelArtifactError(f"Model hash sidecar is invalid: {sidecar_path}")
        if not hmac.compare_digest(model_sha256, expected):
            raise ModelArtifactError(f"Model SHA-256 mismatch: {model_path}")
        sidecar_declared_sha256 = expected
        sidecar_sha256 = hashlib.sha256(sidecar_raw).hexdigest()

    return LoadedModelArtifact(
        payload=_deserialize_model_payload(raw, model_path=model_path),
        model_sha256=model_sha256,
        sidecar_declared_sha256=sidecar_declared_sha256,
        sidecar_sha256=sidecar_sha256,
    )


def load_model_payload(model_path: Path) -> dict[str, Any]:
    """Load the repository model format, preserving the historical payload API."""

    return load_model_payload_with_evidence(model_path).payload


def save_model_payload(payload: dict[str, Any], model_path: Path) -> None:
    """Atomically save a validated model payload and its SHA-256 integrity sidecar."""

    _validate_payload(payload)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar_path = _sha256_sidecar_path(model_path)

    model_temp_path: Path | None = None
    sidecar_temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=model_path.parent, delete=False) as handle:
            model_temp_path = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            dir=model_path.parent,
            delete=False,
        ) as handle:
            sidecar_temp_path = Path(handle.name)
            handle.write(f"{digest}\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(model_temp_path, model_path)
        model_temp_path = None
        os.replace(sidecar_temp_path, sidecar_path)
        sidecar_temp_path = None
    finally:
        if model_temp_path is not None:
            model_temp_path.unlink(missing_ok=True)
        if sidecar_temp_path is not None:
            sidecar_temp_path.unlink(missing_ok=True)
