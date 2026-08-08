from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from ml.model_artifact import (
    ModelArtifactError,
    load_model_payload,
    save_model_payload,
)
from ml.predict import MODEL_PATH


class _MaliciousPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        expression = f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')"
        return eval, (expression,)


def test_shipped_model_loads_through_restricted_unpickler() -> None:
    payload = load_model_payload(MODEL_PATH)

    assert payload["feature_cols"]
    assert callable(payload["model"].predict_proba)


def test_restricted_loader_rejects_code_execution_global(tmp_path: Path) -> None:
    marker = tmp_path / "pickle-executed.txt"
    model_path = tmp_path / "malicious.pkl"
    model_path.write_bytes(pickle.dumps(_MaliciousPayload(marker)))

    with pytest.raises(ModelArtifactError, match="builtins.eval"):
        load_model_payload(model_path)
    assert not marker.exists()


def test_model_hash_mismatch_is_rejected_before_deserialization(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(MODEL_PATH.read_bytes())
    model_path.with_suffix(".pkl.sha256").write_text("0" * 64 + "\n", encoding="ascii")

    with pytest.raises(ModelArtifactError, match="SHA-256 mismatch"):
        load_model_payload(model_path)


def test_save_emits_hash_sidecar_and_round_trips(tmp_path: Path) -> None:
    payload = load_model_payload(MODEL_PATH)
    model_path = tmp_path / "roundtrip.pkl"

    save_model_payload(payload, model_path)

    assert model_path.with_suffix(".pkl.sha256").is_file()
    loaded = load_model_payload(model_path)
    assert loaded["feature_cols"] == payload["feature_cols"]
