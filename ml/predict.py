"""
ml/predict.py
-------------
Lightweight inference module for the live trading loop.

Public API
----------
predict_proba(feature_vector)          -> float
to_ml_signal(feature_vector, ...)      -> MlSignal

Both functions use a module-level singleton loaded on first call (lazy init).
Re-importing or calling predict_proba a second time does NOT reload from disk.

Errors
------
RuntimeError  — model file not found (call load() explicitly to surface early)
ValueError    — feature_vector has wrong length
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).resolve().parent
MODEL_PATH = _HERE / "models" / "logistic_latest.pkl"

# ---------------------------------------------------------------------------
# Defaults — mirror backtest_runner DEFAULT_ML_PROBABILITY_BUY / _SELL
# ---------------------------------------------------------------------------

DEFAULT_BUY_THRESHOLD  = 0.55
DEFAULT_SELL_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_payload: Optional[dict] = None   # loaded on first call


def load(path: Path = MODEL_PATH) -> dict:
    """
    Load (or reload) the model from *path*.

    Returns the raw payload dict so callers can inspect metadata.
    Raises RuntimeError if the file does not exist.
    """
    global _payload
    if not path.exists():
        raise RuntimeError(
            f"Model file not found: {path}\n"
            "Run `python -m ml.train` to create it."
        )
    with open(path, "rb") as fh:
        _payload = pickle.load(fh)
    return _payload


def _ensure_loaded() -> dict:
    if _payload is None:
        load()
    return _payload  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------

def predict_proba(feature_vector: "list[float]") -> float:
    """
    Return the model's estimated probability that price will rise > threshold
    over the next LABEL_HORIZON bars.

    Parameters
    ----------
    feature_vector : list[float]
        Must have exactly len(CORE_FEATURE_COLS) = 7 elements, ordered as:
        ret_1, ret_3, ret_5, dist_sma_10, dist_sma_20, rolling_vol, vol_ratio

    Returns
    -------
    float in [0, 1]

    Raises
    ------
    RuntimeError  if model file is missing
    ValueError    if feature_vector has the wrong length
    """
    payload = _ensure_loaded()
    expected = len(payload["feature_cols"])
    actual   = len(feature_vector)
    if actual != expected:
        raise ValueError(
            f"predict_proba: expected {expected} features "
            f"({payload['feature_cols']}), got {actual}"
        )
    # Wrap in a named DataFrame to avoid sklearn feature-name warnings.
    X    = pd.DataFrame([feature_vector], columns=payload["feature_cols"])
    prob = payload["model"].predict_proba(X)[0][1]
    return float(prob)


def to_ml_signal(
    feature_vector:  "list[float]",
    buy_threshold:  float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
) -> "MlSignal":
    """
    Run inference and wrap the result in an MlSignal for the strategy engine.

    Parameters
    ----------
    feature_vector  : 7-element list matching CORE_FEATURE_COLS
    buy_threshold   : probability above which strategy should consider buying
    sell_threshold  : probability below which strategy should consider selling

    Returns
    -------
    MlSignal  (imported from strategy; frozen dataclass)
    """
    # Import here so ml/ has no hard dependency on strategy.py at module load.
    # The live loop already imports strategy before calling this function.
    from strategy import MlSignal

    payload  = _ensure_loaded()
    prob     = predict_proba(feature_vector)
    metadata = payload.get("metadata", {})

    # model_age_seconds: seconds since the model was trained (UTC).
    trained_at_str = metadata.get("trained_at", "")
    model_age      = 0.0
    if trained_at_str:
        try:
            from datetime import datetime, timezone
            trained_at = datetime.fromisoformat(trained_at_str.replace("Z", "+00:00"))
            model_age  = (datetime.now(timezone.utc) - trained_at).total_seconds()
        except Exception:
            pass

    return MlSignal(
        probability_up   = prob,
        confidence       = abs(prob - 0.5) * 2.0,
        training_rows    = metadata.get("train_rows", 0),
        model_age_seconds= model_age,
        feature_names    = tuple(payload["feature_cols"]),
        buy_threshold    = buy_threshold,
        sell_threshold   = sell_threshold,
        validation_rows  = 0,
        model_name       = "logistic",
    )


# ---------------------------------------------------------------------------
# Self-test  (python -m ml.predict)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("Loading model ...")
    try:
        pl = load()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    meta = pl["metadata"]
    print(f"  Trained at   : {meta.get('trained_at', 'unknown')}")
    print(f"  Dataset      : {meta.get('dataset', 'unknown')}")
    print(f"  Train rows   : {meta.get('train_rows', '?'):,}")
    print(f"  Test AUC     : {meta.get('test_roc_auc', '?')}")
    print(f"  Test F1      : {meta.get('test_f1', '?')}")
    print(f"  Features     : {pl['feature_cols']}")
    print()

    # -- predict_proba with a zero vector (neutral, near-50% expected) ------
    n_features = len(pl["feature_cols"])
    zero_vec   = [0.0] * n_features
    prob       = predict_proba(zero_vec)
    print(f"predict_proba(zeros) = {prob:.4f}")

    # -- to_ml_signal -------------------------------------------------------
    signal = to_ml_signal(zero_vec)
    print(f"to_ml_signal(zeros)  = {signal}")
    print()

    # -- wrong-length error -------------------------------------------------
    print("Testing wrong-length input ...")
    try:
        predict_proba([0.0] * (n_features + 1))
        print("  ERROR: expected ValueError was not raised")
    except ValueError as exc:
        print(f"  Caught expected ValueError: {exc}")
    print()

    # -- timing -------------------------------------------------------------
    import timeit
    rounds = 1000
    elapsed = timeit.timeit(lambda: predict_proba(zero_vec), number=rounds)
    print(f"Inference time : {elapsed/rounds*1e6:.1f} us / call  ({rounds} calls)")
    print()
    print("Self-test passed.")
    print()
