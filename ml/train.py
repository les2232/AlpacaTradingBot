"""
ml/train.py
-----------
Train and evaluate ML models using the feature pipeline.

Two models are trained:
  - LogisticRegression on CORE_FEATURE_COLS  (primary — compatible with live loop)
  - RandomForestClassifier on ALL_FEATURE_COLS (comparison — offline only)

Only the logistic model is saved.  It is saved as a dict containing the fitted
pipeline, the feature column list, and metadata so the loader never has to
guess what features were used.

Usage:
    python -m ml.train                          # uses newest dataset
    python -m ml.train --dataset datasets/...  # use a specific dataset
"""

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Resolve paths relative to this file so the script works from any cwd.
_HERE        = Path(__file__).resolve().parent
DATASETS_DIR = _HERE.parent / "datasets"
MODELS_DIR   = _HERE / "models"
MODEL_PATH   = MODELS_DIR / "logistic_latest.pkl"

from ml.feature_pipeline import (
    ALL_FEATURE_COLS,
    CORE_FEATURE_COLS,
    LABEL_COL,
    LABEL_HORIZON,
    LABEL_THRESHOLD,
    build_features_multi_symbol,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Logistic regression — matches backtest_runner._train_ml_model exactly.
LR_C            = 0.5
LR_MAX_ITER     = 1000
LR_CLASS_WEIGHT = "balanced"
LR_RANDOM_STATE = 42

# Random forest — offline comparison only, not saved.
RF_N_ESTIMATORS = 100
RF_CLASS_WEIGHT = "balanced"
RF_RANDOM_STATE = 42
RF_MAX_DEPTH    = 10   # cap depth to avoid extreme overfitting

TRAIN_FRACTION  = 0.80   # chronological; rest is test


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _newest_dataset(datasets_dir: Path) -> Path:
    candidates = [p for p in datasets_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No dataset folders found in '{datasets_dir}'")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _split_per_symbol(
    df: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split applied independently per symbol.

    Each symbol is split at its own (train_fraction * n) row index so no
    future data leaks into the training set and all symbols contribute
    equally to both sets.
    """
    train_parts, test_parts = [], []
    for _symbol, group in df.groupby("symbol", sort=True):
        n     = len(group)
        split = int(n * train_fraction)
        train_parts.append(group.iloc[:split])
        test_parts.append(group.iloc[split:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_data_summary(
    raw_rows: int,
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    total = len(train_df) + len(test_df)
    print(f"  Raw bars         : {raw_rows:,}")
    print(f"  Feature rows     : {total:,}  (dropped {raw_rows - total:,} warm-up + label rows)")
    print(f"  Train rows       : {len(train_df):,}")
    print(f"  Test  rows       : {len(test_df):,}")
    print()

    # Date ranges per split (all symbols share the same timestamps)
    train_start = train_df.timestamp.min().strftime("%Y-%m-%d")
    train_end   = train_df.timestamp.max().strftime("%Y-%m-%d")
    test_start  = test_df.timestamp.min().strftime("%Y-%m-%d")
    test_end    = test_df.timestamp.max().strftime("%Y-%m-%d")
    print(f"  Train period     : {train_start} to {train_end}")
    print(f"  Test  period     : {test_start} to {test_end}")
    print()

    # Class balance
    train_pos = train_df[LABEL_COL].mean()
    test_pos  = test_df[LABEL_COL].mean()
    print(f"  Label threshold  : >{LABEL_THRESHOLD*100:.1f}% over {LABEL_HORIZON} bars")
    print(f"  Train target=1   : {train_pos:.1%}")
    print(f"  Test  target=1   : {test_pos:.1%}")
    print()


def _evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> dict:
    """Run predictions and return a metrics dict."""
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "label":     label,
        "rows":      len(y),
        "accuracy":  accuracy_score(y, y_pred),
        "roc_auc":   roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred, zero_division=0),
        "f1":        f1_score(y, y_pred, zero_division=0),
    }


def _print_metrics_table(results: list[dict]) -> None:
    hdr = f"  {'Model':<28} {'Rows':>6}  {'Acc':>6}  {'AUC':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(
            f"  {r['label']:<28} {r['rows']:>6}"
            f"  {r['accuracy']:>6.3f}"
            f"  {r['roc_auc']:>6.3f}"
            f"  {r['precision']:>6.3f}"
            f"  {r['recall']:>6.3f}"
            f"  {r['f1']:>6.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _train_logistic(X_train: pd.DataFrame, y_train: pd.Series):
    """
    StandardScaler + LogisticRegression + Platt calibration.
    Parameters match backtest_runner._train_ml_model for live-loop compatibility.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            C=LR_C,
            class_weight=LR_CLASS_WEIGHT,
            max_iter=LR_MAX_ITER,
            random_state=LR_RANDOM_STATE,
        )),
    ])
    # cv=5 calibration gives better probability estimates than a fixed split
    # when training offline on the full train set.
    calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def _train_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Random forest on ALL_FEATURE_COLS.  Saved only for comparison — not
    persisted to disk, because ALL_FEATURE_COLS != CORE_FEATURE_COLS so it
    is not a drop-in for the live loop.
    """
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _save_model(
    model,
    feature_cols: list[str],
    dataset_path: Path,
    train_rows: int,
    test_metrics: dict,
    output_path: Path,
) -> None:
    """
    Save a dict containing the model + everything predict.py needs.
    Saves as plain pickle — no joblib dependency required.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":        model,
        "feature_cols": feature_cols,
        "metadata": {
            "trained_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset":          str(dataset_path),
            "train_rows":       train_rows,
            "label_horizon":    LABEL_HORIZON,
            "label_threshold":  LABEL_THRESHOLD,
            "test_roc_auc":     round(test_metrics["roc_auc"],   4),
            "test_f1":          round(test_metrics["f1"],        4),
            "test_precision":   round(test_metrics["precision"], 4),
            "test_recall":      round(test_metrics["recall"],    4),
        },
    }
    with open(output_path, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models for the trading bot.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset directory. Defaults to newest in datasets/.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else _newest_dataset(DATASETS_DIR)
    bars_path    = dataset_path / "bars.parquet"
    if not bars_path.exists():
        print(f"ERROR: bars.parquet not found at {bars_path}", file=sys.stderr)
        sys.exit(1)

    # ── 1. Load and build features ─────────────────────────────────────────
    print()
    print("Loading dataset ...")
    raw = pd.read_parquet(bars_path)
    print(f"  {dataset_path.name}")
    print(f"  {len(raw):,} raw bars  |  {raw['symbol'].nunique()} symbols: "
          f"{', '.join(sorted(raw['symbol'].unique()))}")
    print()

    print("Building features ...")
    df = build_features_multi_symbol(raw, include_label=True)
    print(f"  {len(df):,} feature rows ready")
    print()

    # ── 2. Chronological split per symbol ──────────────────────────────────
    print("Splitting data ...")
    train_df, test_df = _split_per_symbol(df, TRAIN_FRACTION)
    _print_data_summary(len(raw), train_df, test_df)

    X_train_core = train_df[CORE_FEATURE_COLS]
    X_test_core  = test_df[CORE_FEATURE_COLS]
    X_train_all  = train_df[ALL_FEATURE_COLS]
    X_test_all   = test_df[ALL_FEATURE_COLS]
    y_train      = train_df[LABEL_COL].astype(int)
    y_test       = test_df[LABEL_COL].astype(int)

    # ── 3. Train ────────────────────────────────────────────────────────────
    print("Training LogisticRegression (core features) ...")
    lr_model = _train_logistic(X_train_core, y_train)
    print("  Done.")
    print()

    print("Training RandomForestClassifier (all features) ...")
    rf_model = _train_random_forest(X_train_all, y_train)
    print("  Done.")
    print()

    # ── 4. Evaluate ────────────────────────────────────────────────────────
    print("Evaluating ...")
    results = [
        _evaluate(lr_model, X_train_core, y_train, "LogisticReg  train (core)"),
        _evaluate(lr_model, X_test_core,  y_test,  "LogisticReg  test  (core)"),
        _evaluate(rf_model, X_train_all,  y_train, "RandomForest train (all) "),
        _evaluate(rf_model, X_test_all,   y_test,  "RandomForest test  (all) "),
    ]
    _print_metrics_table(results)

    # ── 5. Save logistic model only ─────────────────────────────────────────
    print("Saving logistic model ...")
    lr_test_metrics = results[1]   # test metrics for LR
    _save_model(
        model        = lr_model,
        feature_cols = CORE_FEATURE_COLS,
        dataset_path = dataset_path,
        train_rows   = len(train_df),
        test_metrics = lr_test_metrics,
        output_path  = MODEL_PATH,
    )
    print()
    print("Done.")
    print()


if __name__ == "__main__":
    main()
