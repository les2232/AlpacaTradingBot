from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_runner import _generate_walk_forward_windows, load_dataset
from run_edge_audit import _frame_text, _load_runtime_payload


DEFAULT_LIVE_CONFIG_PATH = Path("config") / "live_config.json"
DEFAULT_TREND_CONFIG_PATH = Path("config") / "trend_pullback.example.json"
DEFAULT_AUDIT_HORIZONS = (5, 10, 20)
DEFAULT_TREND_SMA_BARS = 50
DEFAULT_RECENT_HIGH_LOOKBACK = 20
DEFAULT_ATR_BARS = 14
DEFAULT_MAX_POSITIONS = 5
DEFAULT_POSITION_SIZE = 1000.0
DEFAULT_COMMISSION_PER_ORDER = 0.01
DEFAULT_SLIPPAGE_PER_SHARE = 0.05
DEFAULT_TRAIN_DAYS = 40
DEFAULT_TEST_DAYS = 20
DEFAULT_STEP_DAYS = 20
SCORE_MODE_RETURN_20 = "return_20"
SCORE_MODE_RETURN_20_PLUS_60 = "return_20_plus_60"
SCORE_MODE_TREND_CONSISTENCY_20 = "trend_consistency_20"
SCORE_MODE_TREND_CONSISTENCY_20_X_SLOPE_20 = "trend_consistency_20_x_slope_20"
SCORE_MODE_TREND_CONSISTENCY_20_X_SLOPE_20_OVER_ATR_20 = "trend_consistency_20_x_slope_20_over_atr_20"
SCORE_MODE_CHOICES = (
    SCORE_MODE_RETURN_20,
    SCORE_MODE_RETURN_20_PLUS_60,
    SCORE_MODE_TREND_CONSISTENCY_20,
    SCORE_MODE_TREND_CONSISTENCY_20_X_SLOPE_20,
    SCORE_MODE_TREND_CONSISTENCY_20_X_SLOPE_20_OVER_ATR_20,
)


@dataclass(frozen=True)
class StrategyVariant:
    family: str
    score_mode: str
    rank_lookback_bars: int
    eligible_percent: float
    hold_bars: int
    pullback_depth_atr: float | None = None
    use_early_no_follow_through_exit: bool = False
    use_recent_follow_through_filter: bool = False

    @property
    def variant_id(self) -> str:
        parts = [
            self.family,
            self.score_mode,
            f"lb{self.rank_lookback_bars}",
            f"top{int(round(self.eligible_percent * 100))}",
            f"hold{self.hold_bars}",
        ]
        if self.pullback_depth_atr is not None:
            parts.append(f"pb{self.pullback_depth_atr:.1f}")
        if self.use_early_no_follow_through_exit:
            parts.append("enf")
        if self.use_recent_follow_through_filter:
            parts.append("rftf")
        return "_".join(parts)


@dataclass(frozen=True)
class FoldWindow:
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def infer_clean_sip_dataset(datasets_root: Path) -> Path:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for manifest_path in datasets_root.glob("**/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("feed") != "sip":
            continue
        if payload.get("session_filter") != "regular":
            continue
        if payload.get("dataset_variant") != "sip_clean_regular_session":
            continue
        created = (
            pd.Timestamp(payload.get("created_at_utc"))
            if payload.get("created_at_utc")
            else pd.Timestamp.min.tz_localize("UTC")
        )
        candidates.append((created, manifest_path.parent))
    if not candidates:
        raise RuntimeError("No clean SIP regular-session dataset found. Pass --dataset explicitly.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def normalize_symbols(symbols: list[str] | tuple[str, ...] | None) -> list[str]:
    if not symbols:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return normalized


def load_runtime_payload(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    payload = _load_runtime_payload(config_path)
    runtime = dict(payload["runtime"])
    source = payload["source"] if isinstance(payload["source"], dict) else {}
    dataset = source.get("dataset") if isinstance(source.get("dataset"), str) else None
    return payload["payload"], runtime, dataset


def resolve_dataset_and_symbols(
    *,
    dataset: str | None,
    config_path: Path,
    symbols: list[str] | None,
) -> tuple[Path, list[str], dict[str, Any]]:
    _, runtime, source_dataset = load_runtime_payload(config_path)
    dataset_path = (
        Path(dataset)
        if dataset
        else (Path(source_dataset) if source_dataset else infer_clean_sip_dataset(Path("datasets")))
    )
    df, manifest = load_dataset(dataset_path)
    manifest_symbols = normalize_symbols(manifest.get("symbols"))
    runtime_symbols = normalize_symbols(runtime.get("symbols"))
    resolved_symbols = normalize_symbols(symbols) or runtime_symbols or manifest_symbols or normalize_symbols(
        df["symbol"].astype(str).tolist()
    )
    if not resolved_symbols:
        raise RuntimeError("No research symbols could be resolved from CLI, runtime, manifest, or dataset content.")
    available = set(normalize_symbols(df["symbol"].astype(str).tolist()))
    active = [symbol for symbol in resolved_symbols if symbol in available]
    if not active:
        raise RuntimeError("Resolved research symbols are not present in the dataset.")
    return dataset_path, active, manifest


def load_research_panel(
    *,
    dataset_path: Path,
    symbols: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, manifest = load_dataset(dataset_path)
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["symbol"].isin(symbols)].copy()
    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df["timestamp"] <= end_ts]
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No dataset rows remain after applying the requested filters.")
    return df, manifest


def normalize_score_mode(score_mode: str) -> str:
    normalized = str(score_mode or SCORE_MODE_RETURN_20).strip().lower()
    if normalized not in SCORE_MODE_CHOICES:
        raise ValueError(
            f"Unsupported score_mode {score_mode!r}. Choose from: {', '.join(SCORE_MODE_CHOICES)}."
        )
    return normalized


def normalize_rank_lookbacks_for_score_mode(
    score_mode: str,
    rank_lookbacks: tuple[int, ...],
) -> tuple[int, ...]:
    score_mode = normalize_score_mode(score_mode)
    normalized = tuple(sorted({int(value) for value in rank_lookbacks}))
    if not normalized:
        raise ValueError("Expected at least one rank lookback.")
    if normalized != (20,):
        raise ValueError(
            f"score_mode {score_mode!r} uses a fixed 20-bar anchor for this research step. "
            "Use rank lookbacks of exactly: 20."
        )
    return normalized


def _atr_series(group: pd.DataFrame, atr_bars: int) -> pd.Series:
    prev_close = group["close"].shift(1)
    true_range = pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - prev_close).abs(),
            (group["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(atr_bars, min_periods=atr_bars).mean()


def prepare_panel_features(
    df: pd.DataFrame,
    *,
    rank_lookback_bars: int,
    audit_horizons: tuple[int, ...],
    score_mode: str = SCORE_MODE_RETURN_20,
    trend_sma_bars: int = DEFAULT_TREND_SMA_BARS,
    recent_high_lookback: int = DEFAULT_RECENT_HIGH_LOOKBACK,
    atr_bars: int = DEFAULT_ATR_BARS,
) -> pd.DataFrame:
    score_mode = normalize_score_mode(score_mode)
    if int(rank_lookback_bars) != 20:
        raise ValueError(
            f"score_mode {score_mode!r} uses a fixed 20-bar anchor for this research step. "
            "Pass rank_lookback_bars=20."
        )
    frame = df.copy()
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    by_symbol = frame.groupby("symbol", sort=False)
    frame["bar_index"] = by_symbol.cumcount()
    frame["return_20"] = by_symbol["close"].transform(
        lambda series: series / series.shift(20) - 1.0
    )
    frame["return_60"] = by_symbol["close"].transform(
        lambda series: series / series.shift(60) - 1.0
    )
    frame["sma_20"] = by_symbol["close"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    frame["close_above_sma_20"] = (frame["close"] > frame["sma_20"]).astype(float)
    frame["trend_consistency_20"] = by_symbol["close_above_sma_20"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    frame["slope_20"] = by_symbol["sma_20"].transform(
        lambda series: series / series.shift(20) - 1.0
    )
    frame["score_mode"] = score_mode
    frame["trend_sma"] = by_symbol["close"].transform(
        lambda series: series.rolling(trend_sma_bars, min_periods=trend_sma_bars).mean()
    )
    frame["in_uptrend"] = frame["close"] > frame["trend_sma"]
    frame["recent_high"] = by_symbol["high"].transform(
        lambda series: series.shift(1).rolling(recent_high_lookback, min_periods=recent_high_lookback).max()
    )
    atr_parts: list[pd.Series] = []
    for _, group in frame.groupby("symbol", sort=False):
        atr_series = _atr_series(group[["high", "low", "close"]], atr_bars)
        atr_series.index = group.index
        atr_parts.append(atr_series)
    frame["atr"] = pd.concat(atr_parts).sort_index() if atr_parts else pd.Series(dtype=float)
    if score_mode == SCORE_MODE_RETURN_20:
        frame["score_return"] = frame["return_20"]
    elif score_mode == SCORE_MODE_RETURN_20_PLUS_60:
        frame["score_return"] = frame["return_20"] + frame["return_60"]
    elif score_mode == SCORE_MODE_TREND_CONSISTENCY_20:
        frame["score_return"] = frame["trend_consistency_20"]
    elif score_mode == SCORE_MODE_TREND_CONSISTENCY_20_X_SLOPE_20:
        frame["score_return"] = frame["trend_consistency_20"] * frame["slope_20"]
    else:
        frame["score_return"] = frame["trend_consistency_20"] * (frame["slope_20"] / frame["atr"])
    frame["pullback_distance_atr"] = (frame["recent_high"] - frame["close"]) / frame["atr"]
    frame["prev_high"] = by_symbol["high"].shift(1)
    frame["prev_pullback_distance_atr"] = by_symbol["pullback_distance_atr"].shift(1)
    frame["next_open"] = by_symbol["open"].shift(-1)
    frame["next_timestamp"] = by_symbol["timestamp"].shift(-1)
    timestamps_et = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["month"] = timestamps_et.dt.strftime("%Y-%m")
    frame["date"] = timestamps_et.dt.strftime("%Y-%m-%d")
    frame["signal_hour"] = timestamps_et.dt.hour
    frame["time_bucket"] = timestamps_et.apply(classify_time_bucket)
    for horizon in audit_horizons:
        frame[f"fwd_{horizon}b_return_pct"] = by_symbol["close"].transform(
            lambda series, horizon=horizon: (series.shift(-horizon) / series - 1.0) * 100.0
        )
    return frame


def classify_time_bucket(ts: pd.Timestamp) -> str:
    minutes = ts.hour * 60 + ts.minute
    if minutes < (10 * 60):
        return "open_30m"
    if minutes < (12 * 60):
        return "morning"
    if minutes < (14 * 60):
        return "midday"
    return "afternoon"


def assign_cross_sectional_ranks(
    panel_df: pd.DataFrame,
    *,
    eligible_percent: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = list(panel_df.columns) + [
        "cross_section_size",
        "cross_section_rank",
        "rank_pct",
        "bucket",
        "eligible_long",
    ]
    if panel_df.empty:
        return pd.DataFrame(columns=columns), {"timestamps_total": 0, "timestamps_ranked": 0, "timestamps_skipped": 0}

    rows: list[dict[str, Any]] = []
    timestamps_total = 0
    timestamps_ranked = 0
    timestamps_skipped = 0

    for timestamp, group in panel_df.groupby("timestamp", sort=True):
        timestamps_total += 1
        valid = group[group["score_return"].notna()].copy()
        if len(valid) < 3:
            timestamps_skipped += 1
            continue
        valid = valid.sort_values(["score_return", "symbol"], ascending=[False, True]).reset_index(drop=True)
        cross_section_size = len(valid)
        top_count = max(1, cross_section_size // 3)
        bottom_count = max(1, cross_section_size // 3)
        eligible_cutoff = max(1, int(math.ceil(cross_section_size * eligible_percent)))
        timestamps_ranked += 1

        for idx, (_, row) in enumerate(valid.iterrows()):
            rank = idx + 1
            if idx < top_count:
                bucket = "top"
            elif idx >= cross_section_size - bottom_count:
                bucket = "bottom"
            else:
                bucket = "middle"
            record = row.to_dict()
            record.update(
                {
                    "cross_section_size": cross_section_size,
                    "cross_section_rank": rank,
                    "rank_pct": rank / cross_section_size,
                    "bucket": bucket,
                    "eligible_long": rank <= eligible_cutoff,
                }
            )
            rows.append(record)

    diagnostics = {
        "timestamps_total": timestamps_total,
        "timestamps_ranked": timestamps_ranked,
        "timestamps_skipped": timestamps_skipped,
    }
    return pd.DataFrame(rows, columns=columns), diagnostics


def _return_summary(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "observation_count": 0,
            "mean_return_pct": 0.0,
            "median_return_pct": 0.0,
            "std_return_pct": 0.0,
            "hit_rate_pct": 0.0,
        }
    return {
        "observation_count": int(len(clean)),
        "mean_return_pct": float(clean.mean()),
        "median_return_pct": float(clean.median()),
        "std_return_pct": float(clean.std(ddof=0)),
        "hit_rate_pct": float((clean > 0).mean() * 100.0),
    }


def summarize_bucket_forward_returns(
    ranked_df: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket, bucket_df in ranked_df.groupby("bucket", sort=True):
        for horizon in horizons:
            stats = _return_summary(bucket_df[f"fwd_{horizon}b_return_pct"])
            rows.append({"bucket": bucket, "horizon_bars": horizon, **stats})
    return pd.DataFrame(rows).sort_values(["horizon_bars", "bucket"]) if rows else pd.DataFrame(
        columns=["bucket", "horizon_bars", "observation_count", "mean_return_pct", "median_return_pct", "std_return_pct", "hit_rate_pct"]
    )


def build_spread_observations(
    ranked_df: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timestamp, group in ranked_df.groupby("timestamp", sort=True):
        top = group[group["bucket"] == "top"]
        bottom = group[group["bucket"] == "bottom"]
        if top.empty or bottom.empty:
            continue
        for horizon in horizons:
            top_returns = pd.to_numeric(top[f"fwd_{horizon}b_return_pct"], errors="coerce").dropna()
            bottom_returns = pd.to_numeric(bottom[f"fwd_{horizon}b_return_pct"], errors="coerce").dropna()
            if top_returns.empty or bottom_returns.empty:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "month": str(group.iloc[0]["month"]),
                    "date": str(group.iloc[0]["date"]),
                    "time_bucket": str(group.iloc[0]["time_bucket"]),
                    "horizon_bars": horizon,
                    "spread_return_pct": float(top_returns.mean() - bottom_returns.mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_spread(spread_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in spread_df.groupby("horizon_bars", sort=True):
        stats = _return_summary(group["spread_return_pct"])
        rows.append({"bucket": "top_minus_bottom", "horizon_bars": int(horizon), **stats})
    return pd.DataFrame(rows).sort_values("horizon_bars") if rows else pd.DataFrame(
        columns=["bucket", "horizon_bars", "observation_count", "mean_return_pct", "median_return_pct", "std_return_pct", "hit_rate_pct"]
    )


def summarize_time_slices(
    ranked_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, group in ranked_df.groupby("month", sort=True):
        for bucket, bucket_df in group.groupby("bucket", sort=True):
            for horizon in horizons:
                stats = _return_summary(bucket_df[f"fwd_{horizon}b_return_pct"])
                rows.append({"period": period, "bucket": bucket, "horizon_bars": horizon, **stats})
    for (period, horizon), group in spread_df.groupby(["month", "horizon_bars"], sort=True):
        stats = _return_summary(group["spread_return_pct"])
        rows.append({"period": period, "bucket": "top_minus_bottom", "horizon_bars": int(horizon), **stats})
    return pd.DataFrame(rows).sort_values(["period", "horizon_bars", "bucket"]) if rows else pd.DataFrame(
        columns=["period", "bucket", "horizon_bars", "observation_count", "mean_return_pct", "median_return_pct", "std_return_pct", "hit_rate_pct"]
    )


def summarize_symbol_concentration(
    ranked_df: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    total_top = max(1, int((ranked_df["bucket"] == "top").sum()))
    rows: list[dict[str, Any]] = []
    for (symbol, bucket), group in ranked_df.groupby(["symbol", "bucket"], sort=True):
        record = {
            "symbol": symbol,
            "bucket": bucket,
            "membership_count": int(len(group)),
            "membership_share_pct": float(len(group) / total_top * 100.0) if bucket == "top" else 0.0,
        }
        for horizon in horizons:
            record[f"avg_{horizon}b_return_pct"] = float(
                pd.to_numeric(group[f"fwd_{horizon}b_return_pct"], errors="coerce").dropna().mean()
            )
        rows.append(record)
    columns = ["symbol", "bucket", "membership_count", "membership_share_pct"] + [
        f"avg_{horizon}b_return_pct" for horizon in horizons
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["bucket", "membership_count", "symbol"], ascending=[True, False, True]
    ) if rows else pd.DataFrame(columns=columns)


def evaluate_signal_success(
    bucket_summary_df: pd.DataFrame,
    spread_summary_df: pd.DataFrame,
    time_slice_df: pd.DataFrame,
    symbol_contribution_df: pd.DataFrame,
) -> dict[str, Any]:
    top = bucket_summary_df[bucket_summary_df["bucket"] == "top"]
    middle = bucket_summary_df[bucket_summary_df["bucket"] == "middle"]
    bottom = bucket_summary_df[bucket_summary_df["bucket"] == "bottom"]
    top_beats_middle = 0
    top_beats_bottom = 0
    for horizon in sorted(bucket_summary_df["horizon_bars"].unique()) if not bucket_summary_df.empty else []:
        top_row = top[top["horizon_bars"] == horizon]
        middle_row = middle[middle["horizon_bars"] == horizon]
        bottom_row = bottom[bottom["horizon_bars"] == horizon]
        if not top_row.empty and not middle_row.empty and float(top_row.iloc[0]["mean_return_pct"]) > float(middle_row.iloc[0]["mean_return_pct"]):
            top_beats_middle += 1
        if not top_row.empty and not bottom_row.empty and float(top_row.iloc[0]["mean_return_pct"]) > float(bottom_row.iloc[0]["mean_return_pct"]):
            top_beats_bottom += 1

    spread_positive = int((spread_summary_df["mean_return_pct"] > 0).sum()) if not spread_summary_df.empty else 0
    spread_total = int(len(spread_summary_df))
    time_spread = time_slice_df[time_slice_df["bucket"] == "top_minus_bottom"]
    positive_time_slices = int((time_spread["mean_return_pct"] > 0).sum()) if not time_spread.empty else 0
    top_contrib = symbol_contribution_df[symbol_contribution_df["bucket"] == "top"].sort_values(
        ["membership_count", "symbol"], ascending=[False, True]
    )
    concentration = float(top_contrib.iloc[0]["membership_share_pct"]) if not top_contrib.empty else 0.0
    success = (
        top_beats_bottom >= max(1, spread_total - 1)
        and spread_positive >= max(1, spread_total - 1)
    )
    return {
        "top_beats_middle_horizon_count": top_beats_middle,
        "top_beats_bottom_horizon_count": top_beats_bottom,
        "positive_spread_horizon_count": spread_positive,
        "spread_horizon_count": spread_total,
        "positive_spread_time_slice_count": positive_time_slices,
        "top_bucket_max_membership_share_pct": concentration,
        "signal_audit_pass": bool(success),
    }


def build_walk_forward_windows(
    df: pd.DataFrame,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[FoldWindow]:
    if "date" in df.columns:
        trading_days = sorted(df["date"].dropna().astype(str).unique().tolist())
    elif "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        trading_days = sorted(timestamps.dt.strftime("%Y-%m-%d").dropna().unique().tolist())
    else:
        raise KeyError("Expected either a 'date' or 'timestamp' column to build walk-forward windows.")
    windows = _generate_walk_forward_windows(trading_days, train_days=train_days, test_days=test_days, step_days=step_days)
    return [
        FoldWindow(
            window_idx=int(window["window_idx"]),
            train_start=str(window["train_start"]),
            train_end=str(window["train_end"]),
            test_start=str(window["test_start"]),
            test_end=str(window["test_end"]),
        )
        for window in windows
    ]


def summarize_fold_spreads(
    ranked_df: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    windows: list[FoldWindow],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in windows:
        mask = (ranked_df["date"] >= window.test_start) & (ranked_df["date"] <= window.test_end)
        fold_df = ranked_df[mask]
        spread_df = build_spread_observations(fold_df, horizons=horizons)
        spread_summary = summarize_spread(spread_df)
        for _, row in spread_summary.iterrows():
            rows.append(
                {
                    "window_idx": window.window_idx,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    **row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def generate_strategy_variants(
    *,
    score_mode: str,
    rank_lookbacks: tuple[int, ...],
    eligible_percents: tuple[float, ...],
    hold_bars_list: tuple[int, ...],
    pullback_depths: tuple[float, ...],
    use_early_no_follow_through_exit: bool = False,
    use_recent_follow_through_filter: bool = False,
) -> list[StrategyVariant]:
    score_mode = normalize_score_mode(score_mode)
    variants: list[StrategyVariant] = []
    for lookback in rank_lookbacks:
        for eligible_percent in eligible_percents:
            for hold_bars in hold_bars_list:
                variants.append(
                    StrategyVariant(
                        family="ranking_only",
                        score_mode=score_mode,
                        rank_lookback_bars=int(lookback),
                        eligible_percent=float(eligible_percent),
                        hold_bars=int(hold_bars),
                        use_early_no_follow_through_exit=False,
                        use_recent_follow_through_filter=False,
                    )
                )
                variants.append(
                    StrategyVariant(
                        family="baseline",
                        score_mode=score_mode,
                        rank_lookback_bars=int(lookback),
                        eligible_percent=float(eligible_percent),
                        hold_bars=int(hold_bars),
                        use_early_no_follow_through_exit=False,
                        use_recent_follow_through_filter=False,
                    )
                )
                for depth in pullback_depths:
                    variants.append(
                        StrategyVariant(
                            family="pullback",
                            score_mode=score_mode,
                            rank_lookback_bars=int(lookback),
                            eligible_percent=float(eligible_percent),
                            hold_bars=int(hold_bars),
                            pullback_depth_atr=float(depth),
                            use_early_no_follow_through_exit=bool(use_early_no_follow_through_exit),
                            use_recent_follow_through_filter=bool(use_recent_follow_through_filter),
                        )
                    )
    return variants


def _apply_recent_follow_through_filter(
    ranked_df: pd.DataFrame,
    *,
    horizon_bars: int,
    rolling_window_bars: int = 20,
) -> pd.DataFrame:
    frame = ranked_df.copy()
    target_col = f"fwd_{int(horizon_bars)}b_return_pct"
    filter_col = f"recent_top_bucket_return_{int(horizon_bars)}b"
    if target_col not in frame.columns:
        frame[filter_col] = pd.NA
        return frame

    top_by_timestamp = (
        frame[frame["bucket"] == "top"]
        .groupby("timestamp", sort=True)[target_col]
        .mean()
        .sort_index()
    )
    realized_recent = top_by_timestamp.shift(int(horizon_bars)).rolling(
        rolling_window_bars,
        min_periods=rolling_window_bars,
    ).mean()
    frame = frame.merge(
        realized_recent.rename(filter_col),
        left_on="timestamp",
        right_index=True,
        how="left",
    )
    return frame


def generate_entry_candidates(
    ranked_df: pd.DataFrame,
    *,
    variant: StrategyVariant,
) -> pd.DataFrame:
    frame = ranked_df.copy()
    if variant.use_recent_follow_through_filter:
        frame = _apply_recent_follow_through_filter(
            frame,
            horizon_bars=variant.hold_bars,
        )
        recent_follow_through_ok = (
            frame[f"recent_top_bucket_return_{int(variant.hold_bars)}b"].fillna(float("-inf")) > 0.0
        )
    else:
        recent_follow_through_ok = pd.Series(True, index=frame.index)
    frame["ranking_only_entry_signal"] = frame["eligible_long"] & frame["next_open"].notna()
    frame["baseline_entry_signal"] = frame["eligible_long"] & frame["in_uptrend"] & frame["next_open"].notna()
    frame["pullback_ready_prev"] = (
        frame["eligible_long"]
        & frame["in_uptrend"]
        & frame["prev_pullback_distance_atr"].notna()
        & (frame["prev_pullback_distance_atr"] >= float(variant.pullback_depth_atr or 0.0))
    )
    frame["pullback_entry_signal"] = (
        frame["eligible_long"]
        & frame["in_uptrend"]
        & frame["next_open"].notna()
        & frame["pullback_ready_prev"]
        & frame["prev_high"].notna()
        & (frame["close"] > frame["prev_high"])
        & recent_follow_through_ok
    )
    if variant.family == "ranking_only":
        signal_column = "ranking_only_entry_signal"
    elif variant.family == "baseline":
        signal_column = "baseline_entry_signal"
    elif variant.family == "pullback":
        signal_column = "pullback_entry_signal"
    else:
        raise ValueError(f"Unsupported strategy family {variant.family!r}.")
    return frame[frame[signal_column]].copy()


def _compute_trade_metrics(trades_df: pd.DataFrame) -> dict[str, float]:
    if trades_df.empty:
        return {
            "trade_count": 0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_hold_bars": 0.0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
        }
    winning = trades_df[trades_df["realized_pnl"] > 0]["realized_pnl"]
    losing = trades_df[trades_df["realized_pnl"] < 0]["realized_pnl"]
    gross_profit = float(winning.sum()) if not winning.empty else 0.0
    gross_loss = float(-losing.sum()) if not losing.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    return {
        "trade_count": int(len(trades_df)),
        "expectancy": float(trades_df["realized_pnl"].mean()),
        "win_rate": float((trades_df["realized_pnl"] > 0).mean() * 100.0),
        "profit_factor": float(profit_factor if math.isfinite(profit_factor) else 999.0),
        "avg_hold_bars": float(trades_df["hold_bars"].mean()),
        "total_pnl": float(trades_df["realized_pnl"].sum()),
        "total_return_pct": float(trades_df["return_pct"].sum()),
    }


def run_strategy_backtest(
    ranked_df: pd.DataFrame,
    *,
    variant: StrategyVariant,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    position_size: float = DEFAULT_POSITION_SIZE,
    commission_per_order: float = DEFAULT_COMMISSION_PER_ORDER,
    slippage_per_share: float = DEFAULT_SLIPPAGE_PER_SHARE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if ranked_df.empty:
        empty_trades = pd.DataFrame(columns=["symbol", "entry_ts", "exit_ts", "hold_bars", "realized_pnl", "return_pct", "variant_id"])
        empty_equity = pd.DataFrame(columns=["timestamp", "equity"])
        empty_signals = pd.DataFrame(columns=ranked_df.columns)
        return empty_trades, empty_equity, empty_signals

    rows_by_timestamp = {
        timestamp: group.set_index("symbol", drop=False).copy()
        for timestamp, group in ranked_df.groupby("timestamp", sort=True)
    }
    candidates_df = generate_entry_candidates(ranked_df, variant=variant)
    candidates_by_timestamp = {
        timestamp: group.sort_values(["score_return", "symbol"], ascending=[False, True]).copy()
        for timestamp, group in candidates_df.groupby("timestamp", sort=True)
    }
    symbol_frames = {
        symbol: group.reset_index(drop=True).copy()
        for symbol, group in ranked_df.groupby("symbol", sort=False)
    }
    open_positions: dict[str, dict[str, Any]] = {}
    pending_entries: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    realized_pnl = 0.0
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []

    for timestamp in sorted(rows_by_timestamp):
        current_rows = rows_by_timestamp[timestamp]

        exiting_symbols = [
            symbol
            for symbol, position in open_positions.items()
            if position["exit_ts"] == timestamp
        ]
        for symbol in exiting_symbols:
            row = current_rows.loc[symbol]
            position = open_positions.pop(symbol)
            exit_fill = max(float(row["open"]) - slippage_per_share, 0.0)
            realized = (exit_fill - position["entry_fill"]) * position["shares"] - commission_per_order
            realized_pnl += realized
            trades.append(
                {
                    "symbol": symbol,
                    "entry_ts": position["entry_ts"],
                    "exit_ts": timestamp,
                    "entry_fill": position["entry_fill"],
                    "exit_fill": exit_fill,
                    "score_return": position["score_return"],
                    "cross_section_rank": position["cross_section_rank"],
                    "cross_section_size": position["cross_section_size"],
                    "hold_bars": position["hold_bars"],
                    "score_mode": variant.score_mode,
                    "use_early_no_follow_through_exit": variant.use_early_no_follow_through_exit,
                    "use_recent_follow_through_filter": variant.use_recent_follow_through_filter,
                    "realized_pnl": realized,
                    "return_pct": (realized / position_size) * 100.0,
                    "variant_id": variant.variant_id,
                    "family": variant.family,
                }
            )

        if variant.use_early_no_follow_through_exit:
            for symbol, position in list(open_positions.items()):
                if symbol not in current_rows.index:
                    continue
                if pd.isna(current_rows.loc[symbol]["next_timestamp"]):
                    continue
                bars_since_entry = int(current_rows.loc[symbol]["bar_index"]) - int(position["entry_bar_index"])
                if bars_since_entry < 3:
                    continue
                current_close = float(current_rows.loc[symbol]["close"])
                current_return = (current_close / position["entry_fill"]) - 1.0 if position["entry_fill"] > 0 else 0.0
                if current_return <= 0.0:
                    candidate_exit_ts = pd.Timestamp(current_rows.loc[symbol]["next_timestamp"])
                    if candidate_exit_ts < position["exit_ts"]:
                        position["exit_ts"] = candidate_exit_ts
                        position["hold_bars"] = bars_since_entry + 1

        pending_here = pending_entries.pop(timestamp, [])
        if pending_here:
            pending_here = sorted(pending_here, key=lambda item: (-item["score_return"], item["symbol"]))
            available_slots = max(0, max_positions - len(open_positions))
            for candidate in pending_here[:available_slots]:
                symbol = candidate["symbol"]
                if symbol in open_positions or symbol not in current_rows.index:
                    continue
                row = current_rows.loc[symbol]
                symbol_frame = symbol_frames[symbol]
                entry_bar_index = int(row["bar_index"])
                exit_bar_index = entry_bar_index + variant.hold_bars
                if exit_bar_index >= len(symbol_frame):
                    continue
                exit_ts = pd.Timestamp(symbol_frame.iloc[exit_bar_index]["timestamp"])
                entry_fill = float(row["open"]) + slippage_per_share
                shares = position_size / entry_fill if entry_fill > 0 else 0.0
                realized_pnl -= commission_per_order
                open_positions[symbol] = {
                    "entry_ts": timestamp,
                    "entry_fill": entry_fill,
                    "shares": shares,
                    "entry_bar_index": entry_bar_index,
                    "exit_ts": exit_ts,
                    "hold_bars": variant.hold_bars,
                    "score_return": float(candidate["score_return"]),
                    "cross_section_rank": int(candidate["cross_section_rank"]),
                    "cross_section_size": int(candidate["cross_section_size"]),
                }

        available_slots = max(0, max_positions - len(open_positions))
        if available_slots > 0 and timestamp in candidates_by_timestamp:
            group = candidates_by_timestamp[timestamp]
            chosen = []
            for _, row in group.iterrows():
                symbol = str(row["symbol"])
                if symbol in open_positions:
                    continue
                if any(item["symbol"] == symbol for item in chosen):
                    continue
                chosen.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "score_return": float(row["score_return"]),
                        "cross_section_rank": int(row["cross_section_rank"]),
                        "cross_section_size": int(row["cross_section_size"]),
                    }
                )
                signal_rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "score_return": float(row["score_return"]),
                        "cross_section_rank": int(row["cross_section_rank"]),
                        "cross_section_size": int(row["cross_section_size"]),
                        "score_mode": variant.score_mode,
                        "variant_id": variant.variant_id,
                        "family": variant.family,
                    }
                )
                if len(chosen) >= available_slots:
                    break
            if chosen:
                next_ts = pd.Timestamp(group.iloc[0]["next_timestamp"])
                if pd.notna(next_ts):
                    pending_entries.setdefault(next_ts, []).extend(chosen)

        unrealized = 0.0
        for symbol, position in open_positions.items():
            if symbol not in current_rows.index:
                continue
            mark_price = float(current_rows.loc[symbol]["close"])
            unrealized += (mark_price - position["entry_fill"]) * position["shares"]
        equity_rows.append({"timestamp": timestamp, "equity": realized_pnl + unrealized})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    signals_df = pd.DataFrame(signal_rows)
    if not equity_df.empty:
        running_max = equity_df["equity"].cummax()
        drawdown = equity_df["equity"] - running_max
        equity_df["drawdown"] = drawdown
        equity_df["drawdown_pct"] = drawdown / running_max.replace(0.0, pd.NA) * 100.0
    return trades_df, equity_df, signals_df


def summarize_strategy_run(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    *,
    variant: StrategyVariant,
) -> dict[str, Any]:
    metrics = _compute_trade_metrics(trades_df)
    max_drawdown_pct = 0.0
    if not equity_df.empty and "drawdown_pct" in equity_df.columns:
        max_drawdown_pct = float(equity_df["drawdown_pct"].min() or 0.0)
    elif not trades_df.empty and "realized_pnl" in trades_df.columns:
        path = trades_df.copy()
        if "exit_ts" in path.columns:
            path = path.sort_values("exit_ts").reset_index(drop=True)
        path["equity"] = path["realized_pnl"].cumsum()
        running_max = path["equity"].cummax()
        drawdown = path["equity"] - running_max
        capital_base = float(DEFAULT_POSITION_SIZE * max(1, DEFAULT_MAX_POSITIONS))
        max_drawdown_pct = float((drawdown.min() / capital_base) * 100.0 if capital_base > 0 else 0.0)
    return {
        "variant_id": variant.variant_id,
        "family": variant.family,
        "score_mode": variant.score_mode,
        "rank_lookback_bars": variant.rank_lookback_bars,
        "eligible_percent": variant.eligible_percent,
        "pullback_depth_atr": variant.pullback_depth_atr,
        "hold_bars": variant.hold_bars,
        "use_early_no_follow_through_exit": variant.use_early_no_follow_through_exit,
        "use_recent_follow_through_filter": variant.use_recent_follow_through_filter,
        **metrics,
        "max_drawdown_pct": abs(max_drawdown_pct),
    }


def summarize_trades_by_symbol(trades_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, group in trades_df.groupby("symbol", sort=True):
        stats = _compute_trade_metrics(group)
        rows.append(
            {
                "symbol": symbol,
                **stats,
            }
        )
    return pd.DataFrame(rows).sort_values(["total_pnl", "symbol"], ascending=[False, True]) if rows else pd.DataFrame(
        columns=["symbol", "trade_count", "expectancy", "win_rate", "profit_factor", "avg_hold_bars", "total_pnl", "total_return_pct"]
    )


def variant_selection_score(summary: dict[str, Any]) -> tuple[float, float, float]:
    trade_count = float(summary.get("trade_count", 0))
    expectancy = float(summary.get("expectancy", 0.0))
    profit_factor = float(summary.get("profit_factor", 0.0))
    penalized_expectancy = expectancy if trade_count >= 3 else -9999.0
    return (penalized_expectancy, profit_factor, trade_count)


def select_best_variant(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not summaries:
        return None
    ranked = sorted(summaries, key=variant_selection_score, reverse=True)
    return ranked[0]


def run_walk_forward_validation(
    full_df: pd.DataFrame,
    *,
    variants: list[StrategyVariant],
    train_days: int = DEFAULT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    position_size: float = DEFAULT_POSITION_SIZE,
    commission_per_order: float = DEFAULT_COMMISSION_PER_ORDER,
    slippage_per_share: float = DEFAULT_SLIPPAGE_PER_SHARE,
) -> dict[str, pd.DataFrame]:
    if "date" not in full_df.columns:
        if "timestamp" not in full_df.columns:
            raise KeyError("Expected either a 'date' or 'timestamp' column in the validation input frame.")
        working_df = full_df.copy()
        timestamps = pd.to_datetime(working_df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        working_df["date"] = timestamps.dt.strftime("%Y-%m-%d")
    else:
        working_df = full_df.copy()

    windows = build_walk_forward_windows(working_df, train_days=train_days, test_days=test_days, step_days=step_days)
    variant_rows: list[dict[str, Any]] = []
    fold_winner_rows: list[dict[str, Any]] = []
    stitched_trade_rows: list[pd.DataFrame] = []

    for window in windows:
        train_df = working_df[(working_df["date"] >= window.train_start) & (working_df["date"] <= window.train_end)].copy()
        test_df = working_df[(working_df["date"] >= window.test_start) & (working_df["date"] <= window.test_end)].copy()

        family_order = list(dict.fromkeys(variant.family for variant in variants))
        for family in family_order:
            family_variants = [variant for variant in variants if variant.family == family]
            train_summaries: list[dict[str, Any]] = []
            for variant in family_variants:
                train_prepared = prepare_panel_features(
                    train_df,
                    rank_lookback_bars=variant.rank_lookback_bars,
                    audit_horizons=(variant.hold_bars,),
                    score_mode=variant.score_mode,
                )
                train_ranked, _ = assign_cross_sectional_ranks(
                    train_prepared,
                    eligible_percent=variant.eligible_percent,
                )
                train_trades, train_equity, _ = run_strategy_backtest(
                    train_ranked,
                    variant=variant,
                    max_positions=max_positions,
                    position_size=position_size,
                    commission_per_order=commission_per_order,
                    slippage_per_share=slippage_per_share,
                )
                train_summary = summarize_strategy_run(train_trades, train_equity, variant=variant)
                train_summary.update(
                    {
                        "window_idx": window.window_idx,
                        "split": "train",
                        "train_start": window.train_start,
                        "train_end": window.train_end,
                        "test_start": window.test_start,
                        "test_end": window.test_end,
                    }
                )
                train_summaries.append(train_summary)
                variant_rows.append(train_summary)

                test_prepared = prepare_panel_features(
                    test_df,
                    rank_lookback_bars=variant.rank_lookback_bars,
                    audit_horizons=(variant.hold_bars,),
                    score_mode=variant.score_mode,
                )
                test_ranked, _ = assign_cross_sectional_ranks(
                    test_prepared,
                    eligible_percent=variant.eligible_percent,
                )
                test_trades, test_equity, _ = run_strategy_backtest(
                    test_ranked,
                    variant=variant,
                    max_positions=max_positions,
                    position_size=position_size,
                    commission_per_order=commission_per_order,
                    slippage_per_share=slippage_per_share,
                )
                test_summary = summarize_strategy_run(test_trades, test_equity, variant=variant)
                test_summary.update(
                    {
                        "window_idx": window.window_idx,
                        "split": "test",
                        "train_start": window.train_start,
                        "train_end": window.train_end,
                        "test_start": window.test_start,
                        "test_end": window.test_end,
                    }
                )
                variant_rows.append(test_summary)

            winner = select_best_variant(train_summaries)
            if winner is None:
                continue
            fold_winner_rows.append(
                {
                    "window_idx": window.window_idx,
                    "family": family,
                    "score_mode": winner["score_mode"],
                    "selected_variant_id": winner["variant_id"],
                    "selection_expectancy": winner["expectancy"],
                    "selection_trade_count": winner["trade_count"],
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                }
            )

            selected_variant = next(variant for variant in family_variants if variant.variant_id == winner["variant_id"])
            selected_test_prepared = prepare_panel_features(
                test_df,
                rank_lookback_bars=selected_variant.rank_lookback_bars,
                audit_horizons=(selected_variant.hold_bars,),
                score_mode=selected_variant.score_mode,
            )
            selected_test_ranked, _ = assign_cross_sectional_ranks(
                selected_test_prepared,
                eligible_percent=selected_variant.eligible_percent,
            )
            selected_test_trades, _, _ = run_strategy_backtest(
                selected_test_ranked,
                variant=selected_variant,
                max_positions=max_positions,
                position_size=position_size,
                commission_per_order=commission_per_order,
                slippage_per_share=slippage_per_share,
            )
            if not selected_test_trades.empty:
                selected_test_trades = selected_test_trades.copy()
                selected_test_trades["window_idx"] = window.window_idx
                selected_test_trades["test_start"] = window.test_start
                selected_test_trades["test_end"] = window.test_end
                stitched_trade_rows.append(selected_test_trades)

    variant_results_df = pd.DataFrame(variant_rows)
    fold_winners_df = pd.DataFrame(fold_winner_rows)
    stitched_selected_trades_df = (
        pd.concat(stitched_trade_rows, ignore_index=True)
        if stitched_trade_rows
        else pd.DataFrame(columns=["symbol", "entry_ts", "exit_ts", "hold_bars", "realized_pnl", "return_pct", "variant_id", "family", "window_idx", "test_start", "test_end"])
    )

    stitched_summary_rows: list[dict[str, Any]] = []
    for family, group in stitched_selected_trades_df.groupby("family", sort=True):
        family_variant = StrategyVariant(
            family=family,
            score_mode=str(group["score_mode"].iloc[0]) if "score_mode" in group.columns and not group.empty else SCORE_MODE_RETURN_20,
            rank_lookback_bars=0,
            eligible_percent=0.0,
            hold_bars=0,
        )
        summary = summarize_strategy_run(group, pd.DataFrame(), variant=family_variant)
        summary["variant_id"] = f"selected_{family}_stitched_oos"
        summary["rank_lookback_bars"] = None
        summary["eligible_percent"] = None
        summary["pullback_depth_atr"] = None
        summary["hold_bars"] = int(group["hold_bars"].mean()) if "hold_bars" in group.columns and not group.empty else 0
        top_symbol_share = 0.0
        if not group.empty:
            per_symbol = group.groupby("symbol")["realized_pnl"].sum().sort_values(ascending=False)
            positive = per_symbol[per_symbol > 0]
            if not positive.empty:
                top_symbol_share = float(positive.iloc[0] / positive.sum())
        summary["top_positive_symbol_share"] = top_symbol_share
        stitched_summary_rows.append(summary)
    stitched_selected_summary_df = pd.DataFrame(stitched_summary_rows)

    stitched_variant_rows: list[dict[str, Any]] = []
    if not variant_results_df.empty:
        test_only = variant_results_df[variant_results_df["split"] == "test"]
        for variant_id, group in test_only.groupby("variant_id", sort=True):
            record = group.iloc[0][["variant_id", "family", "score_mode", "rank_lookback_bars", "eligible_percent", "pullback_depth_atr", "hold_bars"]].to_dict()
            record.update(
                {
                    "fold_count": int(group["window_idx"].nunique()),
                    "avg_expectancy": float(group["expectancy"].mean()),
                    "avg_profit_factor": float(group["profit_factor"].mean()),
                    "avg_trade_count": float(group["trade_count"].mean()),
                    "total_pnl": float(group["total_pnl"].sum()),
                    "positive_fold_ratio": float((group["expectancy"] > 0).mean()),
                }
            )
            stitched_variant_rows.append(record)
    stitched_variant_summary_df = pd.DataFrame(stitched_variant_rows).sort_values(
        ["family", "avg_expectancy", "total_pnl", "variant_id"], ascending=[True, False, False, True]
    ) if stitched_variant_rows else pd.DataFrame()

    return {
        "windows": pd.DataFrame([asdict(window) for window in windows]),
        "variant_results": variant_results_df,
        "fold_winners": fold_winners_df,
        "stitched_selected_trades": stitched_selected_trades_df,
        "stitched_selected_summary": stitched_selected_summary_df,
        "stitched_variant_summary": stitched_variant_summary_df,
    }


def evaluate_strategy_success(
    stitched_selected_summary_df: pd.DataFrame,
    stitched_selected_trades_df: pd.DataFrame,
) -> dict[str, Any]:
    if stitched_selected_summary_df.empty:
        return {
            "pullback_improves_quality_metric": False,
            "pullback_trade_count_ok": False,
            "robustness_ok": False,
            "promotion_ready": False,
        }
    summary_lookup = {
        row["family"]: row
        for row in stitched_selected_summary_df.to_dict("records")
    }
    baseline = summary_lookup.get("baseline", {})
    pullback = summary_lookup.get("pullback", {})
    baseline_trade_count = float(baseline.get("trade_count", 0))
    pullback_trade_count = float(pullback.get("trade_count", 0))
    quality_improved = (
        float(pullback.get("profit_factor", 0.0)) > float(baseline.get("profit_factor", 0.0))
        or float(pullback.get("expectancy", 0.0)) > float(baseline.get("expectancy", 0.0))
        or float(pullback.get("max_drawdown_pct", 999.0)) < float(baseline.get("max_drawdown_pct", 999.0))
    )
    trade_count_ok = pullback_trade_count >= max(3.0, baseline_trade_count * 0.5)
    family_top_share: dict[str, float] = {}
    if not stitched_selected_trades_df.empty:
        for family, group in stitched_selected_trades_df.groupby("family", sort=True):
            per_symbol = group.groupby("symbol")["realized_pnl"].sum().sort_values(ascending=False)
            positive = per_symbol[per_symbol > 0]
            family_top_share[family] = float(positive.iloc[0] / positive.sum()) if not positive.empty else 0.0
    robustness_ok = all(share <= 0.70 for share in family_top_share.values()) if family_top_share else False
    return {
        "pullback_improves_quality_metric": bool(quality_improved),
        "pullback_trade_count_ok": bool(trade_count_ok),
        "robustness_ok": bool(robustness_ok),
        "promotion_ready": bool(quality_improved and trade_count_ok and robustness_ok),
        "family_top_positive_share": family_top_share,
    }


def print_table(title: str, df: pd.DataFrame, *, max_rows: int | None = None) -> None:
    print(f"\n=== {title} ===")
    print(_frame_text(df, max_rows=max_rows))
