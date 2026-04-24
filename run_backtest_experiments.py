"""
run_backtest_experiments.py
---------------------------
Run a small, repeatable set of backtest experiments and summarize the results.

The first preset automates the breakout tuning loop we were running manually:
baseline, larger target, ORB filter, and both changes combined.

Outputs are written to a dedicated subdirectory under results/experiments/ so
research runs do not clutter the top-level results/ folder.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from research.experiment_log import log_experiment_run


PROJECT_ROOT = Path(__file__).resolve().parent
BACKTEST_RUNNER = PROJECT_ROOT / "backtest_runner.py"
EXPERIMENTS_ROOT = PROJECT_ROOT / "results" / "experiments"

SUMMARY_METRICS = [
    "total_return_pct",
    "sharpe_ratio",
    "max_drawdown_pct",
    "total_trades",
    "win_rate",
    "profit_factor",
    "expectancy",
    "realized_pnl",
    "timing_total_seconds",
]


@dataclass(frozen=True)
class ExperimentJob:
    name: str
    output_stem: str
    extra_args: tuple[str, ...]


PRESET_JOBS: dict[str, tuple[ExperimentJob, ...]] = {
    "breakout_baseline_compare": (
        ExperimentJob(
            name="baseline",
            output_stem="baseline",
            extra_args=("--strategy-mode", "breakout"),
        ),
        ExperimentJob(
            name="exit_1.5x",
            output_stem="exit_1_5x",
            extra_args=(
                "--strategy-mode",
                "breakout",
                "--breakout-exit-style",
                "target_1_5x_stop_low",
            ),
        ),
        ExperimentJob(
            name="orb_filter",
            output_stem="orb_filter",
            extra_args=(
                "--strategy-mode",
                "breakout",
                "--orb-filter-mode",
                "volume_or_volatility",
            ),
        ),
        ExperimentJob(
            name="exit_1.5x_plus_filter",
            output_stem="exit_1_5x_plus_filter",
            extra_args=(
                "--strategy-mode",
                "breakout",
                "--breakout-exit-style",
                "target_1_5x_stop_low",
                "--orb-filter-mode",
                "volume_or_volatility",
            ),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a preset batch of backtest experiments and summarize the results."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory to pass to backtest_runner.py.",
    )
    parser.add_argument(
        "--preset",
        default="breakout_baseline_compare",
        choices=sorted(PRESET_JOBS.keys()),
        help="Named experiment set to run.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional folder name under results/experiments/. Defaults to preset + timestamp.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for subprocess runs. Default: current interpreter.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an experiment when its main CSV already exists in the output folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_name(value: str) -> str:
    allowed = []
    for ch in value:
        allowed.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(allowed).strip("_") or "experiment_run"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_dir(run_name: str | None, preset: str) -> Path:
    if run_name:
        folder_name = _sanitize_name(run_name)
    else:
        folder_name = f"{preset}_{_timestamp()}"
    return _ensure_dir(EXPERIMENTS_ROOT / folder_name)


def build_command(
    python_executable: str,
    dataset: str,
    output_csv: Path,
    job: ExperimentJob,
) -> list[str]:
    return [
        python_executable,
        str(BACKTEST_RUNNER),
        "--dataset",
        dataset,
        *job.extra_args,
        "--output-csv",
        str(output_csv),
    ]


def run_command(command: list[str], *, dry_run: bool) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n>>> {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def load_summary_row(csv_path: Path, run_name: str) -> dict:
    row = pd.read_csv(csv_path).iloc[0].to_dict()
    row["run"] = run_name
    row["source_csv"] = str(csv_path)
    return row


def build_summary_dataframe(summary_rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return df
    cols = ["run", *SUMMARY_METRICS, "source_csv"]
    present = [col for col in cols if col in df.columns]
    return df[present].sort_values(
        by=["total_return_pct", "sharpe_ratio", "profit_factor", "max_drawdown_pct"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_per_symbol_comparison(rows: list[dict]) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    for row in rows:
        per_symbol_path = Path(str(row["source_csv"])).with_name(
            f"{Path(str(row['source_csv'])).stem}_per_symbol.csv"
        )
        if not per_symbol_path.exists():
            continue
        df = pd.read_csv(per_symbol_path)
        if "symbol" not in df.columns or "total_return_pct" not in df.columns:
            continue
        df = df[df["symbol"] != "COMBINED"].copy()
        series_list.append(df.set_index("symbol")["total_return_pct"].rename(row["run"]))

    if not series_list:
        return pd.DataFrame()

    comparison = pd.concat(series_list, axis=1)
    run_columns = list(comparison.columns)
    comparison["best_run"] = comparison[run_columns].idxmax(axis=1)
    comparison["worst_run"] = comparison[run_columns].idxmin(axis=1)
    comparison["spread"] = comparison[run_columns].max(axis=1) - comparison[run_columns].min(axis=1)
    return comparison.sort_values(by="spread", ascending=False).reset_index()


def write_markdown_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    per_symbol_df: pd.DataFrame,
) -> None:
    lines: list[str] = ["# Experiment Summary", ""]
    if summary_df.empty:
        lines.append("No summary rows were produced.")
    else:
        best = summary_df.iloc[0]
        lines.extend(
            [
                f"Best run: `{best['run']}`",
                "",
                "## Portfolio Comparison",
                "",
                "```text",
                summary_df.to_string(index=False),
                "```",
            ]
        )
    if not per_symbol_df.empty:
        lines.extend(
            [
                "",
                "## Symbols With Largest Spread",
                "",
                "```text",
                per_symbol_df.head(15).to_string(index=False),
                "```",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    jobs = PRESET_JOBS[args.preset]
    output_dir = build_output_dir(args.run_name, args.preset)

    print(f"Preset:      {args.preset}")
    print(f"Dataset:     {args.dataset}")
    print(f"Output dir:  {output_dir}")

    summary_rows: list[dict] = []
    for job in jobs:
        output_csv = output_dir / f"{job.output_stem}.csv"
        if args.skip_existing and output_csv.exists():
            print(f"\nSkipping existing run: {job.name} -> {output_csv}")
        else:
            command = build_command(args.python, args.dataset, output_csv, job)
            run_command(command, dry_run=args.dry_run)

        if not args.dry_run:
            summary_rows.append(load_summary_row(output_csv, job.name))

    if args.dry_run:
        return

    summary_df = build_summary_dataframe(summary_rows)
    per_symbol_df = build_per_symbol_comparison(summary_rows)

    summary_csv = output_dir / "comparison_summary.csv"
    per_symbol_csv = output_dir / "comparison_per_symbol.csv"
    report_md = output_dir / "comparison_report.md"

    summary_df.to_csv(summary_csv, index=False)
    if not per_symbol_df.empty:
        per_symbol_df.to_csv(per_symbol_csv, index=False)
    write_markdown_report(report_md, summary_df, per_symbol_df)

    print("\nSaved:")
    print(f"  - {summary_csv}")
    if not per_symbol_df.empty:
        print(f"  - {per_symbol_csv}")
    print(f"  - {report_md}")

    if not summary_df.empty:
        print("\nTop runs:")
        print(summary_df.to_string(index=False))
        best_row = summary_df.iloc[0].to_dict()
        log_experiment_run(
            run_type="backtest_experiment_batch",
            script_path=__file__,
            entrypoint=sys.argv[0],
            strategy_name=str(best_row.get("run") or args.preset),
            dataset_path=args.dataset,
            params={
                "preset": args.preset,
                "run_name": args.run_name,
                "skip_existing": args.skip_existing,
                "jobs": [
                    {
                        "name": job.name,
                        "output_stem": job.output_stem,
                        "extra_args": list(job.extra_args),
                    }
                    for job in jobs
                ],
            },
            metrics={
                "total_return_pct": best_row.get("total_return_pct"),
                "profit_factor": best_row.get("profit_factor"),
                "sharpe": best_row.get("sharpe_ratio"),
                "win_rate": best_row.get("win_rate"),
                "max_drawdown_pct": best_row.get("max_drawdown_pct"),
                "trade_count": best_row.get("total_trades"),
                "expectancy": best_row.get("expectancy"),
                "realized_pnl": best_row.get("realized_pnl"),
            },
            output_path=output_dir,
            summary_path=summary_csv,
            extra_fields={
                "best_run_name": best_row.get("run"),
                "report_path": str(report_md),
                "comparison_per_symbol_path": str(per_symbol_csv) if per_symbol_df.empty is False else None,
            },
        )


if __name__ == "__main__":
    main()
