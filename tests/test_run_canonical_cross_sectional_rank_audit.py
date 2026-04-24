import sys

from run_canonical_cross_sectional_rank_audit import _parse_horizons
from research.run_canonical_cross_sectional_rank_audit import parse_args


def test_parse_horizons_deduplicates_and_sorts() -> None:
    assert _parse_horizons("10,5,10") == (5, 10)


def test_parse_args_accepts_score_mode(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--score-mode", "trend_consistency_20_x_slope_20_over_atr_20"])
    args = parse_args()
    assert args.score_mode == "trend_consistency_20_x_slope_20_over_atr_20"
