# Legacy Research Audit

## Summary

This audit covers older sweep-era experiment runners that still live at the repo root:

- `breakout_eval.py`
- `breakout_sweep.py`
- `compare_mr_strategies.py`
- `vwap_sweep.py`
- `vwap_atr_sweep.py`

These files appear to be historical, self-contained research scripts preserved for reproducibility rather than part of the active TradeOS runtime or the newer grouped research area.

## Findings

### 1. `breakout_eval.py`

- File/path: `breakout_eval.py`
- Purpose: historical breakout strategy evaluation against earlier mean-reversion baselines using a fixed dataset and fixed IS/OOS split
- Current references:
  - self-reference in module docstring
  - mentioned in `breakout_sweep.py` comments/docstring
  - listed in `pyproject.toml`
  - mentioned in planning/audit docs
- Decision: move now
- Reason: clearly research-only, not part of active CLI/runtime flow, no tests import it, and no active code depends on it
- Risk notes: preserve root path via wrapper because it is still listed in packaging metadata and may be run manually by historical name

### 2. `breakout_sweep.py`

- File/path: `breakout_sweep.py`
- Purpose: stage-3 breakout parameter sweep using a fixed historical dataset and assumptions carried forward from `breakout_eval.py`
- Current references:
  - self-reference in module docstring
  - comment references to `breakout_eval.py`
  - listed in `pyproject.toml`
  - mentioned in planning/audit docs
- Decision: move now
- Reason: clearly historical research-only and not tied to current active research workflows or tests
- Risk notes: preserve root path via wrapper for manual reproducibility

### 3. `compare_mr_strategies.py`

- File/path: `compare_mr_strategies.py`
- Purpose: controlled comparison experiment between SMA and VWAP mean-reversion variants on a fixed dataset
- Current references:
  - self-reference in module docstring
  - listed in `pyproject.toml`
  - mentioned in planning/audit docs
- Decision: move now
- Reason: research-only, standalone, not imported by tests or runtime code
- Risk notes: preserve root path via wrapper for manual use and packaging continuity

### 4. `vwap_sweep.py`

- File/path: `vwap_sweep.py`
- Purpose: walk-forward parameter sweep for VWAP Z-score mean reversion on a fixed historical dataset
- Current references:
  - self-reference in module docstring
  - referenced by `vwap_atr_sweep.py` docstring/context
  - listed in `pyproject.toml`
  - mentioned in planning/audit docs
- Decision: move now
- Reason: clearly historical sweep-era research, not part of the active runtime or current research wrapper cluster
- Risk notes: preserve root path via wrapper because a later historical script refers to it conceptually

### 5. `vwap_atr_sweep.py`

- File/path: `vwap_atr_sweep.py`
- Purpose: second-stage sweep adding ATR percentile filtering to the earlier VWAP mean-reversion sweep
- Current references:
  - self-reference in module docstring
  - references `vwap_sweep.py` in module docstring/context
  - listed in `pyproject.toml`
  - mentioned in planning/audit docs
- Decision: move now
- Reason: clearly historical research-only, standalone, and not on current active flows
- Risk notes: preserve root path via wrapper for reproducibility

## Related Nearby Files

### `universe.py`

- Decision: defer
- Reason: not clearly part of the same sweep-era experiment set; it is a supporting module and already documented as intentionally standalone

### `run_breakout_research.ps1`

- Decision: defer
- Reason: wrapper script, not one of the historical Python experiment modules requested here

## Overall Recommendation

Move the five audited files into `research/legacy_experiments/` now.

Do not delete them. Leave root-level wrappers that:

1. re-export module contents
2. preserve `python old_name.py` behavior through `main()`

This gives a cleaner repo root while preserving historical reproducibility and minimizing risk.
