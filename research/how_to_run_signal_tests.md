# How To Run Signal Tests

## 1. Add One Score Mode

- Add one new `score_mode` in `research/canonical_rank_pullback.py`.
- Keep the change limited to the ranking signal only.
- Do not change pullback logic, exits, or validation structure in the same experiment.

## 2. Run The Audit

Example:

```powershell
python run_canonical_cross_sectional_rank_audit.py --score-mode return_20 --output-dir results/canonical_cross_sectional_rank_audit_r20
python run_canonical_cross_sectional_rank_audit.py --score-mode return_20_plus_60 --output-dir results/canonical_cross_sectional_rank_audit_r20_r60
```

## 3. Inspect This File First

- Open `bucket_summary.csv` first.

What to look for:

- does `top > middle > bottom` hold?
- is `top_minus_bottom > 0`?
- does that happen across multiple horizons?

## 4. Classify The Result

- `promising`
  - clean ordering and positive spread across multiple horizons
- `weak`
  - some separation exists, but it is incomplete or tiny
- `no signal`
  - ordering is mixed and spread is near zero
- `inverted`
  - top loses to bottom or spread is negative

## 5. When Validation Is Allowed

- Only run validation if the audit looks clearly good enough.
- If the audit is weak or inverted, stay at the signal layer.

## 6. Record The Result

- Add an entry to `research/signal_experiment_log.md`.
- Record:
  - signal name
  - hypothesis
  - exact formula
  - audit command
  - output folder
  - bucket summary numbers
  - classification
  - decision

## 7. Keep The Process Clean

- one signal change at a time
- no execution tuning on weak signals
- no casual grid expansion
- no optimistic reinterpretation of weak results
