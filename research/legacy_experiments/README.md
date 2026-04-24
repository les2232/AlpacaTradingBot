# Legacy Experiments

This folder holds older sweep-era and historically important experiment runners that are preserved for reproducibility and reference.

What belongs here:

- historical one-off experiment scripts
- parameter sweeps tied to fixed datasets or dated research phases
- scripts that are still useful to rerun for comparison, but are no longer part of the main active research flow

What does not belong here:

- runtime-sensitive code
- current operator workflows
- newer active research validation scripts that belong in `research/`

Notes:

- These files are kept intentionally rather than deleted.
- Root-level compatibility wrappers may still exist so older commands and imports continue to work.
- New experiments should generally go in `research/`, not here.
