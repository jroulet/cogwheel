---
date: 2026-08-07
section: Backlog
---

### Training entry point gains regions filter

Closes `lensing_training_path_cannot_be_run_per_region`.

`train()` and `_train_band_charts()` now accept a `regions` parameter
(`tuple[str, ...] | None`). Passing a subset (e.g. `regions=('wedge_interior',)`)
restricts training to those chart types without any pipeline reimplementation.
`regions=None` is byte-identical to the previous behavior. The `--regions` flag
was added to `scripts/train_lens_surrogate.py` for command-line use.

Acceptance: 28 tests pass covering per-region runs; the wedge probes from
[[2026-08-07_subdivision-recursion-wedge-v3-r-caustic]] can now be expressed as
`train(..., regions=('wedge_interior',))` calls. Interior-only band runs complete
in minutes rather than the ~40 min full-path cost.
