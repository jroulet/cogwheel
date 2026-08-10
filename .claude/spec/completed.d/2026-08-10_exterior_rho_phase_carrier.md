---
date: 2026-08-10
section: Backlog
---
## Exterior ghost-region tile exclusion

Implemented in build `exterior_rho_phase_carrier`, commit cf81d66.

- `_exclude_ghost_dominated` added to `surrogate_training.py`: probes tile
  corners and centre in eigenframe source coordinates via `geometry.ghost_kernel`.
  Excludes tiles where the ghost exists but `Im(tau_c) < _GHOST_DECAY_IM_THRESHOLD`
  (the ghost-transition zone where the unsubtracted ghost dominates `KERNEL_SUM`
  residual ~2-3x — no spline can fit it). Ghost non-existence → retainable
  (`KERNEL_SUM` is ghost-free there). Positive-parity only.
- Wired into `_farfield_exterior_tiles` via `gamma`, `gamma_band`, and
  `ghost_drop_count` optional params; mirrors `_exclude_near_cusp` in logic and
  gamma-band probe pattern.
- `ghost_excluded_tiles` counter recorded in the per-band exterior region report.
- 19 ghost tests pass in `test_lensing_exterior_admission.py` (DT-1 through DT-7)
  and `test_lensing_surrogate_training.py` (DT-8). 174 admission+training tests green.

ACCEPTANCE: exterior probe producing ~70 charts with all held-out eps under the
1e-3 bar at the 4x4x4 node count, and the census showing ghost-excluded draws
falling to the exact-engine ladder, are driver post-build verification steps
(bulk-training sweep, not in-build).
