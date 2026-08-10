---
date: 2026-08-10
section: Backlog
---
## Exterior fold-carrier phase demodulation

Implemented in build `exterior_fold_carrier_demodulation`, commit b061103.

- `_needs_fold_carrier` added to `surrogate_training.py`: probes tile corners and
  centre via `geometry.ghost_kernel`; returns True when a ghost EXISTS anywhere in
  the tile (regardless of `Im(tau_c)` — existence, not dominance). Replaces the
  prior `_exclude_ghost_dominated` call in the tiler: ghost-dominated tiles are no
  longer dropped, they flow through for fold-carrier training. Positive-parity only.
- `_compute_rho_carrier` added to `surrogate.py`: for each rho grid node, probes
  `geometry.ghost_kernel` at every `(gamma, theta_c)` node and takes the median
  `Re(tau_c)` over valid nodes. Returns None when no ghost exists anywhere in the
  tile.
- `ExteriorPolarChart` gains `rho_carrier` field (np.ndarray or None, default None):
  when not None, from_values demodulates the envelope by
  `exp(-1j*w*rho_carrier[rho_node])` before the residual `carrier_rate` demodulation;
  serve re-modulates in reverse order.
- Schema bumped to `'exterior_polar_rho_log_carrier_v1'`
  (`_EXTERIOR_POLAR_AXIS_SCHEMA_V4`); old tags hard-refuse at load.
- `ghost_excluded_tiles` counter stays 0 by design; `CarrierDiscontinuityError` is
  the safety net for unrescuable tiles.
- Ghost-transition zone (~40% of exterior prior box) recovered at surrogate speed.
  Measured rho-phase winding: 16.7 → 3.2 rad over rho in [1.3, 2.1].
- DT-7 `FarfieldExteriorTilesGhostExclusion` tests retired (assert removed
  exclusion behavior). 258 build tests + 75 admission tests green.
- `test_lensing_exterior_admission.py`: 241 lines removed (DT-7 tiler ghost-exclusion
  wiring tests). Ghost-domination unit tests (`_exclude_ghost_dominated` directly)
  remain.

ACCEPTANCE: full-box exterior probe producing ~70 charts with all held-out eps under
the 1e-3 bar, and census confirming ghost-region draws served by surrogate (not
falling to exact engine), are driver post-build verification steps.
