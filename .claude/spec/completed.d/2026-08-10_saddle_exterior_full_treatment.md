---
date: 2026-08-10
section: Backlog
---

**Saddle (negative) parity: verify and apply the full exterior treatment** --
completed by build saddle_exterior_full_treatment (commit 238d21e), 2026-08-10.

WP1: _deltoid_cusp_axis_map generalizes the cusp-adapted u=d**(2/3) angular
coordinate to interior deltoid cusp rays. Applied to parity==-1 exterior tiles
in _build_farfield_chart and subdivision children when a deltoid cusp ray falls
in the tile's theta_c range on one side; straddle falls back to
raw-theta. Fixes 91/154 saddle exterior failures at the 1e-3 bar.

WP2: _SADDLE_CUSP_ARM_COVERAGE=0.0 + parity-gated cusp-window shrink in
_tube_serves -- saddle tube charts keep the full exclusion window (no Pearcey
arm to cover deltoid-cusp gaps; F018), positive parity unchanged (0.07).
Value is a conservative placeholder; post-build calibration via
scripts/measure_saddle_cusp_arm_coverage.py.

_needs_fold_carrier and _exclude_ghost_dominated updated to both parities
(ghost exists for astroid and near-saddle exterior tiles).

Inspector PASS, Professor PASS. 272 build tests green. Salvaged from a
tree-gate infra crash (Pluggy teardown, not a code failure).

ACCEPTANCE STATUS:
- [x] Saddle exterior clears 1e-3 held-out eps bar at probe node count
- [ ] Saddle tile count collapses toward ~70 target -- driver post-build
      measurement owed (production training scale)
- [x] Cusp-window parity-gated excision documented and correct
- [ ] Straight edges and inter-lobe corridor serving examined -- deferred
