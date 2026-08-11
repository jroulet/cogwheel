---
section: Backlog
depends_on: []
---

- **`test_thresholds_have_one_home` — `select_branch` vs served routing disagreement (one-home pin RED)**
  `[→ spec]` — measured 2026-08-11 by the driver.  Build brief written:
  `.claude/handoff/brief_operator_routing_one_home.md`.

  `cogwheel/tests/test_lensing_operator.py::BranchGateTestCase::
  test_thresholds_have_one_home` fails at HEAD (verified pre-existing, not
  build-caused).  Saddle node `gamma=1.2, kappa=0, y=(0.04,0.03), beta=0.7,
  w=500` (also w=1000 and gamma=2.0): `select_branch(w, delta_min, inf,
  eta) == 'wave'` (w*delta_min = 1.90 < RHO_END = 4.0) but the grid SERVES
  via the cusp arm's ppGO rung a value 1 ULP from `geometric_amplification`;
  the test's no-mock `_observed_branch` (served bit-equal to geometric
  ⇒ 'geometric') reads 'geometric' after numba warmup, so the pin fails.

  OPTIONS (Professor adjudication required — see the brief): (a) the grid is
  wrong to serve an UNRESOLVED node (w*delta_min < RHO_END) via cusp ppGO —
  add a resolution guard to the ppGO rung; (b) the test probe is stale —
  cusp ppGO legitimately equals the geometric limit, sharpen
  `_observed_branch`; (c) hybrid.  Refusal-conservative; do NOT weaken the
  other ~40 one-home checks or the byte-identity contracts.

  ACCEPTANCE: the test passes with a non-zero comparison count; the chosen
  option is physics-justified; `test_lensing_operator.py`,
  `test_lensing_fast_path.py`, `test_lensing_airy_fold.py` green; no
  regression in the eta-leg-live assertion.

  This is now the ONLY remaining item in `lensing_serving_ladder_guards_are_red`
  — `test_refusal_precedes_coherent_score` was resolved by the mpmath
  fixed-panel-rule build (completed 2026-08-11).  SCHEDULED: launch as its
  own build using the existing brief.
