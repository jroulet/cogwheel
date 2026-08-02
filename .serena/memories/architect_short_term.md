# Architect Short-Term Observations

Build 3 (C6) re-planned: production code COMPLETE, but 4 test files still
reference the dead `eta_max`/`eta_floor` fields on TrainingConfig. This is
a test-only compatibility port (is_test_only=true, zero Coder WPs).

Broken files:
1. test_lensing_exterior_admission.py — pin test + _interior_admission call
   missing eta_max arg
2. test_lensing_exterior_windows.py — pin test + _saddle_lobe_admissions calls
   missing eta_max + self.config.eta_max refs
3. test_lensing_ppgo_bandsplit.py — _interior_admission call missing eta_max +
   dataclasses.replace(config, eta_max=...) + self.config.eta_max
4. test_lensing_surrogate_training.py — TrainingConfig(eta_max=..., eta_floor=...)
   invalid kwargs, config.eta_max/eta_floor refs, _build_tube_chart/_tube_heldout_samples
   missing eta_max/eta_floor args

Fix pattern: define local ETA constants, pass eta_max/eta_floor explicitly to
functions that now require them, delete stale pin tests, optionally pin f_max.
Professor confirmed: test fixtures should keep their original 0.05/0.02 values
as local constants (controlled comparisons, not production defaults).
Simplifier: lean, no compat property (semantically impossible), collapse
files 1+3 into shared description if disjoint constraint allows.

Triage (INS-1-004): SPEC.md doc staleness — eta_max foot-of-normal guard
description is stale vs curvature-relative f_max design. Override as
Librarian-routed (doc staleness is not a Coder defect; Inspector's own
diagnosis was correct). Suggested replacement text from Inspector:
"eta_max = f_max * R_c per arc, with the algebraic invariant f_max < 0.5
asserted at training time."