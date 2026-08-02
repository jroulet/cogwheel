# Librarian Short-Term Observations

## 2026-08-01 — Catch CarrierDiscontinuityError in _eps_for

**Scope**: `cogwheel/lensing/surrogate_training.py` (error-handling), `cogwheel/tests/test_lensing_exterior_windows.py` (new test)

**Outcome**: No-op — all documentation surfaces already up to date.

**Why no edits were needed**:
- Change is purely internal error handling in a private function (`_eps_for`). No public API changed, no new module added, no new disk artifact produced (trace entry stays in-memory).
- SPEC.md does not name `_eps_for`, `_reprovision_w_nodes`, or `CarrierDiscontinuityError` in any pipeline step or module list.
- RST docs (`overview.rst`, `api.rst`, `crash_course.rst`) have no reference to these private symbols.
- `sync_derived_docs.py` reported "some issues auto-fixed" but `git diff` showed zero new dirty files — confirmed no-op state flush.
- The four consumer_graph warnings about test-file callers of `LensAmplificationSurrogate.load` are pre-existing and excluded from DATA_CONTRACTS.yaml by convention.

**Fragile cross-references to watch**:
- `tests_reachable_red_on_symptoms.md` TODO fragment mentions `test_unpatched_positive_box_build_raises_carrier_discontinuity` — that test presumably no longer exists post-fix; worth checking if the fragment can be completed in a future housekeeping pass.

**No surprises.**
