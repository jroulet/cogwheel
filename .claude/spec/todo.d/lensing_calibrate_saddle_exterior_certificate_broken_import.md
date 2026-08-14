---
section: Backlog
---

- **`scripts/calibrate_saddle_exterior_certificate.py` no longer imports —
  its module docstring's "at HEAD" claim is now false** `[housekeeping]` —
  the script (added in the `a4ba536` driver calibration commit) imports
  `W_FLOOR, _exact_total_w, _min_delta_tau, _polar_source, _tier1_serve`
  from `cogwheel.tests.test_lensing_saddle_tier1_accuracy`; that test module
  was deleted by the very next commit (`1c90b3a`, build
  `symmetry_tie_c3_admission`), which retired the rho-floor gate the script
  was calibrating and shipped `test_lensing_saddle_serve_gate.py` in its
  place. The script's own docstring (lines ~24-33) still asserts it is
  reusing "pairing-validated production-shaped plumbing... at HEAD" and
  that the (also since-deleted) `scripts/measure_saddle_eta_floor.py` is
  the one that's stale — both claims are now backwards. The script already
  did its job once (it produced the committed
  `scripts/calibration_pilot_followup.json`, cited as calibration
  provenance in SPEC.md's tier-1 saddle row), so this is not blocking
  anything today, but the script cannot be re-run as committed. Librarian
  scope excludes code edits (this is a `scripts/*.py` import, not a doc/spec
  surface) — flagging for whoever next touches saddle calibration: repoint
  the import to `test_lensing_saddle_serve_gate.py`'s equivalent helpers
  (names likely changed along with the gate) or inline the needed helpers,
  and correct the docstring's stale HEAD-state claims.
