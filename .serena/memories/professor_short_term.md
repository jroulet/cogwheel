## professor_short_term — 2026-08-10 Build Review

### Tests run
- `test_lensing_exterior_polar_fold.py`: 41/41 PASSED. All rhocarrier round-trip, phase, magnitude, composition, schema, NPZ, from_engine tests pass.
- `test_lensing_surrogate_training.py` (fast subset): 18/18 PASSED (FoldCarrierNeedsGhost, FoldCarrierContinuitySafetyNet). 9 SKIPPED — require COGWHEEL_TRAIN_TIER=1 (operator-deferred heavy tier).
- `test_lensing_exterior_carrier.py`: 23/23 PASSED.
- `test_lensing_surrogate.py`: 111/111 PASSED.

### Concern
`test_lensing_farfield_envelope.py` fails COLLECTION: imports `_EXTERIOR_POLAR_AXIS_SCHEMA_V3` which was renamed to `_EXTERIOR_POLAR_AXIS_SCHEMA_V4` (tag 'exterior_polar_rho_log_carrier_v1') in this build's surrogate.py. Three references need updating (import line 149, usages at lines 1090 and 2220).

### Operator-deferred
FoldCarrierTrainingIntegration* tests and ghost_drop_count census — require full engine-backed training (COGWHEEL_TRAIN_TIER=1, minutes per class).
