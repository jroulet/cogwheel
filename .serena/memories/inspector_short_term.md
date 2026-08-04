# Inspector Short-Term Observations

## 2025-08-03: Build schwinger_qd review (third pass — final)

### Scope
WP1: Add _f_schwinger_mpmath() in _schwinger.py for w > 60 (up to 150)
with paired N/2N certification, lazy mpmath import.
WP2: Raise _SADDLE_W_CEILING from 58→148 in surrogate_training.py;
update operator.py routing to use W_CEILING_SCHWINGER_QD as the
geometric/arm threshold.

Files changed (production):
- cogwheel/lensing/chang_refsdal/_schwinger.py (new W_CEILING_SCHWINGER_QD=150,
  new _f_schwinger_mpmath, updated f_schwinger dispatch)
- cogwheel/lensing/chang_refsdal/operator.py (routing pivot moved from
  W_CEILING_SCHWINGER to W_CEILING_SCHWINGER_QD, new mpmath sequential batch)
- cogwheel/lensing/surrogate_training.py (_SADDLE_W_CEILING 58→148)
- pyproject.toml (new [training] extra with mpmath)

Files changed (tests):
- cogwheel/tests/test_lensing_schwinger.py (new test classes, updated fixtures)
- cogwheel/tests/test_lensing_operator.py (ONEHOME_WS updated, ceiling references)
- cogwheel/tests/test_lensing_batched_operator.py (XOR_BAND_LS capped, XOR_REFUSING_W)
- cogwheel/tests/test_lensing_waveform.py (BAND_EDGE w→59.9, HARD_CORE w→151)
- cogwheel/tests/test_lensing_surrogate.py (FLIP_REFUSAL_W→160)
- cogwheel/tests/test_lensing_airy_fold.py (_ABOVE_CEILING_W→160, _W_CEILING→QD,
  geometric node w→200)

### Previous findings RESOLVED
- INS-1-002: RESOLVED. BAND_EDGE w_probes now (30, 40, 59.9), all DD path.
  HARD_CORE w=151 correctly refuses. test_lensing_waveform.py: 26 PASS, 8.83s.
- INS-2-001: RESOLVED. Same fix; file runs in 8.83s, well below 5-min ceiling.
- INS-2-002: STILL OPEN (Librarian scope). SPEC.md not updated for QD ceiling.

### NEW FINDINGS

#### BUG: test_lensing_airy_fold.py — 2 tests FAIL (INS-3-001)
`UniformArmFallThroughTestCase::test_corrupted_certificate_falls_through_to_named_refusal`
and `test_nan_primitive_falls_through_to_named_refusal` both FAIL.
These tests use `_CUSP_NODE_W = 80.0`. Previously w=80 > 60 (old ceiling)
meant the Schwinger evaluator refused after both arms declined. Now w=80
is in the mpmath band (60 < w <= 150) and evaluates successfully.
The tests expect SchwingerCertificationError but get a successful result.
File was modified in this build (updating other constants) but _CUSP_NODE_W
was missed.
FIX: Either change _CUSP_NODE_W to 151.0 (above QD ceiling), OR
separate the fixture for the fall-through tests from the arm-serve tests
(since those at w=80 exercise the arm correctly at w<150).

#### BUG: test_lensing_levers.py — 1 test FAILS (INS-3-002)
`LMaxEnforcementBracketTestCase::test_wave_branch_serves_below_ceiling_refuses_above`
FAILS. Test uses `LEVER5_KERNEL_CEILING = _schwinger.W_CEILING_SCHWINGER = 60`
and expects refusal at `LEVER5_KERNEL_CEILING + 1 = 61`. But w=61 is now in
the mpmath band and evaluates successfully. File was NOT modified in this
build but the production change broke it.
FIX: Change the above-ceiling probe from `LEVER5_KERNEL_CEILING + 1.0` to
`_schwinger.W_CEILING_SCHWINGER_QD + 1.0` (= 151.0).

#### TRIVIAL: SPEC.md not updated for W_CEILING_SCHWINGER_QD (INS-2-002, carried)
SPEC.md says: "oracle-certified to the 1e-10 bar (F005) up to its ceiling
w <= W_CEILING_SCHWINGER = 60, above which it refuses by name". This is
now incorrect — f_schwinger evaluates up to w=150 via mpmath. The spec
should describe the two-tier ceiling (DD=60, QD=150). Librarian scope.

### Math and correctness verified:
- IBP structure in _f_schwinger_mpmath matches _raw_t_integral_core exactly
- Certification on reconstructed F (post-prefactor) is deliberate and safe
  for mpmath precision (dps = 30 + ceil(w) >> needed digits)
- Mass-sheet phase formula in mpmath batch identical to DD batch
- Refusal ordering (lowest-index-first) preserved across DD + mpmath + ceiling
- DD path byte-identity confirmed by DdPathByteIdentityTestCase (golden hex)
- Lazy import pattern structurally verified by MpmathLazyImportTestCase
- Operator routing: 3-way split (DD w<=60 / mpmath 60<w<=150 / above-QD w>150)
  is correct in both _saddle_grid and _positive_parity_grid

### Passing test files:
- test_lensing_schwinger.py: 66 passed, 3 skipped, 229s
- test_lensing_waveform.py: 26 passed, 8.83s
- test_lensing_operator.py: 15 passed, 50s
- test_lensing_batched_operator.py: 14 passed, 83s
- test_lensing_surrogate.py: 69 passed, 125s
- test_lensing_saddle_geometry.py: 30 passed, 7s
- test_lensing_chang_refsdal_ghost_frame.py: 12 passed, 4s

### Failing test files:
- test_lensing_airy_fold.py: 2 FAIL (INS-3-001)
- test_lensing_levers.py: 1 FAIL (INS-3-002)

### Open issues:
- INS-3-001: test_lensing_airy_fold.py 2 test failures (bug)
- INS-3-002: test_lensing_levers.py 1 test failure (bug)
- INS-2-002: SPEC.md not updated (Librarian scope, trivial)
