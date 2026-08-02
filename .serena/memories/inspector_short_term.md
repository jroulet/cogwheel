# Inspector Short-Term Observations

## Build born-residual-wiring — Wire Born Residual Chart into Serve Path (2026-08-02, pass 3)

### Scope
Pass-3 re-review of WP1: Implement BornResidualChart dataclass and wire into
the serve path at the fact-4 slot in `_surrogate_coefficients`.

**Files changed (production):**
- `cogwheel/lensing/born_residual_chart.py` — NEW (untracked): frozen dataclass
  with `covers()` (box containment) and `evaluate()` (lazy-cached 3-D cubic
  RegularGridInterpolator for real/imag parts)
- `cogwheel/lensing/likelihood.py` — `born_residual_chart=None` kwarg added
  to `__init__`, `get_init_dict` handling, fact-4 slot wired with carrier +
  chart residual reconstruction, new imports: `math`, `types`,
  `FARFIELD_KERNEL_SUM`

**New test file:**
- `cogwheel/tests/test_lensing_born_residual_wiring.py` — 34 tests across
  6 test cases, all pass in ~3.8s

### Previous Findings Re-Check

**INS-11-001 (design) — STILL OPEN:** Confirmed by runtime probe:
`caustic_rho(0.0, 2.0, 0.0)` raises `ZeroDivisionError`, which is NOT
caught by `except (ValueError, LensDomainError)` at line 1673. The prior
open-interval (0, 1.6) makes gamma=0 measure-zero in sampling, but the
exception spec is incomplete for the public API.

**INS-11-002 (trivial, Librarian) — STILL OPEN:** SPEC.md line 132 still
says "STILL NOT wired into the serve path on either parity" and line 133-134
"that chart is a TRAIN_TIER artifact that does not yet exist". Both are
stale — the code NOW conditionally wires the slot when `born_residual_chart`
is non-None.

**INS-11-003 (trivial) — STILL OPEN:** The deferred import comment at line
1694 says "born_carrier_from_partition's module imports channels which may
circle back at module load" — but `born_carrier_from_partition` IS IN
channels.py (it doesn't import channels; it IS channels). channels.py does
NOT import likelihood.py (verified grep: zero hits). The other symbols from
channels.py (`reconstruct_farfield`, etc.) are already top-level imported at
line 95. The deferred import is unnecessary and the comment's rationale is
factually incorrect (the actual cycle in channels.py is `_born` importing
channels, which is resolved by a deferred import INSIDE
`born_carrier_from_partition` itself — not related to the likelihood→channels
direction).

### New Finding

**INS-12-001 (trivial/design):** Born path demodulation at line 1710
uses `np.exp(1j * dense_w * geom.t_min)` instead of the authoritative
`_frame_phase(dense_w, geom.t_min)` helper. The `_frame_phase` docstring
(channels.py line 1124) says it is "the SINGLE authoritative source of that
phase, called by BOTH sites" and coder_knowledge mandates it (Build 8h-d2).
`reconstruct_farfield` uses `_frame_phase` for its re-modulation. The
mismatch is functionally safe (libm handles |w*t_min|=223 arg reduction
precisely; the tests pass at 1e-13) but violates the stated "single source"
convention. Fix: import `_frame_phase` from channels.py and use
`np.exp(1j * _frame_phase(dense_w, geom.t_min))`.

### Production Code Assessment
The fact-4 Born residual path (lines 1662-1714):
- Correctly guards: kappa==0 (upstream line 1558), beta==0 (upstream line
  1571), born_chart is None, rho > 1.0, chart box containment via
  `covers(gamma, rho)`
- Exception handling for `caustic_rho`: catches `ValueError` and
  `LensDomainError` but misses `ZeroDivisionError` at gamma=0 (INS-11-001)
- Partition namespace carries all attributes `born_carrier_from_partition`
  reads: w, source, gamma, beta, kappa, matrix, t_min, delays,
  saddle_kernels, real_mask, images ✓
- Reconstruction algebra: `(f_total - ppgo) * exp(1j*w*t_min)` demodulates,
  `reconstruct_farfield` re-modulates via `_frame_phase` (mod 2π); the
  round-trip cancels to machine precision ✓ (minor: producer side should
  also use _frame_phase for convention compliance — INS-12-001)
- Uses `FARFIELD_KERNEL_SUM` (S_a=1 on real channels, tau_c=0): correct
  for a total that already includes all channel contributions ✓
- Returns `(delays, k0, k1, geom)` matching the expected 4-tuple shape ✓
- Pickle round-trip: `born_residual_chart` rides in `__dict__`, preserved
  by `__getstate__`/`__setstate__` ✓
- JSON round-trip: `get_init_dict` pops when None, raises NotImplementedError
  when non-None (same pattern as `amplification_surrogate`) ✓
- New `born_residual_chart` kwarg is keyword-only with default None — all
  existing callers remain compatible ✓

### BornResidualChart Assessment
- Frozen dataclass with lazy-cached `RegularGridInterpolator` (cubic, 3-D)
- `object.__setattr__` for caching on frozen dataclass — correct pattern
- `bounds_error=False, fill_value=None` means cubic extrapolation outside
  grid — safe IFF chart coverage matches the band (training driver's job)
- `covers()` checks only (gamma, rho) not the w-axis — training-driver-
  responsibility contract for w-band coverage
- No `__post_init__` validation of grid size ≥ 4 — trivial gap, not
  production-reachable

### Test Suite Assessment
- 34 passed in 3.77s (well under budget)
- Tests cover: no-chart returns None (HEAD behavior), chart-present serves,
  out-of-box fallthrough (rho above/below/interior, gamma outside), kappa/beta
  guard precedence, self-falsification (wrong residual detectable, covers()
  has teeth, rho guard fires independently of chart, mock surrogate declines)
- Anti-vacuity patterns (n_checks, tearDown) applied throughout ✓
- Non-circular: test independently computes carrier + residual and compares
  against the served k0/k1 reduction ✓

### Existing Test Survival (verified this pass)
- test_lensing_born_residual_wiring.py: 34 passed (3.77s)
- test_lensing_surrogate.py: 69 passed (111.85s)
- test_lensing_ghost_gate.py: 18 passed (5.48s)
- test_lensing_likelihood.py: 17 passed, 12 skipped, 1 xfailed (26.69s)

### Carried Forward (pre-existing)
- INS-10-001: DATA_CONTRACTS.yaml "uniform" gamma phrasing stale (Librarian)
- INS-9-001: Dead allowlist entries in test_lensing_part0_mechanical.py (trivial)
- INS-8-001: test_raising_constant_to_two_refuses_an_admit_config fails (pre-existing)
- INS-5-001: SPEC.md old annulus references — Librarian
- INS-5-003: DATA_CONTRACTS.yaml line 228 'caustic-frame annulus rho' — Librarian
- Pre-existing: GhostFrameCollapseTestCase failures (GhostDomainError, 3 failures)

### Verdict
**PASS** — no blocking issues. Three previously-open findings remain
(INS-11-001 design, INS-11-002/003 trivial/Librarian). One new trivial
finding (INS-12-001: _frame_phase convention). The code is functionally
correct, all tests green, existing suites unregressed.
