# Professor short-term — 2026-08-11

## PpgoRungSelfFalsificationTestCase review

All 3 tests pass (5.06 s). Broader lensing suite: 199 passed, 23 skipped, 4 xfailed.
No regressions.

### Physics audit of `test_resolution_gate_isolated_admit_and_refuse`

Saddle fixture `_PPGO_SADDLE_SOURCE=(-0.5,0.5)` at `gamma=1.2`:
- 2 images, `delta_min = 0.644`, `nearest.distance = 0.389 > _ETA_MAX_FOLD=0.3`,
  no merging fold pair.
- At w=500: `w*delta_min = 322 >> _PPGO_RESOLUTION_GATE=4.0` — the gate ADMITS
  naturally. The spec's estimate `w*delta_min≈1.9` is **wrong** for this fixture
  (actual: 322).

Test design adapts correctly:
- (a) gate=4.0, w=500: admits (322≥4) ✓
- (b) gate=1000, w=500: refuses (322<1000) ✓ — proves gate has teeth
- (c) gate=0.0, w=500: admits ✓ — proves disable works
- (d) gate=1000, w=20000: admits (12882≥1000) ✓ — proves resolved nodes admitted

The gate logic is a simple `w*delta_min >= threshold` inequality. Both branches
are exercised. Variable isolation is clean: same w for admit/refuse branches,
varying only the gate threshold. This is actually stronger than the spec's
scenario (which would have used different w values for admit vs refuse).

### Spec-implementation discrepancy

Spec §(a) says "assert cusp_amplification returns None (ppGO rung refuses)" at
w=500 with gate=4.0. The actual fixture at w=500 is resolved (322>>4), so
refusal is physically impossible. The test instead proves refusal by inflating
the gate to 1000. The underlying claim — "w*delta_min≈1.9<4.0" — is factually
incorrect for this fixture (δ_min=0.644, not ~0.0038). The spec's numerical
estimate appears to be a copy-paste error from a different configuration.
Resolution: the self-falsification goal is met — the test proves the gate is
both active and correctly scoped. The spec's particular scenario was infeasible.

### Verdict: PASS
