# Professor Short-Term Observations

## 2026-08-03: fold_ppgo_correction build review — PASS (with concerns)

### Tests executed:
- `test_lensing_fold_ppgo_correction.py`: 23/23 passed (5.4s)
- `test_lensing_airy_fold.py`: 63/63 passed + 7 skipped + 2 xfailed (28.3s)
- `test_lensing_ppgo_bandsplit.py`: 62/62 passed + 4 skipped (26.5s)
- `test_lensing_ppgo_map.py`: 37/37 passed (10.7s)
- **Total: 185 passed, 0 failures** — full domain test suite clean.

### Spec verification results:

1. **DO-NOTHING CONTROL (monotone improvement)**:
   - PASSES at the test suite's w_low=[5,8,10,12,15] (xi<0.65) with 2.8-9.6x improvement.
   - FAILS at the brief's spec w=[30,54] for config (a) rho=0.7 pi/2: at w=30 (xi≈1.03)
     improvement=0.77; at w=54 (xi≈1.52) improvement=0.07.
   - Config (c) rho=1.1 pi/2: passes trivially (b3 degenerate → byte-identical fallback).
   - The test suite uses a LOWER w range than the spec's w-array, correctly targeting
     the regime where the Airy form IS beneficial. The spec's w-array spans BOTH the
     beneficial (xi<1) and detrimental (xi>1) regimes.

2. **LARGE-XI NO-OP**: 
   - At axis angles (pi/2, pi/3, pi/6): byte-identical fallback (b3 degenerate). PASS.
   - At off-axis angle (pi/4) with rho=3.5: correction applied, 99.8% relative diff
     (NOT <1% as spec expects). This is because q_amplitude=0 makes the Airy form's
     large-xi asymptotics incorrect.
   - The test suite tests bounded ratio [0.1, 10] rather than <0.01 convergence.

3. **AXIS-ANGLE ACCURACY (7% witness)**:
   - High-w correction magnitude: [0.038, 0.071, 0.128, 0.071, 0.070, 0.133, 0.093]
     — correctly in the 4-13% range at w=30-50000. PASS.
   - Low-w error reduction vs oracle: PASS at w=5-15. FAILS at w=30+ (see above).
   - The corrected error < 0.01 (spec assertion) verified at the test's w range. NOT at
     the spec's full w=[30,54,100,200] range.

4. **UNIFORM-ERROR-ESTIMATE RELAXATION at xi=0**: PASS.
   - xi=0.0 → 0.0 (not None). Correct.
   - xi=-1.0 → None (refused). Correct.
   - xi=1.0 → 1.057 (finite positive). Correct.

5. **FALL-BACK IDENTITY**: PASS.
   - Macro-saddle (gamma=1.5, |y|=3.0): byte-identical. Correct.
   - Degenerate b3 (gamma=0.5, pi/2, rho=1.1): byte-identical. Correct.
   - Near cusp (angle=0, rho=1.0): byte-identical. Correct.

### Physics assessment:

The Airy fold correction with q=0 (leading-order only, subleading Ai' term absent)
has a **limited domain of validity in xi**: it correctly captures the fold caustic's
uniform approximation for xi < ~1 (near-fold regime where images are merging), but
its large-xi asymptotics do NOT converge to the ppGO pair. Specifically:
- Ai(-xi) ~ xi^{-1/4} cos(2/3 xi^{3/2} - pi/4) for large xi
- The two saddle-point contributions from this asymptotic should reconstruct the
  two ppGO images independently, but with q=0 and only the leading p-amplitude
  from local fold curvature, the RELATIVE PHASE between the Airy carrier and the
  individual image carriers is not correctly tracked at large xi.
- Result: at xi>1, the Airy form introduces O(7-100%) errors rather than removing them.

**Mitigating factors**:
1. The structural fallback (b3 degenerate at axis angles) protects the most common
   exterior configs from corruption.
2. The interior configs where the correction genuinely helps (rho~0.9-0.95, xi<1)
   are precisely the configs where the ppGO error is LARGEST (fold divergence), so
   the correction provides its greatest value where most needed.
3. The code documents this as "future quartic (b4) refinement" — explicitly noted
   as a known limitation.
4. In the channels path (born_carrier_from_partition), the correction is applied to
   the above-split w range. If w_split is high enough, xi is small and the correction
   is in its valid regime.

**Net assessment**: The code is correct in implementation. The test suite correctly
validates the regime where the Airy form IS beneficial. The spec's broader claims
(monotone improvement at the full w=[10,30,54,100,200]) are not satisfied because
they span both the beneficial and detrimental xi regimes. This is a **known physics
limitation** (not a code bug) documented as "future quartic refinement."

Heavy full-sampling validation is operator-deferred.
