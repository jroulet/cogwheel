## Session 2026-08-10: Saddle exterior cusp-adapted u-coordinate review

### Verdict: PASS

All 32 tests across 6 specs PASS with no numerical concerns.

**Spec A (Round-trip)**: SaddleCuspUCoordinateRoundTripTestCase — 8/8 PASS. theta_to_u shape (2,≥100), strictly increasing rows, u_fine[0]~0, endpoint match within 1e-12, round-trip np.interp reproduces u_grid within rtol*max(u_grid), mismatched-row falsification has detectable error (~5e-5 per docstring). Diagnostic plot shows expected overlay: blue fine-map curve, red node points, vertical cusp-ray line.

**Spec B (Accuracy)**: SaddleCuspAdaptedAccuracyTestCase — 3/3 PASS. Chart A (cusp-adapted u) vs Chart B (raw theta_c) on 50 held-out points. Chart A median eps beat Chart B by ≥2x; Chart A eps ≤1e-3 production bar. d**(-1/3) divergence absorption confirmed for deltoid cusps.

**Spec C (Parity gating)**: TubeCuspWindowParityGatingTestCase — 10/10 PASS. Saddle (parity=-1) coverage=0.0 refuses all cusp-window interior queries (including mid-window, near-cusp). Positive (parity=1) coverage=0.07 admits only within the 0.07 rad shrink margin, refuses at 0.04 (inside residual), admits at 0.04+shrink outside. Physics correct: deltoid cusp arm is not near the true source in saddle parity.

**Spec D1 (Mutation self-falsification)**: SaddleThetaToUMutationSelfFalsificationTestCase — 6/6 PASS. With-chart eps ≤1e-3; without-chart eps is ≥1.5x different and ≤10× bar (not unbounded). Mismatched theta_to_u maps also produce distinguishable eps. Guards have teeth.

**Spec D2 (Coverage constant self-falsification)**: TubeCuspWindowParityGatingSelfFalsificationTestCase — 2/2 PASS. Positive with coverage=0.0 refuses (incorrectly), saddle with coverage=0.07 admits (incorrectly). Falsifications go red; restored constants pass.

**Spec E (Serving geography)**: SaddleCuspAdaptedServingTestCase — 3/3 PASS. Cusp-adapted eps ≤ raw-theta eps (non-degradation); cusp-adapted eps ≤1e-3 (production bar). Diagnostic scatter plot with y=x diagonal confirms.

### Physics notes
- A3 universality across astroid and deltoid cusps is confirmed numerically: d**(-1/3) absorption works on saddle exterior tiles with deltoid caustics exactly as on positive-parity astroid tiles.
- No numerical anomalies, no tolerance-edge results, no near-threshold concerns.
- Full-sampling validation (COGWHEEL_BRUTE_ACCURACY/COGWHEEL_STRICT_TIMING) remains operator-deferred — beyond fast-test scope.
