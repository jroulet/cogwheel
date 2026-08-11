# Build Brief: Interior cusp serving barrier — investigate and fix

## Mission

Understand and fix why interior cusp sources (inside the caustic near a cusp vertex, the 3-comparable-images regime) are refused by the serving ladder at ALL w, and get them served by a fast path. This is the last unserved cusp region.

## Findings (collected 2026-08-11, driver measurements)

1. **The `_cusp_vertex` routing fix (source-plane-nearest cusp) is NECESSARY but NOT SUFFICIENT.** It correctly selects the cusp vertex (distance 0.14, correct vertex), but interior cusp sources STILL refuse:
   - Interior source (gamma=0.5, |y|=0.9·r_caustic at phi=0.02), w=60/100/150/200/300: `cusp_amplification` returns None at ALL w.
   - R at w=150 is ~8.2 (above radius_min=7.37), so the radius gate is not the blocker.
   - Even with `envelope_bar=0.5` (relaxed), it refuses — not the envelope bar.
2. **The refusal is in the arm's CERTIFICATION** (the calibration certificate — scaled delays vs stationary-phase match, or the normal-form evaluation), NOT routing or the radius gate.
3. **ppGO returns a finite value at the same interior sources** (measured ppGO = -1.12+1.08j at w=150) — so the physics IS evaluable; only the Pearcey path refuses to certify.
4. **The exterior path is unaffected by the routing fix** (exterior serves via ppGO at w=200).
5. **Stale test assertions** (4 failing, from the routing-fix build):
   - `test_cusp_vertex_uses_o1_geometry_calls` — asserts O(1) geometry calls; the multi-cusp probe is O(n_cusps)~44. Update to assert bounded (constant) call count.
   - `ServedValueOldVersusNewTestCase` (2 tests) — old seed-theta vs new source-distance finder differ for interior sources; needs the carve-out (assert new vertex has smaller |source-vertex|, skip equivalence where cusps differ).
   - `ServedValueVertexInsensitivityTestCase` — verify it still holds (the served value should be insensitive to vertex-angle perturbation regardless of selection).

## The problem to solve

Interior cusp sources are the diffraction regime (R < 71) where the Pearcey uniform asymptotic is the CORRECT physics (3 coalescing images), yet the arm refuses. They currently fall to the exact engine. The goal: serve them fast.

## Work (investigate first, then fix)

1. **Identify the exact refusal gate**: instrument `cusp_amplification` for an interior cusp source to find which check refuses (calibration certificate? `_soft_normal_form`? `_cusp_vertex` returning None? the stationary-phase match?). Determine WHY the calibration fails for interior configs.
2. **Candidate fixes to evaluate** (the user suggests these):
   a. **Fix/relax the calibration certificate** for interior cusp configs — is the certificate's assumption (source outside caustic, well-separated) violated inside? Can it be made correct for the 3-image interior regime?
   b. **Bounded F - P_asymp residual table limited to the interior regime** — resurrect the residual idea but ONLY where |P_asymp| is bounded (the earlier attempt blew up near the fold caustic |P_asymp|~1e9; the interior cusp regime away from the fold may be bounded). Evaluate whether a residual table over the INTERIOR cusp region (small R, |x| bounded) is viable.
   c. **Verify the wedge/tube interior charts with cusp-adapted u already cover the interior cusp** — the `InteriorWedgeChart` (positive parity) with the u=d^(2/3) coordinate may already serve the interior cusp; if so, the arm refusing there is fine (it's not the right rung for interior). Check the actual serving ladder for interior cusp sources.
   d. **ppGO for interior?** — ppGO returns finite values; is it ACCURATE for interior cusp (3-image)? If the R^(-3/2) error bound holds, maybe lower the ppGO threshold for the interior. But the Professor noted interior R~1-4 ≪ r_ppgo_min, so ppGO likely inaccurate there — verify.
3. **Fix the serving**: whichever candidate works, make interior cusp sources serve via a fast path (table / ppGO / wedge chart), refusal-conservative. Verify no live quadrature.
4. **Update the stale tests**: the O(1) geometry-call assertion, the old-vs-new carve-out, and any others.

## Measured facts (re-probe at HEAD before coding)
- Interior cusp source (gamma=0.5, |y|=0.9·r_caustic(phi=0.02)): refuses at w=60..300, R~8 at w=150, even with envelope_bar=0.5
- ppGO at same source, w=150: -1.12+1.08j (finite) — but accuracy for interior unverified
- Routing fix: `_cusp_vertex` now selects source-plane-nearest cusp (in tree, +178 lines) — keep it
- Stale tests: DirectCuspVertexCorrectnessTestCase, ServedValueOldVersusNewTestCase, ServedValueVertexInsensitivityTestCase (test_lensing_airy_fold.py)
- Serving ladder: surrogate charts (wedge/tube interior) -> geometric -> arms (fold+cusp) -> Schwinger -> exact
- The residual-table attempt (P - P_asymp) was reverted: blew up near fold caustic (|P_asymp|~1e9)

## Constraints
- Fast tests. Follow AGENTS.md.
- The interior cusp is the diffraction regime (R<71) where Pearcey is the correct physics — it SHOULD serve, not refuse. Do not accept "falls to exact engine" as the answer unless a real barrier is identified.
- No live quadrature in the hot path.
- Refusal-conservative.
- Keep the routing fix (it's correct and necessary).
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The user: "collect your findings + point out the stale assertions and launch a new build to figure out a fix. It could involve resurrecting some aspects of the reverted fix with the F - F_ppgo but maybe with different limits? Or maybe there's a different treatment." Investigate the actual refusal gate first (the certification check), then evaluate the candidates. The residual idea may be viable if bounded to the interior regime where |P_asymp| is finite.
