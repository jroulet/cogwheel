# Professor short-term (saddle_serve_gate INFERENCE REVIEW, 2026-08-14)

Reviewed the c3-led rewrite of `_saddle_farfield_analytic_serves` (likelihood.py
L555-621) + census mirror. Ran `test_lensing_saddle_serve_gate.py`: 33/33 PASS in
5.4s (env cogwheel-newlal). Verdict PASS.

## Independently reproduced numbers (not just green)
- TIED MIRROR (gamma=2, y=(1,0)): 2 images at (-0.25, +/-0.5204), sep=1.041 >> 0.05,
  delta_tau = 0.00e+00 EXACTLY (y->-y mirror across x2 axis), S*est=7.63e-4 < bar=1e-3
  => SERVES. HEAD's `delta_taus>0` leg gave product 0 < 4 => refused. Regression fixed.
- CERTIFICATE SLOPE = -3.000000 exact, strictly monotone decreasing in w_lo. Confirms
  band floor = worst case (ppgo_error_estimate = sum sqrt|mu|*|c3|/w_min**3).
- MERGE (gamma=1.6, rho=1.001): est=1.57e15 (FINITE but astronomical), S*est=3.1e16
  >> bar => refuses via CERTIFICATE; sep=2.07 >> floor so backstop passes. est(w=-1)
  is None (degenerate-floor trigger). Gate & census agree draw-for-draw (live test).
- 6 diagnostic plots + census mirror table written to tests/output/.

## Physics notes
- Gate matches the 2026-08-14 consult rulings verbatim: S=20, bar=1e-3 at w_lo,
  c3-only (no ghost), sep floor 0.05 as defense-in-depth. Correct.
- MINOR (documented in test module, NOT blocking): invariant #2's literal ask
  ("ppgo_error_estimate is None at the physical near-fold") is NOT met physically —
  the DD root finder keeps the merging image just off the exact critical curve, so
  the refusal is driven by a FINITE-but-huge est (S*est=3e16), and the diverging
  quantity is c3 (max|mu| only ~60 here), not mu. The None branch is exercised via
  the genuine w_min<=0 degenerate trigger. Intent (merging pair refuses via the
  certificate, not the separation backstop) is fully satisfied; honest handling.
- Census `_saddle_farfield_analytic` builds real_images = np.asarray(geom.images)
  (INS-1-001 double-mask fix confirmed present at both sites).

Heavy full-sampling / brute-lnL validation is operator-deferred (out of fast scope).
