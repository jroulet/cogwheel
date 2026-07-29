# Professor short-term (2026-07-29)

## Review: caustic-derivatives 3 new gates (test_lensing_caustic_derivatives.py)
Verdict PASS. Ran fast suite (20 tests, 4.7s, cogwheel-newlal env).
- STAGE-1 curve pin: oracle y=p*r*T vs shipping critical_point.source over 68
  real F038 cases; worst rel err 5.144e-15 (<< 1e-13 gate; spec claimed 5.14e-15).
  Closes the lam*u hole — derivative gate alone can't see a consistently-
  differentiated WRONG curve.
- Positive-parity branch=-1: bit-for-bit == branch=+1, all finite, no
  RuntimeWarning (promoted to error). Correct: critical_point uses only +root
  at |g|<1-k; cascade mirrors it, no sqrt(neg).
- fold_opening_direction: n+ - n- = +2 at all 18 pts (n+=4 inside two-image
  side, n-=2), |d| unit to 1e-16, invariant under soft_axis sign flip.
  (0.9,0.3) macro saddle correctly filtered. Flat +2 = right convention
  (fold A2 creates one image pair).
Self-falsification gates + RuntimeWarning positive control all fire red.
Heavy full-sampling validation is operator-deferred (out of turn budget); not
needed for these geometry gates.
