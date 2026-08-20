## diffractive_certificate_fit_interior_fix (INS-3-001) — Phase 1 rulings

Consulted the Architect on the deep-interior calibration build. Key physics
established by reading `_diffractive.py`, `_schwinger.py` (W_CEILING_SCHWINGER=60,
the DD `e^{pi w/4}` cancellation hard ceiling; `f_schwinger` refuses >60),
`_hyp1f1.point_mass_g_derivatives` (kernel cancellation law ~ eps·e^(w√s), but
this is NOT the interior-ceiling driver), and the fit script + gated tests.

Rulings given (design-authoritative):
- Q1: KEEP the literal `min(w_fit, _DIFFRACTIVE_FIT_CEILING)` line — it is a hard
  ORACLE-DOMAIN cap (w_low must never exceed 60 because f_schwinger does not exist
  there), not the interior's margin. "Remove the clip-as-conservativeness" = make
  the de-rated fit conservative on its own so the clip is a no-op; the committed
  gated test `test_clip_is_not_the_conservativeness_mechanism` (served<60 AND
  raw(de-rate=1)<60 in the interior) is the SPEC for this. Literal removal would
  let a future regression return w_low>60 and crash the f_schwinger >60 refuse.
- Q2: interior honest ceiling is smooth/monotone (engine-measured 6-34 at
  rho 0.2-0.5), the over-serve is a calibration gap not a representation wall IN
  rho∈[0.2,0.6]. BUT the degree-2 log(s) poly has NO correct s->0 asymptotics:
  log(s)^2 must diverge, so below the sampled floor the raw fit either over-serves
  (clipped to 60) or under-serves. The low interior (6-34) is at FINITE rho; the
  "series exact as s->0" limit (ceiling->60) lives at rho<0.2 (<4% of interior
  prior mass), exactly where the log(s)^2 representation limitation bites. Guarded
  by clip + gated test at rho=0.2 (r≈0.06, below the r=0.1 grid floor) + low prior
  mass. Prefer the conservative (≤0 log(s)^2) outcome if the bake produces it.
- Q3: full grid linspace(0.1,1.3,7) = {0.1,0.3,0.5,0.7,0.9,1.1,1.3} (N=7);
  6×7×32=1344 grid +336 off-grid ≈1692 measured ≈ ~56 min serial (driver step).
  Smoke: interior gamma{0.2,0.3}×r{0.1,0.2} (4 cells=128 rows) + keep (0.5,0.3)
  interior + (0.5,0.9) near-exterior anchors, TRIM the 6-cell smooth block
  (192 rows) to 1-2 exterior anchors; target ~195-227 grid rows (~490-570 s),
  under the 600 s budget. Interior cells must be PAID FOR by trimming smooth
  cells, not added on top (current 259 rows = 653 s already at ceiling).
- Q4: de-rate min(0.85, 1/max_overpred) is the sole interior margin; no
  interior-specific de-rate (breaks mirror-fidelity + would be redundant once the
  grid samples the interior, since the interior is in the FENCED de-rate domain).
- Q5: 1e-5 one-sided (over-serve trips) tolerance is correct: oracle bisection
  width ~3.4e-7 (2^24 log-w steps) + de-rate 6-decimal rounding ~6e-7 → ~30x
  headroom; a round-off guard, not a physics margin.

## INS-3-001 Professor inference-review verdict (2026-08-20, SHA 362c58e)
Fast tier verified. (1) test_deep_interior_served_below_ceiling_at_calibrated_cell
extended to gamma{0.2,0.3,0.5} x theta{0,pi/4,pi/2} x rho{0.2,0.3,0.5}: all 27
engine-free probes served not-None and strictly < 60 (measured 1.9-34.4). (2)
Corner re-pin CORNER_R=1.1 confirmed in _off_grid_points('full',42), rho=2.188
> 1.4; raw_fit/w_low_true=1.006<1.5 (docstring says ~1.01x), raw=1.23 not
clipped, caustic coeff -0.824 negative, drop-caustic inflates to 2.34. (3)
Gated honest-serve skip reason ACCURATE and suite is genuinely red: measured
w_fit/w_true=1.1244 (gamma .5 cusp rho .2) and 1.0233 (rho .3) -- the 0.85
de-rate clamp is insufficient at the cusp-direction deep interior; axis dirs
conservative (0.88x, 0.69x). GAP: FullGridCertificateOracleTestCase docstring
still says "[0.3,1.3]" and "252 on-grid + 240 off-grid" (stale); actual new
grid = 1014 on-grid + 250 off-grid = 1264 rows, and script docstring "~40 min
serial" is stale (~56 min). _CALIBRATION_S_MAX comment WAS updated. Verdict
CONCERN (docs only; no physics/test defect).
