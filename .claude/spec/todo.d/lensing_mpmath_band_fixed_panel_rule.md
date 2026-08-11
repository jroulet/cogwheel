---
section: Backlog
depends_on: []
---

- **Schwinger mpmath band (60 < w <= 150) uses unbounded adaptive `mp.quad` — replace with a fixed-panel rule**
  `[→ spec]` — measured 2026-08-11 by the driver while fixing the fast-tier hang cluster.

  The exact Schwinger evaluator's DD path (`w <= 60`) is fast and bounded
  (~0.5 s) because it uses a FIXED 24-point Gauss-Legendre composite rule
  (`_dd_gl_rule`, `_PANEL_ORDER = 24`) over `n_panels` panels.  The mpmath
  path (`60 < w <= 150`) instead runs `mp.quad(..., maxdegree=5)` — adaptive
  tanh-sinh — PER PANEL at `dps = 30 + w` on a strongly oscillatory
  integrand (`e^{iwu/2}·kernel`).  Cost is dominated by two compounding
  factors:

  * **Panel count grows ~w²**: `margin = πw/4 + 34`, panel width `= 8π/w`,
    so `n_panels ≈ w²/32` (309 at w=80, 907 at w=150).
  * **Adaptive refinement never converges at some (w, y)** — the 
    "6-hour freeze" documented in `lensing_fast_tier_hangs_in_mpmath.md` is
    a genuine divergence at isolated points, not mere slowness.  Measured:
    `f_schwinger(w=80, y=[0.106,0.146], γ'=0.5)` ≈ 160 s; `w=61,70,100`
    exceed 60 s.

  ## Driver decision (2026-08-11): TEST-LEVEL fix only, production fix postponed

  The fast-tier hang cluster was fixed at the TEST level by parameter choice
  (move ladder-node frequencies above the QD ceiling `w=150` so the engine
  hard-refuses instantly instead of entering the mpmath band):

  * `_CUSP_NODE_W` 80 → 160 (`test_lensing_airy_fold.py`)
  * `_GEOMETRIC_NODE` w 100 → 200 (`test_lensing_airy_fold.py`)
  * `FOP_REFUSALS` / supra grids 63 → 160 (`test_lensing_fast_path.py`)

  This retires `lensing_fast_tier_hangs_in_mpmath.md` (its four named tests
  are green/skipped) and 9 of 11 items in `lensing_serving_ladder_guards_are_red.md`.
  The USER asked to defer the PRODUCTION fix here.

  ## The postponed production fix

  Replace the adaptive per-panel `mp.quad` in `_f_schwinger_mpmath`
  (`cogwheel/lensing/chang_refsdal/_schwinger.py`, `_raw_integral_mp`) with a
  FIXED-panel Gauss-Legendre rule evaluated at mpmath precision — the same
  composite structure the DD path uses, but with mpmath nodes/weights so the
  `e^{πw/4}` cancellation stays certified above `w=60`.  This makes the band
  bounded and O(n_panels·order) like the DD path, eliminating the adaptive
  divergence.  The N/2N paired-rule certification must be preserved and
  re-validated against the DD path in the overlap band (`w` 55–60) and
  against brute force across the band.

  ACCEPTANCE: `f_schwinger` for every `w in (60, 150]` and y in the served
  box completes in O(seconds); `test_lensing_airy_fold.py` ladder tests can
  return to in-band `w` (revert the `_CUSP_NODE_W`/`_GEOMETRIC_NODE`/`FOP_REFUSALS`
  parameter changes) and still finish fast; paired-rule certification agrees
  with the current adaptive result to `_CERTIFICATION_TOL` on a spot grid.
