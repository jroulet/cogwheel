# Professor short-term — Build 1c inference review (analytic cusp vertex + y''')

Session 2026-07-29. Reviewed the shipped analytic `_cusp_vertex` + y''' build.
Ran fast domain tests: `test_lensing_airy_fold.py` + `test_lensing_caustic_derivatives.py`
= 88 passed / 7 skipped / 2 xfailed in ~25s (cogwheel-newlal env). VERDICT: PASS.

## Independently verified (not just green checks)
- PRIMARY gate (vertex insensitivity, load-bearing): recomputed by hand — 15/15
  configs serve (100% >> 60% anti-vacuity floor `_VERTEX_MIN_SERVE_FRACTION`),
  worst |dF|/max|F| = 1.03e-2 vs F016 bar `_DEFAULT_ENVELOPE_BAR`=0.05 (margin ~4.8x).
  Deviation scales ~LINEARLY with perturbation (3.1e-5 / 3.1e-4 / 1.0e-2 for dtheta
  1e-4 / 1e-3 / 0.0245) — clean scaling = correct frame mapping (a frame/bracketing
  error would give a perturbation-INDEPENDENT flat swing). Constants match spec:
  angle_tol 1e-10, speed_ratio 1e-8, perturbs {±0.0245,±1e-3,±1e-4}, w{20,40,80}.
- O(1) cost: 11-12 geometry calls per `_cusp_vertex` (< `_MAX_GEOMETRY_CALLS`=20),
  never the retired ~258-pt scan. Verified via call-counter on all 7 direct configs.
- Saddle wedge-edge refusal: new finder returns None at diverging wedge edge, old
  scan serves finite-meaningless — carve-out documented & tested.
- y''' STAGE-2: live worst abs 9.507e-12 (gamma=0.99 near parity wall, |exp|=88),
  worst rel 1.302e-13 over 128 rel-dominated pts. Floors ATOL_3=1e-10/RTOL_3=1e-9
  WIN (3x measured < floors) → gate is spec floor, clears by ~10x/~2500x. dps 40→60
  residual dps-INVARIANT ⇒ closed form correct (float64-limited, not oracle-limited).
  Self-falsification: 1e-6 component corruption goes red vs rtol 1e-9 (teeth). AST
  oracle-independence guard extended to forbid caustic_third_derivative/_caustic_cascade.

## Mild observation (NOT a concern)
Config set drops brief's gamma=0.05 (never serves — caustic too weak for finite-w
uniform form) and gamma∈{1.02,0.9} saddles (serve only marginally); saddle frozen at
gamma=1.3. Documented measure-first deviation (test-dev "never anchor on un-measured
brief coords"); still spans both parities, beta{0,0.37,1.1}, kappa{0,0.3}. Slight
saddle-coverage reduction only. Primary-gate margin at coarsest perturb is modest
(4.8x) — watch if serve-shell later tightens. Heavy full-sampling validation is
operator-deferred (out of turn budget).
