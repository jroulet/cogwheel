# Professor Short-Term Observations (2026-08-14 session)

## D2 tube-fold build review (test_lensing_tube_d2_fold.py) — VERDICT PASS
Ran `cogwheel/tests/test_lensing_tube_d2_fold.py` (env cogwheel-newlal): **30/30 pass in ~5s.**
Covers all six specs + a self-falsification class per spec (all have teeth).

Physics checks I confirmed by hand:
- `_fold_caustic_theta` (surrogate.py:2823): `if y1_eig<0: t=pi-t; if y2_eig<0: t=-t`.
  Reproduces the closed form mod 2pi for all four octants — (-,-) gives -(pi-t)=t-pi≡pi+t. Correct.
- PRIMARY D2 equality pin: spec demanded BIT-EXACT across ALL four octants; build honestly
  delivers bit-exact ONLY for the two negation-only octants (+,+)/(+,-) (float negation is
  exact in IEEE-754) and a near-machine bound (rtol 1e-9, atol 1e-11) for the two pi-reflection
  octants (-,+)/(-,-), because `math.pi - theta` rounds by <=1 ULP (pi irrational in float64).
  This is the MATHEMATICALLY CORRECT treatment, thoroughly documented; bound is ~1e10 tighter
  than a sign-bug's O(0.1) divergence, so the pin keeps full teeth (self-falsification drops the
  s2 branch and the y2<0 octants break). Deviation from literal spec is justified, not a defect.
- F079 half-ring closure test asserts the premise (physical angle OUTSIDE charted arc) before
  asserting served=True+finite; self-falsification (identity fold) reopens the astroid hole. Solid.
- Arc selection returns exactly 1 fundamental arc bracketing pi/4 (pi/2 is a cusp, not interior);
  saddle arcs returned unchanged; training 4x reduction (detect 4, train 1) pinned + falsified.
- Census fold-consistency: route-equality across 4 sign images, serve-fraction folded==unfolded,
  c3-admission D2-equivariant — all pass, non-vacuous (anti-vacuity `_count` guard in base class).

Could NOT view PNGs (no image tool in this toolset) but every plotting test also asserts the
underlying invariant numerically (no-gap map asserts all_served; scatter coincidence enforced by
the equality tests), so plot content is validated by assertions.

NOT run (budget): heavier neighbor suites test_lensing_surrogate*/training/census timed out >240s
as a batch — those are the operator's out-of-band gate. Modified files touched by build:
surrogate.py, surrogate_census.py, surrogate_training.py.

(Prior tube-chart D2 consultation findings retained in professor_code_observations.)
