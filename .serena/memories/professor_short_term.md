# Professor short-term (Build 8e uniform-asymptotics REVIEW, 2026-07-21)

Reviewed Build 8e (uncommitted on HEAD 8d): Airy fold arm
(`chang_refsdal/_airy_fold.py`) + Pearcey cusp arm (`_pearcey_cusp.py`),
tests in `cogwheel/tests/test_lensing_airy_fold.py` (2611 lines).
Env: `cogwheel-newlal` (py3.10). Ran FAST tier (no COGWHEEL_BRUTE_ACCURACY):
**48 passed, 7 skipped (EXACT-HEAVY, correctly gated), 1 xfail, 17.9s.**

## Physics verified (non-vacuous; anti-vacuity tearDown guard active)
- Airy sign convention = Ai(-xi) (my Q5 falsifier (i)): present side (xi>0)
  oscillates >=4 maxima, absent side (xi<0) monotone evanescent decay,
  handoff at caustic xi=0 (tail<peak). CONFIRMS the +iw tau carrier ruling.
- At-caustic finiteness (my divergent-mu flag): value at xi=0 finite & =
  mpmath 2 sqrt(pi) e^{i(...)}(p w^{1/6}Ai(0) - i q w^{-1/6}Ai'(0)); |F| a
  finite peak at first fringe xi<2, NOT a pole; served amplitude p from
  fold curvature (hard_eigenvalue + soft-axis cubic) stays bounded while
  raw sqrt|mu| diverges. The normalization trap is guarded AND self-falsified.
- Sum/diff swap + sqrt|mu| amplitude are caught by self-falsification gates.
- Pearcey primitive: P(0,0)=Gamma(1/4)/2 e^{iπ/8} exact; matches scipy
  QUADPACK ref; paired-rule certificate refuses gross under-resolution.
  Semicubical 27y^2=-8x^3; correct (x~w^1/2, y~w^3/4) exponents w-invariant.
- Fall-through F010: corrupted cert / NaN primitive -> named refusal, served
  bit-identical to cusp arm; threshold moves flip serve<->refuse (not dead).
- Ladder determinism + fixed priority; certified paths byte-identical
  (max|diff|=0.0), only previously-refusing nodes change; L_MAX==48 pinned,
  select_branch frozen.

## Deferrals (both legitimate, noted not failed)
- EXACT-HEAVY accuracy SCANS (far-field envelope convergence + xi^-3/2 decay,
  mpmath oracle cert, scaling-exponent fits, cross-arm envelope <=1e-3) are
  gated behind COGWHEEL_BRUTE_ACCURACY = operator's out-of-band sweep.
- Extended CENSUS (WP1: Wilson intervals, fold/cusp arg CDFs, (a)-(d)
  fractions) is NOT part of this uniform-arms build. Honest @expectedFailure
  tripwire (xfail) + purity AST gate + threshold pins all pass; tripwire
  flips RED when the extended API lands. This is honest scoping, not a gap.

Verdict: PASS. Diagnostic PNGs emitted (sign_convention_handoff,
at_caustic_finite_peak, pearcey_route_vs_threshold, pearcey_control_scaling,
pearcey_certificate_vs_true_error, serving_ladder_byte_identity_diffs).
