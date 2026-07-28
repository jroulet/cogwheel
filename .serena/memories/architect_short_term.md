# Architect Short-Term Observations

## Born carrier + band-split (brief_born_carrier_bandsplit, 2026-07-28)
- Wire Born rung, far annulus 3.0<|y|<=4.2426, positive parity, no quadrature.
- `_born.py` currently DORMANT; likelihood.py born slot (~L1650) returns None.
- Fixes: `_born_factors` add a0 (5-tuple), b1 sign fix; correction += a0/q2r;
  born_gate guard A rescale by b1**2 + re-key to w*r0_sq; docstring backwards.
- Band split at w_split keyed on w*r0_sq<~8 (named const, settable). LOW band =
  born carrier alone; HIGH band w>=w_split = EXISTING far-field ppGO+ghost
  machinery (geometric_amplification + farfield_ghost_term / kernel-sum-minus-ghost).
- Census: add 'born' category to classify_fallthrough + _FALLTHROUGH_CATEGORIES.
- OUT: macro-saddle (gamma>1), low-w analytic rung, cusp balls, census RUN,
  TRAIN_TIER shipped artifact (driver-owned). In-build tests FAST only.
- Real gate = residual node counts within ~2x of F023 table on small synth config.

## Professor rulings (2026-07-28, born build)
- CURRENCY = w*Delta_tau (NOT w*r0_sq; in-scope/guard-A bullets saying r0_sq
  are stale errata). Named RHO_END=4 (operator.RHO_END). w_split=RHO_END/Delta_tau.
  Delta_tau = |geom.delays[i]-geom.delays[j]| over the two real_mask channels
  (frame-invariant, no re-solve). Standalone guard: find_images+delay diff.
- b1=-1 exact at point mass. a0 REAL, w-independent, in BOTH born_amplification
  & born_envelope correction. Independent oracle: explicit inv, x0@Ainv@x0;
  a0_oracle=-lam*gamma*cos(2*(atan2(x0_2,x0_1)-beta))/det_a. Agreement tol 2.2e-14.
- GUARD A re-keyed: refuse when w*Delta_tau>=RHO_END. Retire O(w^2) magnitude
  estimate to soft diagnostic. Guard B (parity) unchanged.
- LIVE-SERVE: SHIP function+gate+census+coefficients; DO NOT wire live serve.
  Keep likelihood Born slot fall-through (update status comment). Mirror ppGO.
- Residual test: gamma{0.2,0.25,0.3,0.45}, |y|{3.2,3.8,4.2}, theta{0.1,0.7,1.2},
  beta=0, kappa{0,0.3}, gamma<=0.45. eps=_LOO_STOP 4e-3. HIGH band demod via
  switched_analytic_channels (single carrier inflates theta 7->161). Nodes ~2x:
  low 4-15 log_w+4/y; high 4-8 log_w+4/y.
- Existing test_lensing_born.py accuracy(11.3%)+guardA fixtures MOVE->Test Dev.
