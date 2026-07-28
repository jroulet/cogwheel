# Architect Short-Term Observations

## Born band-split RE-ISSUE (2026-07-28, F025 supersedes earlier brief)
F025 OVERRIDES the older "a0 in carrier" plan below. SERVE CARRIER = LEAD-ONLY
`sqrt(mu_macro)*exp(1j*w*phi_geo)`; a0 NOT in serve path (violates F009). a0/b1
stay in module as physics+diagnostic. Prof confirmed (this session):
- LIVE-SERVE STILL NOT WIRED: ship functions+gate+census+coefficients; likelihood
  slot stays fall-through (residual chart is DRIVER TRAIN_TIER artifact, not built).
  Update stale 8h-c1 status comment (b1-placeholder rationale now wrong).
- b1/a0 brief forms == F023 matrix forms; test vs INDEP matrix solve, tol 2.2e-14;
  point mass gives b1=-1, a0=0. Invariant b1-a0 == -lam^2*mu_macro (cheap check).
- Band-split ASSEMBLER lives in channels.py (has geometric_amplification import,
  farfield_ghost_term, switched_analytic_channels); _born.py stays pure scalar +
  lead-only entry point. Delta_tau = diff of TWO REAL images' FULL Fermat delays
  (geometry.delay, incl -ln|x|), NOT phi_geo. Split w moves per-config; enforce at
  worst case (largest Delta_tau). Above-split residual demod via
  switched_analytic_channels (single-carrier inflates theta counts).
- Guard A re-key: refuse w*Delta_tau>=RHO_END (=4, reuse SACR-C const); EPS_BORN
  retires as accuracy bar (keep guard B parity margin).
- 'born' census category: insert 2nd slot after 'dropped-sliver', before
  'cusp-window'. Predicate: detA>0 AND gamma<0.75 AND 3.0<|y|<=4.2426.
- Acceptance #7 (ghost raises still serves) REQUIRES above-split path -> keep both
  bands in assembler despite Simplifier trim. Ghost-tolerance test = F023 config
  (kappa=0.3,beta=0.5 NON-prod witness) or monkeypatch.
- Node-count acceptance #5: only LOW band [1e-3,0.05] fast-tier (N<=10, azimuthal
  AND radial); higher bands driver-post-build. Primary reachable-red for a0 = max|
  residual| ratio lead-only < a0 (>=5x at gamma=0.45), not a full node count.

## (STALE below - superseded by F025) Born carrier + band-split

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
