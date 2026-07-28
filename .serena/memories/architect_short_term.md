# Architect Short-Term Observations

## Saddle Born carrier build (brief_saddle_born_carrier, 2026-07-28)
Twin of positive-parity 31ee133; ADD A BRANCH, not a mechanism. 3 Coder WPs
(all lean per Simplifier), tests->Test Dev.
- WP1 _born.py: _born_factors sqrt_mu=1/sqrt(ABS(det_a)) (byte-id positive);
  born_lead_carrier applies morse=(-1j)**n_macro (n_macro=1 iff det_a<0),
  EXACT literal -1j not cmath.exp (Prof: 6e-17 real part breaks |F| pin #2).
  det_a=(1-kappa)^2-gamma^2 beta-independent. New helper saddle_caustic_max_y
  (F026 closed form, max of off-axis u_c & on-axis cand, sqrt(lam) reduction).
  born_gate: guard B -> abs(gamma_p-1)<=DELTA (two-sided wall), gamma_p<1 keeps
  EXISTING positive fence raise verbatim, gamma_p>1 saddle fence via helper;
  guard A shared unchanged (find_images returns exactly 2, index thm). Diagnostics
  born_amplification/born_envelope: add fail-loud guard radicand<=0 (abs() removed
  the implicit math.sqrt ValueError).
- WP2 channels.born_carrier_from_partition: saddle iff det_a<0; below split
  unchanged (born_lead_carrier auto-Morse); above split REFUSE ghost (don't call
  farfield_ghost_term) -> FARFIELD_KERNEL_SUM + zero envelope = pure 2-image ppGO.
  Explicit refusal comment naming underived exp(-0.5j*pi) branch ref.
- WP3 surrogate_census.classify_fallthrough: saddle arm det_a<0 & saddle_caustic_max_y
  <3.0 & annulus -> 'born'. Prof tols: #1 1e-13 rel, #2 1e-12 rel (phase NOT w-flat,
  only |F| + const -i pinned), #3 1e-10 off-axis branch, #4 gamma1.2/th0.3/y3.05 dtau35.3,
  #5 gamma1.6/y4.243/w5 N4->14 eps4e-3, #6 eps4e-3 azimuthal+radial, #7 gamma1.2/y3.5.
  #8 driver byte-diff MUST include a positive cfg through new diagnostic guard.


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
