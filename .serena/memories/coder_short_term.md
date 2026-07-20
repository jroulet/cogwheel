# Coder Short-Term Observations

- WP3 (Build 7b): widened UniformReducedShearPrior.range_dic gamma
  (0.0,0.45)->(0.0,1.6) in lensing/prior.py — ONE uniform range spans
  positive parity (gamma<1) AND macro saddle (gamma>1); NO discrete
  parity label, NO second sub-prior (parity is deterministic fn of
  gamma), transform stays pure IdentityTransformMixin+unit Jacobian so
  CombinedPrior MRO/round-trips untouched. Rewrote the class docstring:
  dropped obsolete 0.45-headroom rationale -> post-7a certified-or-named-
  refuse strong-shear + saddle-branch-for-gamma>1 + gamma=1 det-A=0 is
  measure-zero named refusal by geometry.macro_matrix -> -inf at
  posterior net (no prior special-casing). Added deltoid-fold-validity
  note to UniformSourcePositionPrior docstring: ['u1','u2'] astroid
  quadrant fold stays valid on the 3-cusp deltoid caustic (reflection
  symmetry of Fermat potential is parity-blind). UNCHANGED: _Y_SCALE=307,
  _Y_SCALE_CAP=3.0, _LN_M_LENS_RANGE, folded_reflected_params=['u1','u2'],
  transforms, all other priors. Verified: ast.parse OK; import of
  LensedIASPrior + LensedMarginalizedExtrinsicIASPrior succeeds (runs
  check_inheritance_order at class-def time) with gamma range (0.0,1.6).
  Docstring-only for source prior; range+docstring for shear prior.

- WP4 (Build 7b): gamma'-keyed LOO stop in lensing/likelihood.py.
  Split _LOO_STOP -> _LOO_STOP_FAST=4e-3 (unchanged) + _LOO_STOP_STRONG=
  1e-3 + _STRONG_SHEAR_STOP_THRESHOLD=0.5 (threshold on gamma'=
  gamma/(1-kappa)). Added module-level PURE helper _loo_stop_for_lens(
  lens) next to _leave_one_out_errors: returns STRONG iff gamma'>=0.5
  else FAST; reads only lens['gamma']/lens['kappa'] (both always in the
  lens dict, _LENS_PARAMS includes kappa; engine calls at 998/1777
  already read lens['kappa']). Wired ONCE in _refine_envelope_grid
  (the single stop site shared by direct _envelope_loo_nodes AND ratio
  _ratio_loo_nodes, both pass `lens`) -> computed `loo_stop` before the
  while-loop, replaces bare _LOO_STOP in the break test. _LOO_MAX_NODES=
  48 and _ENVELOPE_SCALE_FLOOR=1e-12 untouched; NO min-|F| seed (stays
  documented escalation). Updated 3 docstrings + constant comments to
  the gamma'-keyed rationale. crown gamma'=0.20<0.5 -> FAST byte-
  identical node count/warm cost; CANCELLATION_CONFIG gamma=0.405,
  kappa=0.57 -> gamma'=0.94>=0.5 -> STRONG (abs(gamma)=0.405 would
  wrongly stay FAST — key on gamma' is load-bearing). Stop is pure fn of
  lens only -> memoized-fiducial contract preserved. ast.parse OK.
  UNVERIFIED (downstream/Test Dev, not Coder): runtime that FAST region
  node count is bit-identical to HEAD + FewMsTimingTestCase passes, and
  that the rescued-node gate (<0.1 nats direct AND ratio vs brute on the
  3-config set incl gamma'=0.94) passes under 1e-3 — reasoned correct by
  inspection (FAST branch returns exactly 4e-3 so <0.5 gamma' path is
  byte-identical; STRONG only tightens, bounded by 48-node ceiling).
  OWED to Test Dev: rescued-node envelope gate (item 4 of build7b_brief)
  + its paired under-seeded falsification. If the gate still >0.1 nats,
  escalation is min-|F| seed into the LOO seed grid (pure fn of
  candidate) — NOT implemented here.

- WP1 (Build 7b): lifted the interim positive-parity guard at top of
  ChangRefsdalChannels.evaluate (channels.py). DELETED only the
  `if not 1.0 - kappa > abs(gamma): raise LensDomainError(...)` block +
  its Build-7 comment; guard sat BEFORE any parity-dependent branch so
  positive-parity codepaths byte-identical. Verified downstream is
  parity-blind: macro_matrix (geometry.py:146) now accepts BOTH parities
  (raises named LensDomainError only for lam<=0 Type III and
  |gamma|==1-kappa det A=0 boundary); find_images = pure quartic alias;
  _channel_switch keys on delay separations only. Rewrote evaluate's
  Raises docstring to the closed reachable vocab: LensDomainError
  (macro_matrix 2 refusals + census/fold guards), operator.Cancellation
  Error (F005), SchwingerCertificationError (F013 ceiling). ALSO fixed a
  stale positive-parity Raises docstring in real_image_delays (same file,
  ~line 607) that falsely claimed macro_matrix raises on
  `1-kappa<=|gamma|` — corrected to the two named refusals + census, both
  parities return normally. No operator/_schwinger/geometry edits; no
  threshold/constant/switch-scale/LOO changes. ast.parse OK.
  UNVERIFIED (downstream to verify, not Coder): runtime that a resolved
  2-image saddle (gamma'=1.3, y=(0.4,0.3)) now returns finite complex F,
  and byte-identity of positive-parity outputs — reasoned correct by
  inspection (guard was pre-dispatch, additive removal only).
  OWED to Test Dev: the channels-layer saddle-guard pin test (currently
  asserts evaluate raises on saddles) must flip to construction-succeeds
  + certified-or-named-refuse per build7b_brief.

- WP2 (Build 7a): operator.py cross-parity Schwinger fallback.
  Added private `_positive_parity_grid_with_fallback(w_array,y,gamma,*,
  beta,kappa,max_order)` next to _grid_certified: FIRST try batched
  _grid_certified (all-certified hot path returns 5-tuple UNCHANGED,
  byte-identical); ONLY on CancellationError fall to per-node loop —
  retry single-element _grid_certified, on its CancellationError if
  w<=W_CEILING_SCHWINGER reconstruct via _mass_sheet_map (pos-parity,
  succeeds since caller guards lam>|gamma|) + f_schwinger, exactly
  mirroring _saddle_grid's wave prefactor; if w>60 re-raise; let
  SchwingerCertificationError propagate. Fallback-node diagnostics =
  zeros/converged=True (saddle-arm convention). Wired F_op_grid & F_op
  pos-parity arms to helper (saddle arm untouched). _schwinger.py:
  relaxed _validate_inputs guard `gamma_prime>1.0`->`>0.0` + rewrote
  msg; updated f_schwinger docstring param + ValueError lines
  (gamma'<=1 -> gamma'<=0). Verified ast.parse OK both files.
  UNVERIFIED (sandbox std::bad_alloc on heavy numba/lal import):
  runtime bit-freeze + fallback value + w>60 refusal — reasoned
  correct by inspection (per-node single-elem _grid_certified is
  bit-identical to batched per-node arithmetic).
- WP1 (Build 7a): added `_check_image_census(images, matrix)` next to
  `morse_index` in geometry.py; called once at end of
  `find_images_quartic` after sort. Refuses (LensDomainError, 'Image
  census defect') when `sum((-1)**morse_index) != sign(det A) - 1`.
  No tolerance band, no count check (Professor: signed sum is complete;
  tr(Hess)=2*lam>0 forbids maxima). Verified: pos-parity 2/4-img
  signed=0 (det>0), saddle signed=-2 (det<0), no spurious raise;
  dropped-pair raises. Env note: numpy C-ext fails when python launched
  from source-tree cwd — run with cwd=/home/tejaswi to smoke-test.
