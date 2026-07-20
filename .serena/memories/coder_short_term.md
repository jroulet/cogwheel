# Coder Short-Term Observations

- WP3 (Build 8a): wired the amplification surrogate into the lensed
  likelihood dispatch. likelihood.py LensedRelativeBinningLikelihood:
  added ctor kwarg amplification_surrogate=None (stored on self); added
  module const _SURROGATE_CAUSTIC_FLOOR=0.05; SINGLE fast-path intercept
  at TOP of _amplification_coefficients (before lens=self._lens_params):
  if surrogate is not None -> served=_surrogate_coefficients(par_dic);
  return served if not None, else fall through to UNCHANGED exact path
  (seed eval / fiducial cache / ratio / LOO / direct). Documented as
  superseding the two-intercept in-place swap (full envelope emulator
  short-circuits the whole per-candidate cost). New helper
  _surrogate_coefficients: computes lens+dense_w (returns None on
  non-positive w so exact path raises LensedBinningError); returns None
  if not surrogate.in_domain(gamma,y1,y2,beta); builds geom=
  ChangRefsdalChannels(dense_w).geometry_partition(gamma,y=(y1,y2),beta,
  kappa) (WP2 — LensDomainError propagates UNSWALLOWED, refusal symmetry
  preserved); returns None if real_mask.sum()!=region count OR
  caustic_distance<0.05; queries envelope(dense_w,...)->(E,served),
  None if not served; kernels,_=reconstruct_from_envelope(dense_w,E,
  geom.delays,geom.saddle_kernels,geom.switch,geom.critical_delay);
  k0,k1=self._reduce_dense_kernels(kernels); delays=self._image_delays(
  lens,geom); returns (delays,k0,k1,geom) SAME 4-tuple shape as exact
  path (consumers ignore 4th elem; _image_delays reads only .delays,
  present on ChangRefsdalGeometryPartition). Helper
  _surrogate_region_image_count: WP1 surrogate exposes NO image-count
  attr and surrogate.py is out-of-WP3-scope, so region count computed
  LAZILY once from box-center geometry_partition (gamma/y1/y2 grid
  midpoints, beta=0,kappa=0, w=[1,2] since count is w-independent),
  cached in self._surrogate_region_nimg (derived cache). PICKLE:
  __getstate__ now pops _fid_cache AND _surrogate_region_nimg;
  amplification_surrogate rides along in __dict__.copy() (preserved for
  workers, small flat ndarrays); __setstate__ resets both caches. JSON:
  get_init_dict override pops amplification_surrogate key when None
  (byte-identical to HEAD) else raises NotImplementedError (fitted-
  surrogate JSON deferred post-build; sampling out of scope).
  marginalized_likelihood.py LensedMarginalizedExtrinsicLikelihood:
  added amplification_surrogate=None kwarg + docstring; stored on self
  BEFORE super().__init__ (JSONMixin name-match, next to delta_t_max/
  bin_delay_tol/kernel_subsamples); threaded into the internal RB
  _engine construction in _set_summary so the marginalized path inherits
  the fast path via self._engine._amplification_coefficients; added
  parallel get_init_dict override (pop-None/raise-non-None). Verified:
  ast.parse OK both files; import OK; default None on both ctors; both
  get_init_dict overrides present; _surrogate_* helpers present; API
  signatures cross-checked vs exact path (in_domain(gamma,y1,y2,beta),
  envelope(w,gamma,y1,y2,beta), reconstruct_from_envelope arg order ==
  _kernels_from_dense_envelope, _reduce_dense_kernels/(n_dense,n_chan),
  _image_delays(lens,partition) reads .delays only). NO edits to
  surrogate.py / channels.py / engine / _data_term / _norm_term /
  reconstruct logic.
  UNVERIFIED (downstream/Test Dev, not Coder — no full-instance runtime
  in sandbox): (a) amplification_surrogate=None gives byte-identical
  lnlike + JSON round-trip vs HEAD across configs; (b) served path
  returns (delays,k0,k1,geom) matching exact-path k0/k1 within the
  envelope gate eps<1e-3 on held-out in-domain configs both parities and
  never invokes _exact_total/LOO/fiducial cache; (c) refusal-set
  preservation (engine-refused -> LensDomainError propagates through the
  surrogate path; F010 falsification the guard can go red); (d)
  marginalized inherits the fast path through _engine. Reasoned correct
  by inspection: intercept is additive/pre-lens; helper returns None on
  every guard miss so the exact path is untouched when None or on any
  miss; LensDomainError never caught in _surrogate_coefficients.
  OWED to Test Dev (Build 8a acceptance): surrogate-served-vs-exact k0/k1
  gate on held-out configs; a None-surrogate byte-identity pin (lnlike +
  __getstate__ + JSON) vs HEAD; refusal-preservation + F010 red-gate; a
  caustic-floor / image-count-mismatch fall-through test; marginalized
  _engine fast-path inheritance smoke.

- WP1 (Build 8a): NEW module cogwheel/lensing/surrogate.py exposing
  public LensAmplificationSurrogate — tensor-cubic-spline emulator of the
  SACR-C envelope E(w) over 4-D box (ln w, gamma, y1_eig, y2_eig) at
  beta=0/kappa=0. Imports ONLY chang_refsdal (ChangRefsdalChannels +
  geometry.LensDomainError + operator.CancellationError +
  _schwinger.SchwingerCertificationError) + numpy/scipy; NO likelihood
  import (circular-safe; verified import). from_engine: fresh
  ChangRefsdalChannels(w_grid) per param point (deterministic initial
  labeling), .evaluate(gamma,y=(y1e,y2e),beta=0,kappa=0), label=
  partition.envelope (shape (n_w,) complex); real+imag interpolated
  SEPARATELY via RegularGridInterpolator(method='cubic',
  bounds_error=False, fill_value=nan) (never mag/phase). Refusals: try/
  except the 3 named errors AT ANY w -> whole param point into
  refused_points (n,3) eigenframe (gamma,y1e,y2e); also treats non-finite
  envelope as refusal (F005-conservative). in_domain: R(-beta) rotate
  source to eigenframe (y1e=cos b*y1+sin b*y2, y2e=-sin b*y1+cos b*y2 =
  exp(-i beta)(y1+iy2)) THEN axis-aligned box containment AND outside
  exclusion ball (normalized Euclidean radius 1.0 = one param-grid
  spacing) of every refused point — pure geometric gate, no learned mask.
  envelope(w_array,gamma,y1,y2,beta)->(E,served): in_domain gate + w-band
  guard (served=False if any w outside trained [w_min,w_max]); served eval
  builds (N,4) query pts [ln w, gamma, y1e, y2e]. save/load single npz
  (grids, real/imag arrays, refused_points, provenance JSON string;
  allow_pickle=False). Pickle: __getstate__ drops interp cache /
  __setstate__ rebuilds -> flat-ndarray state. Provenance minimal: box
  bounds, resolution, sha1[:12] training hash.
  ROTATION SIGN VERIFIED by code: macro_matrix=(1-kappa)I-gamma*Q(beta),
  Q(beta)=R(beta)Q(0)R(beta)^T => system (A(beta),y) maps to
  (A(0),R(-beta)y); so query rotate by R(-beta) into beta=0 box. VERIFIED
  runtime (cwd=cogwheel-claude-dev, scipy 1.11.4 cubic OK): ast.parse OK;
  train tiny box gamma(.1,.3) y1(.6,1.2) y2(.2,.8) w(1,30) 4x4x4 grids,
  0 refused; envelope served finite; scalar-w -> 0-d out; BETA-INVARIANCE
  E(beta,R(beta)*eig)==E(0,eig) allclose TRUE (confirms Professor Q2
  beta-elimination empirically); out-of-domain + w-out-of-band ->
  served=False; npz + pickle round-trips allclose; pickle state all
  ndarray/dict. DID NOT modify lensing/__init__.py (no wiring per WP1
  scope; import via cogwheel.lensing.surrogate). No engine edits.
  OWED to Test Dev (Build 8a acceptance): (a) surrogate-vs-engine envelope
  gate eps<1e-3 on held-out configs both parities; (b) a box that
  CAPTURES refusals so in_domain exclusion-ball path is exercised (my
  clean 2-image box had 0 refused — refusal capture verified only
  structurally); (c) refusal-set preservation (engine-refused config ->
  served=False through surrogate) + F010 falsification the gate can red;
  (d) reflection-symmetry validation (u1/u2) is a TEST, not used to shrink
  training. LIMITATION noted: refused nodes filled 0.0 (finite so cubic
  builds); cubic stencil of a served query one cell beyond the exclusion
  ball can still touch a 0-filled node — MVP-acceptable, gate is the
  exclusion ball; flag if a held-out config near a refusal boundary fails.

- WP2 (Build 8a): additive geometry-only partition on channels.py.
  Added frozen dataclass ChangRefsdalGeometryPartition (fields: w,
  delays, saddle_kernels, switch, critical_delay, real_mask,
  caustic_distance) + NEW method ChangRefsdalChannels.geometry_partition
  (*,gamma,y,beta=0,kappa=0) that DUPLICATES verbatim the ~13 cheap
  geometry lines evaluate runs BEFORE _exact_total (macro_matrix ->
  nearest_caustic_point -> find_images -> per-image geometry.delay ->
  _assign_labels(+self._markers update) -> _labeled_delays ->
  _physical_kernels -> _channel_switch) and STOPS before _exact_total /
  switched_analytic_channels. Container carries exactly what
  reconstruct_from_envelope consumes (delays,saddle_kernels,switch,
  critical_delay) so a surrogate envelope can rebuild channels, plus
  caustic_distance for the likelihood's in-domain check + real_mask.
  Chose a dedicated lightweight dataclass over reusing ChangRefsdal
  Partition-with-Nones (ChangRefsdalPartition's np.ndarray fields +
  reconstructed/envelope_reconstruction properties would break on None).
  evaluate BYTE-FOR-BYTE UNCHANGED (git: only 2 removed lines = the
  __all__ rewrite adding ChangRefsdalGeometryPartition; 140 insertions).
  geometry_partition mirrors evaluate's stateful label continuation
  (updates self._markers) so same-continuation-state geometry ==
  evaluate's exactly. macro_matrix runs FIRST so LensDomainError (Type
  III 1-kappa<=0, det A=0 parity boundary, census, fold-degenerate
  metric) still fires at the API boundary; only the expensive total's
  CancellationError/SchwingerCertificationError are skipped (geometry-
  only, by design). VERIFIED (runtime, cwd=/home/tejaswi to dodge
  numpy C-ext source-tree fail): ast.parse OK; import OK; _exact_total
  spy shows 0 calls during geometry_partition; delays/saddle_kernels/
  switch max|diff| vs evaluate = 0.0 and real_mask/critical_delay/
  caustic_distance exact-match on BOTH positive-parity (gamma=0.2,
  kappa=0.1) AND macro-saddle (gamma=0.405,kappa=0.57,gamma'=0.94);
  Type III + parity-boundary raise LensDomainError through
  geometry_partition. No engine-module edits; only channels.py touched.
  OWED to Test Dev: Build 8a surrogate suite should pin (a) geometry_
  partition-vs-evaluate byte-identity of the shared geometry across a
  config sweep both parities, (b) a spy asserting _exact_total is never
  called, (c) refusal-set preservation (macro refusals raise), (d) a
  round-trip feeding a known envelope through reconstruct_from_envelope
  with this container's fields reproduces evaluate's F.


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
