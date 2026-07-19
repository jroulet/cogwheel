# Coder Short-Term Observations

## WP1 verify saddle engine seams + warm cost (Build 6, attempt 6, 2026-07-18)

- VERIFY-ONLY: all 7 seam invariants HELD; only edit = 1-line comment at
  _saddle_grid takeover site noting select_branch is NOT the saddle authority
  in S1 (channels.py/Build 7 wires it) -- the WP's sole permitted edit.
- git diff operator.py: positive-parity path byte-frozen (only ADDITIVE
  leading `if not lam>abs(gamma)` guards in F_op/F_op_grid + docstrings +
  new _saddle_mass_sheet_map/_real_delay_min_separation/_saddle_grid). No
  change to _mass_sheet_map/_grid_certified/_weight_vectors/_contract_grid/
  cancellation_exponent/select_branch/refusal constants.
- _schwinger seams: f_schwinger a=1-gp(<0), b=1+gp; w>60 UNCONDITIONAL hard
  refuse BEFORE compute; N-vs-2N certify on RAW t-integral at
  _CERTIFICATION_TOL=3e-10 BEFORE _reconstruct applies -iw/2*e^{iw|y|^2/2}*
  1/Gamma(iw/2) (phases mod 2pi); SchwingerCertificationError<-RuntimeError;
  12 njit cores all fastmath=False.
- SMOKE (worktree, newlal py): pos F_op(5,[.3,.1],.2)=-0.35753...+1.1664j
  (pin match); saddle F_op(3,[.4,.3],1.3,k=0)==direct f_schwinger bit-exact
  =0.144706+0.406512j, diag=(0,True,0,0); parity boundary (k=.5,g=.5) +
  Type III (k=1.2) both raise LensDomainError.
- WARM COST (diagnostic, prices envelope-surrogate): 24 pts w{5,20,40,55} x
  gp{1.05,1.3,2.0} x y{(.2,.1),(.6,.4)}, repeats=5: mean 149.61 ms/pt,
  min 28.52, max 306.99 (panel count ~pi w/4 -> w=55 drives the max).
- UNVERIFIED (Test Dev/Inspector own): full test battery green-ness; F010
  py_func-chain RED mutation; independent-oracle accuracy across the grid.

## WP3 operator.py saddle parity dispatch (Build 6, 2026-07-18)

- Wired negative-parity (saddle) dispatch into operator.F_op/F_op_grid.
  Positive-parity path BYTE-IDENTICAL to HEAD: loaded HEAD copy via
  importlib (register in sys.modules FIRST so @dataclass OperatorDiagnostics
  resolves) + new module side-by-side over 400 strictly-positive-parity
  configs (gamma<0.98*lam, kappa in [-0.5,0.4], w in [1,25], 3-node grids):
  max|scalar diff|=0.0, max|grid diff|=0.0, refusal-decision match 572/572,
  0 mismatch. Diff is purely ADDITIVE prefixes (284+/14-, the 14 del = F_op
  docstring+return rewrite only, no compute logic).
- Classification gate `if not lam > abs(float(gamma))` EXACTLY mirrors
  _mass_sheet_map's positive-parity gate -> byte-identity guaranteed by
  construction (saddle prefix returns before touching _grid_certified).
- NEW helpers (all in operator.py, none touch byte-frozen _mass_sheet_map /
  cancellation_exponent / select_branch / _grid_certified):
  * _saddle_mass_sheet_map(y,gamma,kappa): lam<=0 -> LensDomainError (Type
    III); lam>=|gamma| -> LensDomainError (parity boundary+pos parity);
    returns (lam, y/sqrt(lam), gamma/lam). DELIBERATE: refuses lam==|gamma|
    as boundary (not WP's literal "0<lam<=|gamma|") for consistency with
    macro_matrix + f_schwinger's gamma_prime>1 requirement -- FLAG for
    Inspector/TestDev.
  * _real_delay_min_separation(source,matrix): min upper-tri pairwise abs
    delay diff over geometry.find_images; 0.0 if <2 images (real-image-only,
    matches channels convention).
  * _saddle_grid(w_array,y,gamma,*,beta,kappa): macro_matrix classify+refuse;
    eigenframe z_eig/y_eig/s computed ONCE (w-indep); delta_min computed ONCE
    only if any w>60. Per node: (w>60 AND w*delta_min>=RHO_END) -> geometric_
    amplification; else f_schwinger + mass_sheet_phase=exp(.5j*w*ln(lam)
    -.5j*w*kappa*s); value = mass_sheet_phase*f_pure/lam.
- Saddle geometric-vs-wave decision made INSIDE _saddle_grid (F_op never
  receives delta_min -- pos-parity channel tracker owns branch selection),
  NOT threaded through select_branch, so pos-parity call byte-unchanged.
  Because f_schwinger hard-refuses w>60, UNRESOLVED saddle at w>60 propagates
  SchwingerCertificationError (verified scalar AND grid).
- Saddle OperatorDiagnostics = order_used=0, converged=True, tail=0.0,
  cancellation_ratio=0.0 (operator-series fields N/A; documented in docstring).
- Smoke VERIFIED: anchor F_op(3,[.4,.3],1.3,kappa=0)=0.14470585550870085+
  0.4065122393352838j (byte-matches brief + direct f_schwinger); grid==scalar;
  Type III (kappa=1.2) + parity boundary (kappa=0.5 g=0.5) -> LensDomainError;
  unresolved w>60 (src=[0.05,0] symmetric on-axis, delta_min=0) ->
  SchwingerCertificationError scalar+grid; cancellation_exponent still saddle-
  refusing (LensDomainError, untouched); kappa=0.5 saddle (lam=0.5,g=0.6) finite.
- UNVERIFIED (no suite run per role): full existing operator/likelihood suite
  green-ness; independent-oracle accuracy of saddle reconstruction across the
  (w,gamma',y) grid (WP2/Test Dev own); geometric-branch agreement at resolved
  w>60 vs Schwinger (research gate 6).

## WP2 verify+complete _schwinger.py saddle wave branch (Build 6, 2026-07-18)

- Delivered _schwinger.py was mathematically COMPLETE; only change = named
  `_CANCEL_SCALE = 0.25*math.pi` (was inlined `0.25*math.pi*w`) to make the
  y-INDEPENDENT cancellation scale (F001-S, L_S=pi w/4) greppable/auditable
  per WP contract; `margin = _CANCEL_SCALE*w + _U_MARGIN_CONST`. NO other
  edits (did NOT rewrite).
- Paper-verified all 7 Professor failure modes: (a) IBP boundary term
  endpoint=+T^s h(T)/s POSITIVE sign (Int u dv=uv-Int v du, dv=t^{s-1}dt,
  v=t^s/s) — the O(1) additive error the N-vs-2N gate can't catch, sign
  correct; (b) dd running sum via dd_complex_add, dd_mul keeps a_hi*b_lo+
  a_lo*b_hi, dd_add QD-compensated; (c) _reconstruct (pref -iw/2, e^{iw|y|^2/2},
  1/Gamma(iw/2), mod-2pi) applied ONCE at end, h'=h*G analytic G(t)=
  amp1/da^2+amp2/db^2-1/2/da-1/2/db; (e) paired N/2N gate 3e-10 on RAW integral
  BEFORE prefactor; (f) w>60 -> SchwingerCertificationError(RuntimeError)
  unconditional; (g) a=1-gamma_prime<0, da_im=-half_w*a>0 matches geometry
  eigenframe (sign flip would conjugate).
- Margin/truncation math checked: both u-ends decay exp (A-integrand ~e^u at
  t->0, B ~e^{-u} at t->inf); +34 slack gives trunc/|I_raw| ~ (w/2t_cap)e^{-34}
  ~6e-16 << 3e-10; trunc is N/2N-IDENTICAL so cancels in gate, bounded
  analytically by margin. Correct.
- INDEPENDENT mpmath oracle (subtract-h(0) regularization — DIFFERENT scheme
  from code's IBP-with-h'; F002-clean) agrees: rel 1.1e-14 (w3,gp1.3),
  1.3e-13 (w10,gp1.05 near boundary), 1.0e-14 (w5,gp1.5,y2<0), 1.65e-10 (w20 —
  mpmath's OWN quad degrading per research Sec12, not code). Confirms value +
  PHASE => sign convention (g) correct; a conjugate would read rel~2. NOTE: a
  naive Int_0^inf t^{s-1}h oracle is ILL-POSED (|t^{s-1}|=t^-1 log-diverges at
  0, only conditionally convergent) — must regularize (subtract-h0 or IBP).
- Smoke: f_schwinger(3,(0.4,0.3),1.3)=0.14470585550870085+0.4065122393352838j
  (matches brief anchor byte-for-byte). py_func reachable on _raw_t_integral_core;
  all 12 njit cores fastmath=False. _CANCEL_SCALE==pi/4 bit-exact.
- WARM COST (diagnostic, non-gating): _measure_warm_cost over w{5,10,20} x
  gp{1.05,1.3} x y{(.2,.1),(.6,.4)} = mean 43.93 ms/point, min 28.52, max 72.49
  (dd quadrature, panel count ~w; w=20 drives the max). Prices envelope-surrogate.
- UNVERIFIED (Test Dev / Inspector own, not authored here): full mpmath dev-oracle
  campaign gate #2 across the whole (w,gp,y)<=60 grid; 2D rotated-contour anchor
  gate #3; certification-refusal RED mutation; WP3 operator.py parity dispatch
  (not my scope, untouched).


## WP1 saddle geometry re-verify (Build 6, attempt 5, 2026-07-18)

- VERIFY-ONLY outcome: delivered geometry.py is complete/coherent vs
  research spec; NO code edits needed. git status: only geometry.py
  modified (+ memories/agent_state); _schwinger.py + scratch untracked
  (WP2/WP3 scope, untouched).
- Gate 7 byte-identity RE-CONFIRMED: loaded HEAD (87dfa4e) copy +
  working copy side-by-side via importlib; max|diff|=0.0 over 600
  strictly-positive-parity configs (gamma<0.98*lam) for macro_matrix,
  critical_point.image, find_images, nearest_caustic_point.image AND
  .distance.
- Census gate 1 (oracle-free, kappa=0 gamma=1.3, evals=[-0.3,2.3]):
  centered find_images=2 imgs both morse=1 (1,1); off-center census
  over 400 sources = exactly {(1,1):378, (0,1,1,1):22}, no other
  counts; two disjoint deltoid lobes (x∈[-1.71,-1.12] and [1.12,1.71]);
  theta_max=0.5*arcsin(lam/|gamma|)=0.4388. Matches research Sec 3.1.
- Domain guards verified: Type III (kappa=1.2) + parity boundary
  (kappa=0.5 g=0.5 float64-exact) raise named LensDomainError on
  macro_matrix/critical_point/nearest_caustic_point; saddle (0.9,0.2)
  and positive parity accepted.
- Parity-agnostic helpers reconfirmed unchanged from HEAD (not in
  diff): hessian (algebraic), morse_index (counts neg evals ->1 for
  saddle), magnification (1/det, neg for saddle), delay (quadratic
  form), image_kernel (abs(mu)+morse phase), _saddle_metric/
  saddle_coefficients (inv of indefinite Hessian). No positive-definite
  assumption anywhere. find_images = pre-existing fixed-tol alias of
  algebraic find_images_quartic -> parity-agnostic, NO redundant
  wrapper added.
- FLAG (unchanged) FOR TEST DEVELOPER: test_lensing_geometry.py:524
  DomainGuardsTestCase::test_macro_matrix_rejects_non_positive_parity
  case (0.9,0.2) is now a SADDLE macro_matrix ACCEPTS -> that subTest
  WILL FAIL; move to saddle-accept. Cases (0.5,0.5) and (0.0,1.0) are
  exact boundaries, still correctly refused. No test edits made.
- UNVERIFIED (no suite run per role): full existing geometry suite pass
  state; WP2/WP3 (Schwinger/operator dispatch); caustic accuracy vs
  independent oracle.

## WP1 Finalize/verify saddle geometry (Build 6, 2026-07-18, re-dispatch)

- Re-verified the uncommitted geometry.py saddle work against research
  Sec 2/3/9. NO code edits needed — delivered work is finalized and
  correct. Only geometry.py modified (git status): operator.py untouched,
  no _schwinger.py (scope fences honored).
- BYTE-IDENTITY reconfirmed vs HEAD (loaded HEAD copy as separate module
  via importlib): max|diff|=0.0 over 400 strictly-positive-parity configs
  (gamma<0.98*lam) for macro_matrix, critical_point.image,
  _centered_source_images, nearest_caustic_point (image+distance). The
  1.0*sqrt==sqrt IEEE exactness + inert disc clamp hold.
- SADDLE self-consistency (oracle-free, kappa=0 gamma=1.3, a=-0.3<0<b=2.3):
  centered find_images -> 2 imgs at [0,±0.6594]=±e_b/sqrt(b) both morse=1;
  two disjoint deltoid lobes (center 0: x∈[0.66,1.83]; center π:
  x∈[-1.83,-0.66]); theta_max=0.5*arcsin(lam/|gamma|)=0.4388; off-center
  census over 200 sources = exactly {(1,1):185, (0,1,1,1):15}, ZERO
  index-sum anomalies (signed parity sum == -2). Matches Sec 3.1.
- macro_matrix domain split VERIFIED: pos-parity + saddle accepted;
  lam<=0 (Type III) and |gamma|==lam (boundary, float64-exact per F004)
  raise LensDomainError.
- FLAG FOR TEST DEVELOPER (unchanged/reconfirmed): existing
  test_lensing_geometry.py::DomainGuardsTestCase::
  test_macro_matrix_rejects_non_positive_parity has cases
  [(0.9,0.2),(0.5,0.5),(0.0,1.0)]. Case (0.9,0.2) is now a SADDLE
  (lam=0.1<0.2) that macro_matrix ACCEPTS -> that subTest WILL FAIL and
  must move to a saddle-accept assertion. The other two are boundary
  equalities and still correctly raise. No test edits made (Coder does
  not author/edit tests).
- UNVERIFIED (no test suite run per role): full existing geometry suite
  pass state; Schwinger evaluator (WP2, not created); numerical caustic
  accuracy vs independent oracle.

## WP1 Parity-aware saddle geometry (Build 6, 2026-07-18)

- `cogwheel/lensing/chang_refsdal/geometry.py` extended to macro saddles
  (negative parity, 0 < lam=1-kappa < |gamma|). Positive parity VERIFIED
  byte-identical: imported HEAD + new modules side-by-side, max abs diff = 0.0
  over 300 random positive-parity configs for macro_matrix, critical_point,
  nearest_caustic_point.
- macro_matrix: domain split. lam<=0 -> LensDomainError (Type III); |gamma|==lam
  (float64-exact, F004 powers-of-two) -> LensDomainError (parity boundary,
  det A=0); saddle (0<lam<|gamma|) ACCEPTED. Return kept `(1.0-kappa)*eye - gamma*shear`
  (not `lam`) for byte-identity.
- _centered_source_images(matrix, *, degeneracy_tolerance): now branches on
  n_positive = #(eigenvalues>0). ==2 positive-parity path byte-identical
  (Einstein-ring guard + original loop); ==1 saddle returns +-e_+/sqrt(lam_+)
  on the positive-eigenvalue axis, census (1,1) both morse=1; ==0 refused.
- critical_point gained `branch: int = 1`. Positive parity uses a SEPARATE
  `if abs(gamma)<lam:` block with the exact frozen expression (byte-identical).
  Saddle block: disc=1-eff_gamma^2 sin^2(2φ); disc<-1e-12 -> refuse (outside
  wedge); clamp max(disc,0); effective_u = eff_gamma*cos(2φ)+branch*sqrt(disc);
  effective_u<=0 -> refuse. Two lobes at φ=theta-beta near 0 and π.
- _caustic_source / _coarse_squared_distances njit helpers gained `branch: float`
  (no numba default args used; positive-parity callers pass 1.0; `1.0*sqrt`==`sqrt`
  IEEE-exact + inert disc clamp -> byte-identical).
- nearest_caustic_point: positive-parity path byte-identical (full-circle grid,
  argsort[:4], branch 1.0). Saddle path scans BOTH wedges (center in beta,beta+π)
  and BOTH branches (+-1) over [center±theta_max], theta_max=0.5*arcsin(lam/|gamma|),
  bounded minimize_scalar refine, global-min over lobes+branches; final
  critical_point call uses winning branch.
- Hessian/delay/morse_index/image_kernel/find_images/quartic solvers UNEDITED
  (git diff: 0 changed lines in each; Fact 4 — parity-agnostic already).
- VERIFIED (runtime smoke, oracle-free): two lobes distinct & non-overlapping
  (lobe0 x in [-0.949,-0.888], lobe1 x in [0.888,0.949]); census over sources
  near lobes gives image counts strictly {2,4}, signed-parity sum (-1)^morse == -2,
  0 anomalies (200 wide + 200 near-lobe sources).
- FLAG FOR TEST DEVELOPER: existing test
  `test_lensing_geometry.py::DomainGuardsTestCase::test_macro_matrix_rejects_non_positive_parity`
  case (kappa=0.9, gamma=0.2) is now a SADDLE (lam=0.1<0.2) that macro_matrix
  ACCEPTS -> that test WILL FAIL and must be updated (move that case to a
  saddle-accept assertion; keep boundary/Type-III cases as refusals).
- UNVERIFIED (no test authored/run per role): full existing geometry suite pass
  state; Schwinger/operator dispatch (other WPs); numerical caustic accuracy vs
  an independent oracle.

## WP2 LensedMarginalizedExtrinsicIASPrior (Build 5, 2026-07-18)

- New registered prior in `cogwheel/lensing/prior.py`:
  `class LensedMarginalizedExtrinsicIASPrior(RegisteredPriorMixin,
  CombinedPrior)`. prior_classes = `IntrinsicIASPrior.prior_classes` (reused,
  DRY) + [FixedLensGeometryPrior, UniformLensMassPrior, UniformReducedShearPrior,
  UniformSourcePositionPrior]. default_likelihood_class =
  LensedMarginalizedExtrinsicLikelihood. Exported from lensing/__init__.py.
- Mirrors LensedIASPrior but starts from INTRINSIC (extrinsic removed) so
  standard_params == marginalized likelihood params. VERIFIED (runtime import):
  registered in prior_registry; default_lik matches; prior_classes head ==
  IntrinsicIASPrior.prior_classes; UniformLensMassPrior precedes
  UniformSourcePositionPrior (m_lens_msun conditioning OK — registration itself
  proves conditioned_on empty); standard_params SET-EQUAL to
  sorted(MEL.params | _LENS_PARAMS) = the 12 intrinsic + 7 lens params.
- Static-equality note: compared prior.standard_params to sorted(MEL.params |
  _LENS_PARAMS) rather than a constructed likelihood instance's `.params`
  property (needs event_data). Algebraically exact: likelihood.params =
  sorted((wfg.params | _LENS_PARAMS) - (wfg.params - MEL.params)) and
  MEL.params ⊆ wfg.params → equals sorted(MEL.params | _LENS_PARAMS). UNVERIFIED
  against a real constructed likelihood instance (no test authored per role).
- LensedPosterior LEFT UNTOUCHED: does not hardcode a likelihood class;
  construction flows Posterior.from_event → prior.default_likelihood_class;
  delta_t_max threads via likelihood_kwargs exactly as the plain
  LensedRelativeBinningLikelihood (already-working LensedIASPrior path). Import
  circular-safety: prior.py now imports marginalized_likelihood (which imports
  likelihood + marginalized_extrinsic; no back-import to prior) — no cycle.

## WP1 LensedMarginalizedExtrinsicLikelihood (Build 5, 2026-07-18)

- New `cogwheel/lensing/marginalized_likelihood.py` subclasses
  `MarginalizedExtrinsicLikelihood` (NOT the Base directly — inheriting gives
  `params` class-attr + `_create_coherent_score` for free = exact mirror, DRY).
  Exported from `cogwheel/lensing/__init__.py`.
- WP TEXT WAS FACTUALLY WRONG about kernel layout: it assumed engine k0 is
  per-EDGE (len(fbin)) with channel as LAST axis. VERIFIED via find_symbol:
  `_engine._amplification_coefficients(par_dic)` returns
  `(delays[s], k0, k1, partition)` with k0/k1 shape (n_channels, n_bins=
  len(fbin)-1) at bin CENTERS, channel FIRST. Coherent-score `_d_h_weights`
  (mtdb) / `_h_h_weights` (mdb) are EDGE-indexed b=len(fbin). Bridged the
  center↔edge mismatch with helper `_edge_amplification(delays,k0,k1)`:
  reconstruct each image kernel K_a at edges from the certified (k0,k1) linear
  model (slope-correct to edges, average adjacent-bin estimates at interior
  edges), then F(f_b)=Σ_a K_a(f_b)·exp(2πi·dt_a·f_b). This is the load-bearing
  deviation from WP literal text — flagged in change report for Inspector/TestDev.
- Engine built in overridden `_set_summary` (runs inside base __init__ via fbin
  setter, BEFORE terminal lnlike(par_dic_0)), NOT in __init__ body — so
  self._engine exists before the base constructor's terminal lnlike call.
  delta_t_max/bin_delay_tol/kernel_subsamples stored as same-named attrs for
  JSONMixin.get_init_dict round-trip (engine NOT an init arg — rebuilt).
- `params` computed from self.waveform_generator (not self._engine — engine not
  built when params first read): sorted((wfg.params | _LENS_PARAMS) - dropped),
  dropped = wfg.params - MarginalizedExtrinsicLikelihood.params.
- Data term uses F·h (h_lensed.conj()); norm uses |F|²·_h_h_weights reusing base
  einsum with UNLENSED h_mpb (F is mode-independent scalar). Proved on paper:
  image-sum before linear contraction == after; delay phase in conj(F) combines
  with weights' exp(2πi f t) → per-image time shift t-dt_a (exact).
- Refusals propagate unswallowed: call `_amplification_coefficients` +
  `_check_candidate_delays` with NO try/except (LensDomainError/CancellationError/
  LensedBinningError reach posterior boundary).
- VERIFIED: py_compile OK; `from cogwheel.lensing import
  LensedMarginalizedExtrinsicLikelihood` imports; MRO head correct; 3 overrides
  present; __init__ sig matches WP.
- UNVERIFIED (no test authored/run per role): numerical lnlike vs brute/direct;
  pickle round-trip drops engine._fid_cache (relied on engine __getstate__, no
  explicit __getstate__ added); JSONMixin full JSON round-trip.
