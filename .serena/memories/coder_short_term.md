# Coder Short-Term Observations

- WP2 (Build 4 registered prior + posterior refusal net): added
  LensedIASPrior to lensing/prior.py, NEW lensing/posterior.py
  (LensedPosterior), exports in lensing/__init__.py. No engine/likelihood/
  threshold edits.
  * LensedIASPrior(RegisteredPriorMixin, CombinedPrior): prior_classes =
    IASPrior.prior_classes + [FixedLensGeometryPrior, UniformLensMassPrior,
    UniformReducedShearPrior, UniformSourcePositionPrior] (mass BEFORE
    source-position: latter conditioned_on m_lens_msun). default_
    likelihood_class = LensedRelativeBinningLikelihood. Distance stays
    physical (reuses IAS UniformLuminosityVolumePrior; d_app deferred to
    Build 5, documented). Verified: registered in prior_registry, no
    leftover conditioned_on, standard_params == IAS.standard_params |
    _LENS_PARAMS (7 lens params), sampled adds ln_m_lens_msun,gamma,u1,u2.
  * IMPORT NOTE: prior.py now imports cogwheel.gw_prior (IASPrior,
    RegisteredPriorMixin) + cogwheel.lensing.likelihood + CombinedPrior.
    No cycle: likelihood.py imports waveform/chang_refsdal/relative_binning,
    never prior; gw_prior never imports lensing. lensing/__init__ imports
    prior then posterior (pulls heavy numba likelihood) -- intended public API.
  * LensedPosterior(Posterior): ONLY overrides lnposterior_pardic_and_
    metadata -> try super(); except (LensDomainError, CancellationError):
    return (-inf, self.prior.transform(*args,**kw), None) matching base's
    neginf-lnprior shape. transform is sampled->standard only (no engine),
    can't itself refuse. NO n_refusals attr. from_event/from_ref_wf_finder
    INHERITED (registry resolves LensedIASPrior + default_likelihood_class).
    __init__'s __signature__ assignment binds to the OVERRIDE __func__ (works
    since self.method.__func__ is subclass fn).
  * GOTCHA: LensDomainError IS-A ValueError, CancellationError IS-A
    RuntimeError. except tuple names both specifically so an UNRELATED
    ValueError still propagates (probed: only the two named types swallowed;
    zero-prior and normal paths pass through unchanged).
  * Verified (parse+import+logic-probe via __new__ stubs): all 4 refusal/
    passthrough cases + all 7 registration/param/ordering checks green.
  * UNVERIFIED (downstream / Test Dev+Inspector): full-suite green; live
    from_reference_waveform_finder construction on crown event (lens
    subpriors take no extra kwargs, inherited like IASPrior); end-to-end
    prior-draw sweep returns finite-or-exact-inf with no escaping exception;
    fork/pickle cache determinism. Role limit: parse/import/logic-probe only.

- WP1 (Build 4 prior layer, NEW cogwheel/lensing/prior.py): four lens
  subpriors, no combined prior / registration / posterior / engine edits.
  * VERIFIED CONVENTION (WP text guessed "+" signs, told me to check):
    cogwheel `ln_jacobian_determinant` returns log|d{sampled}/d{standard}|
    (INVERSE-transform Jacobian), args = standard_params+conditioned_on.
    Confirmed from gw_prior/mass.py UniformDetectorFrameMassesPrior
    (returns -log(...)/5, args m1,m2) and extrinsic.py
    UniformLuminosityVolumePrior (returns log(d_hat/d_lum), args
    d_luminosity+conditioned). => mass prior returns -log(m_lens_msun);
    source prior returns -2*log(Y(m)) -- NEGATIVE, not the WP's tentative
    "+ln_m" / "2 ln Y".
  * (a) FixedLensGeometryPrior(FixedPrior): standard_par_dic=
    {kappa:0,beta:0,z_lens:0}; range_dic={} inherited; no transform math.
  * (b) UniformLensMassPrior(UniformPriorMixin,Prior): range_dic=
    {ln_m_lens_msun:_LN_M_LENS_RANGE=(log10,log3500)}; standard
    ['m_lens_msun']; transform exp, inverse log, jac -log(m). transform is
    @staticmethod @utils.lru_cache() (mirrors mass template).
  * (c) UniformReducedShearPrior(UniformPriorMixin,IdentityTransformMixin,
    Prior): range_dic={gamma:(0,0.45)}. IdentityTransformMixin auto-sets
    standard_params=sampled_params=['gamma'] via __init_subclass__ and
    provides transform/inverse/unit-jac -> no gamma_prime indirection.
    NOTE: didn't add UnitJacobianMixin explicitly since
    IdentityTransformMixin already subclasses it.
  * (d) UniformSourcePositionPrior(UniformPriorMixin,Prior): range_dic=
    {u1:(-1,1),u2:(-1,1)}; standard ['y1','y2']; conditioned_on=
    ['m_lens_msun']; folded_reflected_params=['u1','u2']; NO phase fold
    (Prof constraint 3, XPHM). Y(m)=min(_Y_SCALE=307/m, _Y_SCALE_CAP=3.0)
    via module fn _source_scale; transform y=u*Y, inverse u=y/Y, jac
    -2*log(Y). 307 = 55/(sqrt2*1.2372e-4*1024) box-corner provenance in
    comment.
  * MRO: mixins must precede Prior (check_inheritance_order enforces);
    UniformPriorMixin.lnprior returns max_lnprior from cubesize (uniform),
    no override needed.
  * Verified: AST parse OK; import OK; sampled union =={ln_m_lens_msun,
    gamma,u1,u2}; standard union =={m_lens_msun,z_lens,y1,y2,gamma,beta,
    kappa} (the 7 lens params waveform.py consumes); round-trips exact to
    1e-12; jac signs match template; cap fires (307/50->3.0), 307/1000->
    0.307; conditioned_on/folded_reflected as required; no phi/phase in any
    fold list. Only new file created; no engine/likelihood/posterior touched.
  * UNVERIFIED (downstream / Test Dev+Inspector): full-suite green; folding
    unfold-sum consistency; positive-parity dense-sweep safety; end-to-end
    Posterior smoke. Role limit: parse/import/logic-probe only, no combined
    prior exists yet to exercise folding machinery.

- WP3 (Build 4, fork/pickle safety, lensing/likelihood.py): added
  `__getstate__`/`__setstate__` to LensedRelativeBinningLikelihood ONLY,
  inserted right after `__init__`. No base __getstate__/__setstate__/
  __reduce__ anywhere in repo (grep clean) -> default pickle used
  self.__dict__ before; JSONMixin serializes via get_init_dict (JSON
  path), untouched. `__getstate__` = self.__dict__.copy() then
  state.pop('_fid_cache', None) -> drops ONLY the memoization cache;
  `_force_direct` (testing seam, behavioral flag, NOT derived state) is
  PRESERVED so a pickled instance evaluates identically. `__setstate__`
  = self.__dict__.update(state); self._fid_cache = {} (empty rebuild).
  Rationale docstring on __getstate__: cache is a pure deterministic
  memoization on a fixed lattice, forked worker rebuilds bit-identical
  on first eval (~one direct SACR-C eval per lattice cell per worker),
  determinism preserved.
  * Verified: AST parse OK; import OK; both methods in class __dict__;
    direct state-logic test (getstate excludes _fid_cache, keeps
    _force_direct/delta_t_max/n_bins, copy semantics leave original
    untouched; setstate re-inits _fid_cache={} and restores rest; full
    pickle.dumps/loads round-trip) all green via L.__new__ probe.
  * UNVERIFIED (downstream / needs full event-data instance): pickled-
    then-unpickled instance reproduces parent's lnlike bit-identically
    after cache warm-up; full suite at original tolerances. Role limit:
    parse/import/logic-probe only.

- WP2 (Build 3g ratio layer, lensing/likelihood.py): implemented the
  candidate/fiducial heterodyne on top of WP1's seams. No new public API,
  no constructor param, no njit, no tolerance change. Verified: AST parse
  OK, module imports OK, get_symbols_overview clean (no dup methods), njit
  grep = only two doc-comments (no decorators).
  * `_amplification_coefficients(par_dic)` is now the DISPATCH: eval
    candidate seed ONCE via `_evaluate_envelope(lens, seed_w, pad_w=w_max)`
    with `seed_w = geomspace(dense_w.min(), dense_w.max(), _LOO_SEED_NODES)`;
    pack `seed=(partition_cand, seed_w, seed_env, seed_ftot)`. Order:
    _force_direct bypass -> build/lookup fiducial (try/except ONLY here) ->
    Guard1 real_mask.sum() mismatch -> Guard2 min|E_fid|/max|E_fid| <
    _ENVELOPE_HEALTH_FLOOR -> else `_ratio_coefficients`. Every fallback
    forwards the SAME seed to `_amplification_coefficients_direct(par_dic,
    seed=seed)` (no double engine work).
  * Refusal symmetry: candidate seed eval + direct/ratio LOO refinement
    engine nodes propagate LensDomainError/CancellationError UNSWALLOWED;
    only `_get_or_build_fiducial` is wrapped -> fallback to direct.
  * `_ratio_coefficients` uses CANDIDATE partition's critical_delay/
    delays/saddle/switch in `_kernels_from_dense_envelope` (the Simplifier
    trap); dtau_c = partition_cand.critical_delay - fiducial.partition.
    critical_delay; envelope_dense = exp(-1j*w*dtau_c)*rho_dense*E_fid(w).
  * `_ratio_loo_nodes` node_error: rho=exp(1j*w*dtau_c)*E_cand/E_fid,
    LOO on rho, returns loo*|E_fid| (currency = E_cand/max|F|), scale =
    max|F_cand nodes|; reuses shared `_refine_envelope_grid`.
  * DRY seam: `_refine_envelope_grid(lens, coarse_w, env_nodes, ftot_nodes,
    node_error)` is the ONE LOO loop; `_envelope_loo_nodes` (seed kwarg) and
    `_ratio_loo_nodes` differ only in the node_error closure. Extracted
    `_image_delays` (xi*delays/2pi) and `_reduce_dense_kernels` (einsum
    fit) shared by both paths.
  * `_FiducialEnvelope` frozen dataclass (partition, coarse_w,
    envelope_nodes, spline_real, spline_imag; .envelope(w) evals Re/Im
    cubic-in-ln(w)). `_lens_from_key` is exact inverse of `_fiducial_key`
    (7-tuple order gamma,beta,kappa,y1,y2,m_lens,z_lens). `_fid_cache`
    keyed on _fiducial_key ONLY; `self._force_direct=False` testing seam.
  * Lattice-point identity (dtau_c==0 => ratio==direct to machine eps):
    holds because ratio seed_w == fiducial LOO seed geomspace (same
    _LOO_SEED_NODES over same [dense_w.min,max]) => rho=1 at all seed
    nodes => LOO error 0 => constant rho spline => envelope_dense =
    same fiducial spline the direct candidate nodes reproduce. DOWNSTREAM
    to confirm numerically (I did not run suites).
  * UNVERIFIED (downstream): ratio-vs-direct & ratio-vs-brute agreement at
    tolerances; 10 ms timing; fiducial memoization independence. Only did
    parse+import per role limit.

- WP1 (Build 3g ratio-layer scaffolding, lensing/likelihood.py): behavior-
  preserving refactor + inert scaffolding, no behavior change.
  * Extracted `_kernels_from_dense_envelope(dense_w, envelope_dense,
    partition)` = the saddle/switch/reconstruct_from_envelope tail;
    `_reconstruct_kernels` keeps the spline-of-nodes step then calls it.
  * `_amplification_coefficients` is now a thin dispatch to
    `_amplification_coefficients_direct(par_dic)` (old body byte-for-byte).
    Sole caller `_get_dh_hh_no_asd_drift` still hits the thin wrapper.
  * New module constants near the `_LOO_*` block: `_FID_GAMMA_SPACING=0.03`,
    `_FID_BETA_SPACING=np.pi/16`, `_FID_KAPPA_SPACING=0.02`,
    `_FID_Y_SPACING=0.05`, `_ENVELOPE_HEALTH_FLOOR=0.01`.
  * New pure module fns `_snap(x,dx)=round(x/dx)*dx` and `_fiducial_key(lens)`
    -> 7-tuple (snap gamma/beta/kappa/y1/y2, m_lens_msun & z_lens EXACT).
  * `self._fid_cache = {}` in __init__. Confirmed JSONMixin serializes via
    get_init_dict() (inspect.signature of __init__), NOT __dict__, so
    _fid_cache is not serialized -> no DATA_CONTRACTS change (in-memory only).
  * Working dir is the WORKTREE /home/tejaswi/Work/cogwheel-claude-dev; the
    built-in Edit tool errored on absolute /home/tejaswi/Work/cogwheel path
    (wrong tree) -> use Serena replace_content with relative paths here.
  * insert_at_line at a constant block split a comment from its assignment;
    always re-read and fix ordering after mid-file inserts.
  * Verified: AST parse OK, module imports, constants/_snap/_fiducial_key
    behave per spec. Did NOT run lensing suites (downstream verifies).
