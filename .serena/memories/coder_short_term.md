# Coder Short-Term Observations

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
