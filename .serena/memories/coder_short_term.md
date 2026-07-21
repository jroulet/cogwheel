# Coder Short-Term Observations

- WP5 (Build 8f lever 5, L_MAX gate hardening): edited ONLY
  cogwheel/lensing/chang_refsdal/operator.py (geometry.py/channels.py
  UNTOUCHED; L_MAX stays 48 per Professor override of the brief's
  "relax toward floor"). Three changes, no new exception class:
  (1) L_MAX constant comment (~L206) re-provenanced to the mandated
  double-sided-bracket text (HANDOFF exponent in the wave/geometric
  OVERLAP, wave series acc L~45-46 F005, geometric onset ~50 governed
  by w*delta F013, 48=census-(b) 13.9% crossover, refusal band [46,48]
  by CancellationError, 50=ceiling; raising past 48 pushes nodes onto
  wave past 1e-10 => refuse). Value UNCHANGED (48).
  (2) NEW module helper `_certify_geometric_census(images, matrix)`
  inserted before geometric_amplification. Guard (a) IMAGE-COUNT:
  len(images) not in {2,4} -> geometry.LensDomainError. Guard (b)
  MORSE PARITY-SUM: sum((-1)**geometry.morse_index(im,matrix)) must ==
  (1 if det(matrix)>0 else -1)-1 (0 pos-parity, -2 saddle); else
  LensDomainError. morse_index = eigvalsh strict <0 (reused, DRY).
  This is _check_image_census's core WITHOUT the degeneracy escape
  (strict) — the degenerate fold-3/cusp-2 censuses that the escape
  lets through are exactly the "must not reach geometric" cases (they
  route to wave via the resolution gate in production, so never hit
  here => byte-identical prod).
  (3) geometric_amplification now hoists images=find_images(...) into
  a list, calls the guard, then iterates the SAME list (same order =>
  byte-identical sum). Docstring Raises updated. select_branch: LOGIC
  UNCHANGED, added Notes provenance paragraph only.
  * KEY PHYSICS (why {2,4} is value-preserving for BOTH parities): the
    Morse index theorem forces n_pos-n_neg == sign(detA)-1 (even), so
    n_total is EVEN; quartic gives <=4 real roots => valid served
    counts are exactly {2,4} (2=outside caustic, 4=inside) for positive
    parity AND macro saddle alike. So guard (a)={2,4} never fires on a
    valid saddle config (saddle "outside"=2 saddles, "inside"=4).
  * VERIFIED (engine present): ast.parse OK; import OK. Value-
    preservation vs ungated sum: pos-parity 1385 configs (426 count-2 +
    959 count-4, w={80,500,3000}) tobytes-IDENTICAL, 0 spurious guard
    fires; genuine macro-saddle (|gamma|>1-kappa) 400 configs (396
    count-2 + 4 count-4) all byte-identical, 0 spurious fires; count
    histogram both parities EXCLUSIVELY {2,4} (0 "other"). Guard RED:
    count 1 and 3 -> guard-a LensDomainError; fabricated two-minima
    (morse [0,0], signed sum 2 != 0) -> guard-b LensDomainError. L_MAX
    ==48 exported.
  * OWED to Test Dev (WP5 gates): (1) enforcement graduation of the
    8d headroom audit test — assert shipped L_MAX (48) respects the
    measured worst-config 1e-4 crossing floor + 1.5x margin; the
    empirical bracket floors (L_geo, L_wave) are Test-Dev-measured, NOT
    Coder-derivable (brief). (2) guard red-capability: wrong image
    count (odd/>4) and Morse parity-sum violation each RED via
    geometry.LensDomainError (no new exception). (3) value-preservation
    witness: every resolved served config (both parities) byte-identical
    to HEAD (guards pass silently) — the ~1800-config sweep above is the
    template. (4) NOTE the guard is STRICTER than find_images'
    _check_image_census (no degeneracy escape) — a fold-merged-3 /
    cusp-merged-2 census that find_images returns will be REFUSED here
    (correct: those must route to wave, and the resolution gate ensures
    they never reach geometric in prod). Test authorship declined
    (code+blessing must not share an author).
  * UNVERIFIED (WP5): full lensing test suite pass/fail (downstream, I
    do not run suites). Any existing test that drives
    geometric_amplification on a DEGENERATE (near-caustic, count 1/3 or
    cusp-merged-2) census directly — bypassing the resolution gate —
    will now RED by design (intended strict refusal); such tests must
    move to a resolved fixture or assert the LensDomainError.

- WP1 (Build 8f lever 1, geometry_partition residual): edited ONLY the
  quartic root solve inside cogwheel/lensing/chang_refsdal/geometry.py
  (new `_companion_roots` helper + one call swap in `_generic_candidates`
  line ~615) + NEW committed micro-benchmark
  scripts/profile_geometry_partition.py. channels.py UNTOUCHED;
  image_kernel physics, Newton polish, census, sort ALL UNTOUCHED.
  * PROFILE (committed script, ran as diagnostic): residual ~636us/call
    (n_configs160,n_w128,rep40,seed0). Split: find_images 47.6%
    (312us, DOMINANT, n_w-independent scalar), physical_kernels 41%
    (265us, mostly per-image linalg det/eigvalsh — grows slowly w/ n_w),
    per_image_delays 5.3%, channel_switch 6.1%. nearest_caustic_point
    (122us, Build-8b, NOT optimized) + assign_labels reported as context.
    find_images DOMINANT across n_w in {32,128}. Confirms Professor+
    Simplifier prior.
  * WITHIN find_images (~316us): candidate-gen incl np.roots 73us
    (np.roots alone ~58us), +accept/polish 106us, +sort/census 102us.
    Census (morse_index=eigvalsh per image) + polish are CORRECTNESS
    GATES whose DISCRETE outputs are byte-identity-FRAGILE near folds
    (eigvalsh sign of a ~0 eigenvalue) — UNSAFE to rewrite. The root
    solve is the one BYTE-IDENTICAL win.
  * OPT: replaced np.roots(quartic) with `_companion_roots` = eigvals of
    the Frobenius companion — the SAME algorithm np.roots uses, minus its
    generic wrapper (leading/trailing-zero trim, dtype promo, root pad).
    NOT Ferrari — sidesteps the Professor resolvent-cubic branch-select /
    double-root hazard entirely (backward-stable eigenvalue method,
    identical to HEAD). Guard: only fast-paths a full-degree monic-ish
    quartic (coeff[0]!=0, coeff[-1]!=0, 1-D size>=2, all finite); else
    DEFERS to np.roots verbatim. Production always qualifies (det A!=0
    refused upstream => constant=det^2>0, leading=1.0). np.roots' own
    code path for such polys IS eigvals(companion) with no padding, so
    bit-for-bit identical.
  * VERIFIED (engine present): ast.parse+import OK. np.roots-vs-companion
    tobytes IDENTICAL (maxdiff 0.0) on samples. find_images byte-identity
    vs HEAD side-by-side (git show HEAD:geometry.py loaded standalone):
    1800 configs incl near-caustic eta=+-0.002 crossings — worst image
    rel-diff 0.0, worst delay rel-diff 0.0, refuse_decision_mismatch=0,
    image_count_mismatch=0. End-to-end geometry_partition (400 configs):
    real_mask_mismatch=0, worst kernel/switch/delay abs-diff 0.0 (channels
    unchanged => byte-identical by construction). Post-opt benchmark:
    find_images 312->298us (root solve 58->34.5us in isolation, ~40%
    faster). Timing is a DIAGNOSTIC, not a gate (architect policy).
  * NOTE: np.roots also appears in _pearcey_cusp.py & _pearcey_table.py
    (fold-caustic cubic 4t^3+2xt+y) — DIFFERENT module, NOT the
    geometry_partition residual, left UNTOUCHED (out of WP1 scope).
  * OWED to Test Dev (WP1 gates): (1) find_images value-preservation
    new-vs-HEAD <=1e-10 relative (actually 0.0) over the sweep INCLUDING
    eta=+-0.002 near-caustic; (2) image count / labels / real_mask /
    switch-decision byte-identity; (3) _companion_roots delegation: a
    trailing-zero / leading-zero / non-finite quartic must equal
    np.roots(coeffs) exactly (the deferral branch); (4) the fast-path
    companion tobytes == np.roots on the monic quartic. Test authorship
    declined (code+blessing must not share an author).
  * UNVERIFIED (WP1): full lensing test suite pass/fail (downstream, I do
    not run suites); the reported residual split is FIXTURE-scale
    synthetic (seed0, n_w128) — owner-facing ~2ms real split needs a run
    on production dense_w with real lens configs.

- WP2 (Build 8f lever 2, likelihood moment contraction): edited ONLY the
  mode-pair reduction inside cogwheel/lensing/likelihood.py::_norm_term
  (lines ~549-654) + NEW committed micro-benchmark
  scripts/profile_likelihood_contraction.py. _data_term UNTOUCHED (its
  mode reduction is 'mdb,mdb->db', LINEAR in n_m, only 5 einsums — not the
  measured-dominant term). Everything in _norm_term from `m0 = n00+n11+n22`
  onward (image reduction, kpair/nu/phase/contrib) is BYTE-IDENTICAL to HEAD.
  * FUSION: replaced the 12 three-operand einsums `np.einsum('mMdb,mdb,
    Mdb->db', bp, x_m, y_mprime)` (reduce_pairs called 12x, quadratic in
    n_m) with a hoisted left-contraction per B^(p): r_stack=stack(r0,r1),
    rho_stack=stack(rho0,rho1); bilinear(bp)= einsum('mMdb,pmdb->pMdb',bp,
    r_stack) then einsum('pMdb,qMdb->pqdb',left,rho_stack) -> Q[p,q,d,b].
    Then n00=q0[0,0], n11=q1[1,0]+q1[0,1], n10=q1[0,0], n22=q2[1,1],
    n21=q2[1,0]+q2[0,1], n20=q2[0,0], n32=q3[1,1], n31=q3[1,0]+q3[0,1],
    n30=q3[0,0]. p=left mode (r), q=right mode (rho) — MAPS EXACTLY to
    original reduce_pairs(bp,x=r_p,y=rho_q). 4 B^(p) tensors -> 8 quadratic
    einsums (2 per bilinear) vs 12 originally; the m'-dots are linear. Pure
    re-association (sum over m first then m') — NO math change, NO mask/
    branch change.
  * VERIFIED (engine present): ast.parse both files OK; module imports OK.
    VALUE-PRESERVATION vs HEAD side-by-side copy (git show HEAD:...
    _norm_term regex-extracted, exec'd sharing np/_TWO_PI_I) over 5-shape
    sweep [(5,3,256,4),(2,1,64,2),(6,2,128,5),(1,3,32,1),(4,2,300,3)]:
    WORST max_rel=1.66e-15 (<= 1e-10, PASS), dtype+shape identical every
    shape. Benchmark smoke (n_bins64,repeats20): runs, _norm_term 84.1% of
    contraction — CONFIRMS Simplifier static call that norm-term (quadratic
    n_m) dominates; dominance grows with n_modes. --help OK.
  * Benchmark = self-contained micro-bench (synthetic complex-normal arrays
    at documented shapes, argparse n-modes/n-det/n-bins/n-img/repeats/seed,
    warm-up call each, timeit mean); reports per-call ms + %-split +
    dominant term. TIMING IS A DIAGNOSTIC, not a gate (per WP + my role).
  * OWED to Test Dev (WP2 gates): (1) value-preservation _norm_term
    new-vs-HEAD <=1e-10 RELATIVE over the shape sweep WITH an ABSOLUTE-
    tolerance fallback where the normalization denominator underflows
    (Professor guard — relative tol meaningless there); (2) mask/branch
    byte-identity (no included-term/mask decision changed — structurally
    true, downstream of `m0=...` untouched); (3) dtype/shape identity.
    Test authorship declined (code+blessing must not share an author).
  * UNVERIFIED (WP2): full lensing test suite pass/fail (downstream, I do
    not run suites); the reported 84.1% split is FIXTURE-scale synthetic
    (seed0, small shape) — the owner-facing ~2.3ms real split needs a run
    on production (n_modes,n_det,n_bins) with real moment tensors.

- WP4 (Build 8f lever 4, universal Pearcey P(x,y) spline table as a
  registered data product): NEW
  cogwheel/lensing/chang_refsdal/_pearcey_table.py + scripts/
  train_pearcey_table.py; EDITED _pearcey_cusp.py (consult wiring only,
  pearcey()/pearcey_asymptotic/primitive-quadrature internals UNTOUCHED);
  registered pearcey_table in DATA_CONTRACTS.yaml (0.1.0->0.2.0) +
  data_registry.yaml (1.0.0->1.1.0). NO engine internals touched.
  * TABLE = bicubic RectBivariateSpline (kx=ky=3,s=0) of the FRESNEL-
    DEMODULATED P: store demod = P*exp(-i*phi_sp), phi_sp = t*^4+x t*^2+
    y t* at the DOMINANT-CONTINUOUS real stationary root t* of
    4t^3+2xt+y=0. Selection rule = MAX |phi''|=|12t^2+2x| (the isolated/
    non-merging branch) => carrier CONTINUOUS across the fold caustic
    27y^2=-8x^3 (the required property; the merging pair has curvature->0
    there). Only C0 (value-continuous, phi equal) across the interior
    Maxwell seam y=0 — absorbed by grid grading. remodulate multiplies
    exp(+i phi_sp) back.
  * SERIALIZATION allow_pickle=False: npz of plain float64 arrays x_grid,
    y_grid (graded, symmetric, power=1.6 clustering near 0), demod_real,
    demod_imag (nx,ny) + provenance JSON stored as np.asarray(json.dumps)
    (0-d '<U' unicode scalar, NOT pickled). from_grid REFITS splines at
    load (splines never serialized). content_hash = sha1 over the 4
    arrays' float64 bytes + box scalars [x_max,y_max,margin,oracle_tol]
    bytes; PearceyTable.load recomputes & compares -> ValueError on
    mismatch (NO new exception class per 8f fence; ValueError/OSError/
    KeyError are the load-anomaly vocabulary).
  * BOX derive_box (OFFLINE): ray fan (180 rays) x radial march (240),
    handoff r per ray = smallest r beyond which |pearcey_asymptotic-
    pearcey| < oracle_tol staying below out to r_stop; X_MAX,Y_MAX =
    axis-aligned enclosing half-widths *(1+margin=0.15). Then FORCE the
    served semicubical caustic strictly inside: y_max = max(y_max,
    sqrt(8/27)*x_max^1.5*(1+margin)). derive_box is EXPENSIVE (43200
    pearcey@~45ms ~ 30min) — the offline training run, NOT run by me.
  * CONSULT WIRING (_pearcey_cusp.py): module global _PEARCEY_TABLE=None
    + set/get/use_pearcey_table (use_ = opt-in loader: on ANY load/hash
    anomaly warns, leaves global None, returns False — never installs a
    bad table). cusp_amplification gained kwarg pearcey_table=None; the
    single primitive site `primitive = pearcey(x_eval,y_eval)` became
    `table = pearcey_table or _PEARCEY_TABLE; primitive =
    _consult_pearcey(x_eval,y_eval,table)`. _consult_pearcey: table None
    -> pearcey(x,y) (BYTE-IDENTICAL HEAD); else table.evaluate (None if
    outside box / non-finite / anomaly) -> serve, else fall back to
    pearcey(x,y). evaluate() is TOTAL (returns None, never raises) so the
    consult needs no try/except (avoids generic-except smell). Default
    (no global, no kwarg) => byte-identical to HEAD by construction.
  * pearcey_asymptotic NOT tabled (closed-form, cheap); only the 45ms
    quadrature primitive is amortized. kappa has no table axis — fine,
    P(x,y) is lens-independent (the demod controls already fold in all
    geometry).
  * VERIFIED (smoke, engine present): ast.parse 3 files; import OK;
    pipeline_graph trace pearcey_table -> producer main + 2 consumers;
    list -> registry_path=yes. Tiny-box round-trip (build->save->load):
    hash matches, evaluate reproducible, grid-node table-vs-quad
    abs-err 3e-16 (spline interpolates nodes exactly), outside-box
    evaluate None, byte-identity _consult_pearcey(x,y,None)==pearcey(x,y)
    at 3 pts, consult inside==table & outside==live pearcey, hash-mismatch
    load raises ValueError, use_pearcey_table(bad|missing) False + global
    cleared. cusp_amplification runs both with/without global table (no
    crash; returns None for the test config — calibration cert refuses,
    per WP3-8e, routing verified).
  * UNVERIFIED (WP4 8f): (a) FULL derive_box + build + held_out 1e-8
    certification at shipped resolution — offline ~30min+ run, driver
    executes; NO shipped pearcey_table.npz in tree (loader default is
    package-data path; absent file => use_pearcey_table returns False =>
    live-quad fallback, so nothing breaks). (b) full lensing test suite
    pass/fail (downstream). (c) the served AMPLITUDE the table reproduces
    is only as good as pearcey() quadrature (exact-to-1e-8 of the
    primitive) — the cusp arm's own calibration (b1,b2) is still deferred
    per WP3-8e; the table faithfully amortizes pearcey(), it does not fix
    calibration.
  * OWED to Test Dev (8f table gates): (1) round-trip byte-identity
    (save->load->evaluate); (2) hash-mismatch load RED -> ValueError ->
    use_pearcey_table False (F010-style stored-hash mutation MUST fall
    back, not serve — brief-mandated); (3) missing-file fallback; (4)
    byte-identity cusp_amplification with no table == HEAD (consult(None)
    ==pearcey); (5) held-out 1e-8 accuracy vs quadrature on a small
    fixture box; (6) outside-box -> live-quad fallback both directions.
    Test authorship declined (code+blessing must not share an author).
  * NOTE: sync_derived_docs.py --check rc=1 is PRE-EXISTING
    (lens_amplification_surrogate TEST consumers not in DATA_CONTRACTS,
    consumer-graph not regenerated — rg missing per WP-REG note); my
    pearcey_table entry raised ZERO new complaints.

- WP3 (Build 8f lever 3, node-parallel exact Schwinger): edited ONLY
  cogwheel/lensing/chang_refsdal/operator.py (evaluator bodies
  _schwinger/_dd/_hyp1f1 UNTOUCHED). Added `import math`, `dd_complex_sub`
  from _dd, and a BARE module-global alias
  `_schwinger_raw_t_integral_core = _schwinger._raw_t_integral_core`
  (this alias is the F010 patch handle — numba freezes it at compile time,
  so patching it only bites through .py_func). NEW njit driver
  `_schwinger_raw_integral_map(w_nodes,a,b,y1,y2,u_lo,u_mid,u_hi,n_panels,
  xk_hi,xk_lo,wk_hi,wk_lo)` = njit(parallel=True,cache=True,fastmath=False),
  PURE MAP over numba.prange(m): per node calls core at n_panels[i] then
  2*n_panels[i], writes dd-complex 4-tuples to DISJOINT rows int_n[i,:]/
  int_2n[i,:], returns (int_n,int_2n) shape (m,4). NO reduction inside
  prange. NEW Python wrapper `_schwinger_wave_grid_values(w_nodes,y_eig,
  gamma_prime,lam,kappa,s)`: computes per-node setup (t_cap,log_t_cap,
  margin,u_lo/mid/hi,n_panels via _panel_count) in CPython byte-identical
  to f_schwinger, calls driver, then per-node cert (dd_complex_sub +
  _dd_complex_magnitude + _CERTIFICATION_TOL) + _reconstruct + mass-sheet
  phase; returns (values, certified) — does NOT raise.
- Both _saddle_grid (~910-993) and _positive_parity_grid (gamma'>0 loop,
  ~1470-1540) node loops REPLACED with a Python PRE-PASS classifying nodes
  into geometric (saddle only, resolved & w>60), arm (_uniform_arm_value ->
  value or ceiling_refusers), else batch_index; then ONE
  _schwinger_wave_grid_values call over w_array[batch_index]; then refusal
  reduction in Python: refusers = ceiling_refusers + uncertified batch
  nodes; if any, re-run f_schwinger on min(refusers) to raise the AUTHENTIC
  SchwingerCertificationError (lowest-index = serial first-refuser message
  identity) + unreachable guard raise. Eigenframe reduce (lam,y_scaled,
  gamma_prime,z_eig,y_eig,s) computed ONCE above the pre-pass in both
  (saddle L910-919), passed INTO the batch — never recomputed per node.
  NEW diagnostic `_measure_node_parallel_speedup(...)` (mirrors
  _measure_warm_cost, not in __all__, local `import time`).
- VERIFIED (smoke, engine present): ast.parse OK. Byte-identity grid
  (parallel) vs per-node F_op both parities: max|diff|=0.0, tobytes equal.
  F_op/F_op_grid end-to-end finite both parities (pos kappa=0; saddle
  kappa=0.8). Driver .py_func present. F010 RED via .py_func: clean py_func
  max|int_2n|=7e-6, with mock.patch.object(op,'_schwinger_raw_t_integral_core',
  zeroing corrupt) py_func -> 0.0 (corruption reachable through compiled-
  driver's py_func chain). Refusal identity: serial F_op(w=300) and
  F_op_grid([12,18,300,24]) both raise SchwingerCertificationError, SAME
  type AND SAME message. Determinism: 5 grid runs tobytes-identical.
  Speedup diagnostic: 12 nodes serial 1535.7ms / parallel 299.0ms = 5.14x.
- NOTE: mock.patch.object on the module global does NOT falsify the
  COMPILED F_op_grid path (numba froze the core ref) — that's the F010
  contract; falsification MUST go through .py_func. Confirmed: patched +
  compiled F_op_grid still returned a value (expected); patched + .py_func
  went to 0.0 (red).
- UNVERIFIED (WP3 8f): full lensing test suite pass/fail (I do not run
  suites; downstream). OWED to Test Dev: (1) new/updated F010 gate must
  patch `operator._schwinger_raw_t_integral_core` and drive
  `_schwinger_raw_integral_map.py_func(...)` RED (compiled F_op_grid path
  will NOT go red — that's correct, must use py_func). (2) byte-identity
  sweep parallel-vs-serial per node both parities (max|diff|==0.0,
  diagnostics arrays bit-for-bit). (3) refusal-decision identity (same
  SchwingerCertificationError type+message). (4) scheduling-independent
  determinism (repeated runs byte-identical). (5) maintenance-coupling
  guard: _schwinger_wave_grid_values setup/cert/reconstruct tail MIRRORS
  f_schwinger's body — if f_schwinger's setup/cert changes, the wrapper
  must change in lockstep (byte-identity suite is the guard).

- WP4 (Build 8e wire uniform arms into serving ladder): edited 3 files,
  engine internals (_schwinger/_dd/_hyp1f1) + L_MAX + select_branch
  UNTOUCHED. (1) operator.py: new module-level helper _uniform_arm_value(
  w,y,gamma,*,beta,kappa)->complex|None = fold Airy (_airy_fold.
  fold_amplification) THEN cusp Pearcey (_pearcey_cusp.cusp_amplification),
  first non-None wins, else None (refusal-conservative, NO new exception).
  Import line extended: `from ...chang_refsdal import (geometry,_schwinger,
  _airy_fold,_pearcey_cusp)`. Intercept inserted at BOTH node-level wave
  refusal sites (_saddle_grid ~L647; _positive_parity_grid gamma'>0 loop
  ~L1171), each guarded `if w_node > _schwinger.W_CEILING_SCHWINGER:` ->
  try arm, serve+continue, else fall to existing f_schwinger (named
  SchwingerCertificationError stands). BYTE-IDENTITY by construction: the
  guard is EXACTLY the previously-refusing set (geometric nodes continue
  earlier via resolved gate; w<=60 nodes never satisfy guard -> identical
  old f_schwinger path). _saddle_grid passes `source`(=asarray(y)),
  _positive_parity_grid passes `y` (both physical frame; arms rotate
  internally). Docstrings updated in both fns (uniform rung before named
  refusal). (2) __init__.py: added `from ._airy_fold import
  fold_amplification, airy_fold_value` + `from ._pearcey_cusp import
  cusp_amplification, pearcey, pearcey_asymptotic` (names verified vs each
  module's __all__). (3) surrogate.py: new module const _CUSP_ARM_COVERAGE
  =0.0 (angular half-width rad of certified Pearcey coverage); _tube_serves
  cusp-window loop now uses `residual=max(0.0,delta_theta-_CUSP_ARM_COVERAGE)`
  in place of raw delta_theta. Chart SCHEMA UNCHANGED (shrink applied at
  query time from module const, not stored). At 0.0 -> residual==delta_theta
  => byte-identical, no enable-by-default. Only _tube_serves cusp loop
  touched; gamma guard/eta floor/image-count/farfield/exclusion-balls
  untouched.
- VERIFIED (read-only + smoke, engine present): ast.parse 3 files OK;
  imports+all 5 arm exports+helper+const resolve; w<=60 pos-parity &
  saddle byte-path finite; w>60 pos-parity ladder: w=80 & 2000 SERVED by
  arm (finite), w=300 named SchwingerCertificationError stands (arms
  declined) — ladder tries fold-then-cusp before named refusal exactly per
  brief. arm signatures find_symbol-verified before wiring:
  fold_amplification(w,source,gamma,*,beta,kappa,envelope_bar),
  cusp_amplification(w,source,gamma,*,beta,kappa,envelope_bar).
- UNVERIFIED (WP4): (a) full lensing test suite pass/fail (I do not run
  suites; downstream). Test Dev OWED: ladder-order gate (fold tried before
  cusp), byte-identity witnesses (w<=60 + geometric == HEAD), w>60
  served-vs-named-refusal decision, surrogate cusp-window shrink at
  nonzero coverage. (b) _CUSP_ARM_COVERAGE nonzero value: left 0.0
  (byte-identical, safe). The actual certified angular coverage of the
  Pearcey arm must be pinned by the corner census (WP1 8e census tool)
  before any window is truly shrunk; served AMPLITUDE of both arms is
  itself leading-order/UNVERIFIED (see WP2/WP3 notes: fold q=0 b4 deferred;
  cusp b1,b2 curvature calibration deferred), so opening coverage>0 waits
  on those calibrations too. Arm intercept is refusal-conservative: a
  wrong arm value can only appear if an arm's OWN runtime certificate
  passes — the ladder itself never fabricates a number.
- SELF-INFLICTED BUG fixed same session (lesson): replace_content called
  with a bogus extra kwarg + repl='placeholder' overwrote the
  _DEFAULT_CAUSTIC_FLOOR/artifact-name block with the literal 'placeholder'
  — re-read confirmed corruption, restored via a second replace_content.
  LESSON: pass ONLY the documented replace_content params (needle,repl,
  mode,relative_path); a stray kwarg is silently ignored and repl still
  applies.

- WP2 (Build 8e uniform Airy fold arm): NEW
  cogwheel/lensing/chang_refsdal/_airy_fold.py (~430 lines, engine +
  geometry.py + __init__.py ALL untouched — git status: only new file;
  WP4 dispatch adds the export). __all__=['airy_fold_value',
  'fold_amplification']. NO new exception class; pure fns; complex-or-None.
  * EVALUATOR airy_fold_value(w,tau_bar,xi_control,p,q,sigma): total fn,
    F=2sqrt(pi)exp(i(w tau_bar+sigma))[p w^{1/6}Ai(-xi) - i q w^{-1/6}
    Ai'(-xi)] via scipy.special.airy. xi_control SIGNED: >0 inside-caustic
    (Ai(-xi) oscillatory), <0 outside (Ai(+|xi|) evanescent) — one formula
    Ai(-xi_control) does both. Finite at xi=0 (Ai(0),Ai'(0) finite).
    Verified: evanescent decays (|F| 1e-48 -> 1e-136 for xi -30 -> -60).
  * CONVENTION FIX (resolved brief inconsistency, high-value): brief gives
    BOTH xi=(3 w Delta_tau/4)^{2/3} AND Delta_tau=(tau_minus-tau_plus)/2 —
    mutually inconsistent by factor 2. Self-consistent choice (the one that
    makes large-xi limit reproduce the geometric two-image sum) uses the
    FULL separation DT=tau_minus-tau_plus: xi=(3 w DT/4)^{2/3}. Derivation:
    Airy osc (2/3)xi^{3/2}=w DT/2 must equal carrier-relative image offset
    w(tau_minus-tau_bar)=w DT/2 -> xi^{3/2}=3wDT/4. Documented in module
    docstring; used DT (full) in code.
  * CALIBRATION (Professor flag #1, closed-form, VERIFIED to full prec):
    matched large-xi asymptotic to geometric sum sqrt|mu_+|e^{iw tau_+} +
    sqrt|mu_-|e^{iw tau_- - i pi/2}. Result: sigma=-pi/4, q=0 (leading),
    p=2^{-1/6}|lambda_h|^{-1/2}|b3|^{-1/3}. The divergent merge scale s0
    CANCELS out of p (confirmed numerically: p==m*c0^{1/4} to 1e-15,
    m=sqrt|mu|=1/sqrt(|lambda_h||b3|s0)). So p is FINITE at the fold, built
    from curvatures NOT raw sqrt|mu| — exactly the flag. q=0 is RIGOROUS
    leading for the pure-phase lensing diffraction integral (unit Kirchhoff
    amplitude => two stationary curvatures equal in magnitude to cubic
    order => Ai' channel vanishes); the Ai' correction needs the QUARTIC b4
    (outside gathered inputs), deferred. Large-xi reconstruction of the
    two-image sum verified by construction: rel_err 7.6e-3 @ xi=3.6, 3.2e-5
    @ xi=16.5 (O(xi^{-3/2}) convergence; correct -pi/2 saddle Morse phase).
  * b3 (soft-axis cubic) = 2 q_s(3p-4 q_s^2)/p^3, p=|x_c|^2,
    q_s=x_c·soft_axis, x_c=nearest.image (caustic critical pt), from the
    3rd deriv of -ln|x| along soft axis (quadratic part contributes 0). In
    the eigenframe soft/hard mixed 2nd-deriv=0 so bare cubic == reduced
    (Lyapunov-Schmidt) cubic at leading order. lambda_h=nearest.
    hard_eigenvalue. |b3|<=_B3_MIN=1e-6 -> refuse (cusp neighborhood).
  * Merging pair: _merging_fold_pair scans find_images sorted by delay,
    takes the delay-ADJACENT (n=0 lower, n=1 higher) pair with min gap;
    ONLY delays used (near-fold mags ill-conditioned, never evaluated).
    tau_plus=min, tau_minus=saddle, DT>0 required.
  * SELF-CERT (literal per brief): err=c_A*xi^{-3/2}, c_A=max(|C1_+|,|C1_-|)
    from geometry.saddle_coefficients (guarded LensDomainError->None);
    refuse if err>envelope_bar (default 0.05) or non-finite.
  * DIAGNOSTIC / UNVERIFIED (served region, same class as cusp arm): the
    LITERAL c_A xi^{-3/2} certificate is the FAR-FIELD (geometric) error,
    which is LARGE near the fold (small xi, large C1) — so it REFUSES the
    tight near-fold band (0 served @ eps<=0.06,w<=800) and only SERVES
    well-separated folds (731/4000 @ eps 0.02-0.15, w 200-8000). The true
    uniform-Airy error is O(w^{-1/3}) UNIFORMLY (incl. on-caustic), NOT
    c_A xi^{-3/2}; the literal estimate inverts the intended served region.
    I did NOT re-tune the constant (would fabricate a served region against
    no oracle = grading own homework). Arm is ALIVE (nonempty served set,
    0 exceptions over ~10k configs) and refusal-conservative (never serves
    a wrong number: evaluator+p+sigma verified, only the GATE constant is
    the question). OWED to driver/Test Dev: brute-force calibrate the
    uniform-error gate (likely ~w^{-1/3} with an O(1) c_A, not xi^{-3/2})
    to open the intended near-fold served band; the served AMPLITUDE is
    leading-order (q=0, b4 deferred) => also owe the Ai' quartic refinement
    cross-check. Structure/refusal-gating/primitive-calibration VERIFIED;
    served near-fold region + Ai' term UNVERIFIED. Caveat in module +
    fold_amplification docstrings.
  * Test authorship (fold-arm gates: evaluator, p/sigma calibration match,
    on-caustic finiteness, evanescent decay, certificate red-capability)
    is Test Dev's — declined (code+blessing must not share an author).

- WP3 (Build 8e uniform Pearcey cusp arm): NEW
  cogwheel/lensing/chang_refsdal/_pearcey_cusp.py (726 lines, engine
  untouched, reads geometry.py, NOT yet exported in __init__.py — WP4
  dispatch adds that). __all__=['pearcey','pearcey_asymptotic',
  'cusp_amplification']. NO new exception class; pure fns; complex-or-None.
  * PRIMITIVE pearcey(x,y): P=int exp[i(t^4+x t^2+y t)]dt via rotated
    contour = central [-hw,hw] + right tail on pi/8 valley + left tail on
    9pi/8 valley (=-e^{i pi/8}); brief's "3pi/8" is a HILL (divergent), used
    pi/8 & its reflection 9pi/8 (documented). Certifies P BEFORE any
    prefactor with paired composite GL N/2N (mirrors _schwinger:
    _CERTIFICATION_TOL=3e-10, _PANEL_ORDER=24, _WAVELENGTHS_PER_PANEL=2.0);
    |P_N-P_2N|/|P_2N|>tol -> None. float64.
  * BUG FIXED (real, high-value): left-tail Jacobian had wrong sign
    (-_VALLEY_DIR); correct is +_VALLEY_DIR (deform real-axis left tail
    onto 9pi/8 ray: dt=-e^{i pi/8}du traversed u:inf->0, two sign flips ->
    +). With wrong sign, at x=y=0 (even integrand) left tail EXACTLY
    cancels right tail leaving only central -> P off by the missing
    real-axis tails (~1%). The N/2N certificate CANNOT catch a contour/
    Jacobian error (both rules integrate the same mis-oriented contour;
    error cancels in the ratio). After fix: P(0,0) matches analytic
    Gamma(1/4)/2*e^{i pi/8} to 2.2e-15; large-arg P converges to
    stationary-phase asymptotic (rel 3.4e-4 at (-12,1); ~1% at moderate R
    is the expected O(R^{-3/2}) leading correction).
    LESSON: a paired N/2N quadrature certificate proves QUADRATURE
    convergence only, NOT contour correctness — always cross-check the
    primitive against a closed-form value at >=1 point.
  * C4<0 DUAL CUSP handled (was a dead-arm bug): the standard
    minimum-image cusp has reduced quartic C4<0 (measured -0.196 at the
    gamma=0.4 real cusp). Old guard c4>0 refused the ENTIRE physical class.
    Relaxed _soft_normal_form to accept signed C4 (reject only |C4|<=_C4_MIN
    =1e-6 degeneracy). C4<0 maps to the primitive by the EXACT identity
    int exp[i(C4 s^4+..)]ds = |C4|^{-1/4} conj(P(-x,-y)) (substitution
    s=|C4|^{-1/4}t, NOT a fit): cusp_amplification now sets reflected=c4<0,
    x_eval,y_eval=(-x,-y), conjugates primitive & asymptotic, and flips the
    reduced-phase sign (phase_sign=-1). abs_c4 used in the w^{1/2}/w^{3/4}
    control scalings (math.sqrt(c4) would ValueError for c4<0).
  * DIAGNOSTIC / UNVERIFIED (calibration): with C4<0 accepted the refusal
    moved to the runtime CALIBRATION CERTIFICATE (_calibration_certified:
    each reduced stationary phase must match a distinct geometric cusp-
    cluster scaled delay w*(tau-tau_c) within _CALIBRATION_TOL). At the real
    cusp the reduced stationary phase vs geometric cluster delay differ by a
    factor CONSTANT IN w but VARYING WITH OFFSET (4.26 at eps=0.25, 6.95 at
    eps=0.12, 12.5 at eps=0.05). => the w^{1/2}/w^{3/4} SCALINGS ARE CORRECT
    (mismatch w-independent) but the offset->normal-form-coefficient map
    (b1,b2) uses BARE soft/hard-axis projections and is missing O(1)
    curvature factors (b1=-Delta·e_s times mixed 2nd-derivs of the Fermat
    potential, not the bare offset); also #stationary pts (1) vs geometric
    cluster (3 merging images) => controls land in the wrong Pearcey region.
    The certificate is DOING ITS JOB: refuses the mis-calibrated mapping,
    NEVER serves a wrong number. Broad random sweep (600 configs, seed 0):
    0 exceptions, 4 served, 596 refused — arm is refusal-conservative and
    NOT dead. I did NOT hand-tune the curvature constants (would make the
    certificate pass on an unvalidated amplitude = grading own homework
    against a fitted oracle). OWED to driver/Test Dev: pin & brute-force
    validate the b1,b2 offset->control curvature calibration; the served
    AMPLITUDE is UNVERIFIED (structure/refusal-gating verified; primitive
    verified). Caveat documented in cusp_amplification docstring Notes.
  * Gates (cusp_amplification, in order): input guards (w>0, shape (2,),
    finite, envelope_bar>0) -> geometry LensDomainError -> _cusp_vertex
    (golden-section on caustic-speed min) None -> _soft_normal_form None ->
    F016 radius gate R<R_min=(c_P/bar)^{2/3} -> pearcey None -> per-image
    _leading_geometric None -> _calibration_certified. All return None; no
    raise. Full smoke suite GREEN.

- WP1 (Build 8e corner-scoping census): EXTENDED
  scripts/census_homogenization_corners.py ONLY (engine untouched;
  operator.L_MAX stays 48, select_branch/geometry.py NOT modified —
  git status confirms census script is my sole change). Pure-geometry,
  engine-free, deterministic (same seed -> byte-identical JSON; verified
  n=300 seed=1 twice). Classifies refused high-w corner nodes into 4
  buckets under report['corner_scoping']:
  (a) a_geometric_now = high & geometric-served already (smoke: 1091/3040,
      frac 0.359 +Wilson95).
  (b) b_geometric_under_relaxed_l_max = MEASURED-ONLY: resolved refused
      POSITIVE-parity nodes (L=w*|y'|<=48, blocked only by conservative
      cancellation gate) that L_MAX_RELAXED=60 PLUS image-census guard
      (_image_census_matches: mags.size in {2,4} & merging pair is one
      +min & one -saddle) would move onto geometric. Saddle refusals have
      NO L gate -> never rescued. Reports frac 0.194 + Wilson +
      production_l_max=48 note. select_branch/L_MAX UNCHANGED (raising
      L_MAX would REDUCE geometric since gate is L>L_MAX; (b) is a
      what-if count, not an action).
  (c)/(d) cd_uniform_or_hardcore = refusal & ~relaxed: NO hardcoded
      xi*/R* threshold. Emits full fold (w*Delta_tau, +xi=(0.75*wDtau)^(2/3))
      and cusp (R) argument CDFs as fraction_vs_threshold TABLES over fixed
      grids (FOLD_WDTAU_THRESHOLD_GRID 33pt linspace 0..8;
      CUSP_R_THRESHOLD_GRID 32pt [0]+logspace(-1,2,31)); each row
      fraction_resolvable_ge + Wilson95 + fraction_hardcore_lt; split read
      off at arms' certified thresholds POST-BUILD.
- Geometry-only args per refused node: ONE geometry.find_images solve per
  config (when any w>60); delta_min inline-mirrors
  operator._real_delay_min_separation; Delta_tau=delta_min/2 EXACT (merging
  min(n=0)/saddle(n=1) pair IS the delta_min pair — no re-solve). Topology
  via _classify_fold_or_cusp(delays): 'cusp' when a 3+ cluster of
  consecutive delay gaps within CUSP_CLUSTER_DELAY_RATIO=3*gmin, else
  'fold', else 'degenerate' (<2 real images -> hard-core (d)). Cusp R:
  x=w^0.5*|delta_parallel|, y=w^0.75*|delta_perp| (2/3 law), offsets from
  geometry.nearest_caustic_point(gamma,beta,source,kappa) projected on
  soft_axis(parallel)/hard_axis(perp); guarded try/except LensDomainError
  -> degenerate.
- Memory-bounded: streams cd fold/cusp args into fixed threshold-grid
  histograms (counts_ge += (arg[:,None]>=grid[None,:]).sum(0)) — no storing
  2e5x128 per-node samples; the CDF/table IS the histogram. Wilson95 reuses
  existing wilson_interval helper. partition_check asserts
  n_fold_cd+n_cusp_cd+n_degenerate_cd == cd_node_total (smoke: 1358==1358,
  consistent True). All existing report keys preserved (additive section).
- BUG hit & fixed (recorded lesson): replace_symbol_body on `run` DELETED
  the def+docstring (tool body MUST include the signature line) -> col-0
  IndentationError; repaired via replace_content re-adding def/docstring +
  reindent, then all later replace_symbol_body calls INCLUDED signatures.
  ast.parse + importlib(exec w/ sys.modules registration for KW_ONLY
  dataclass resolution) both GREEN; e2e smoke run GREEN.
- UNVERIFIED (WP1 8e): full N>=1e5/2e5 production run pass/fail + wall-clock
  (smoke n=300 only; each high-w config does one find_images solve +
  possibly one nearest_caustic_point — offline run, owner/downstream
  executes). The reported fractions (a 0.359 / b 0.194 / cd 0.447) and
  topology counts are FIXTURE-scale (seed=1 n=300); owner-facing numbers
  need the full draw. Test authorship (census-extension gates) is Test
  Dev's — declined (code+blessing must not share an author).

- WP3 (Build 8d homogenization-corner census): NEW standalone
  scripts/census_homogenization_corners.py — geometry-only Monte-Carlo
  REPORTING deliverable (NOT a gate), modifies NO engine code (git status:
  only new file; operator.py mod is the pre-existing WP1 change). Draws from
  _LensPriorBox(CombinedPrior) = [FixedLensGeometryPrior, UniformLensMassPrior,
  UniformReducedShearPrior, UniformSourcePositionPrior] via
  generate_random_samples (box READ from prior classes, not hardcoded), mirrors
  surrogate_census idiom but self-contained (all logic in-script per WP "new
  standalone script under scripts/"). Reports: (a) gamma'==0 fraction (==0 in
  smoke, measure-zero) + gamma'<0.01 bonus (tracks analytic 0.01/1.6); (b)
  unresolved-high-w NAMED-refusal corner fraction with Wilson95 interval.
  classify_config mirrors PRODUCTION dispatch per parity: positive
  select_branch gate (resolved w*delta_min>=RHO_END AND L=w*|y'|>L_MAX);
  saddle _saddle_grid gate (resolved AND w>W_CEILING, NO L condition);
  w<=60 -> served-by-Schwinger both parities; w>60 & not-geom-eligible ->
  refusal. delta_min (real-image delay sep, one quartic solve) computed ONCE
  per config and ONLY when any w>60 (perf). macro_matrix/maps/find_images
  LensDomainError (incl. F012 census) -> engine_refused bucket (precedes
  dispatch). gamma_prime read from engine mass-sheet map, cross-checked vs
  analytic gamma/(1-kappa) (smoke: max diff 0.0). Constants imported not
  hardcoded (W_CEILING=_schwinger.W_CEILING_SCHWINGER, RHO_END/L_MAX from
  operator). Smoke n=300 seed=1: gamma'==0=0, gamma'<0.01=1/300, corner
  0.23 [0.186,0.281], engine_refused=0, max_w=441.6<500, xcheck=0.0; JSON
  well-formed (10 top keys). Parse+import+e2e all GREEN.
- UNVERIFIED (WP3): full N>=1e5 production run pass/fail + wall-clock (smoke
  was n=300 only; 2e5 does ~most-config quartic solves for delta_min — offline
  run, downstream/owner executes). Refusal-corner fraction is FIXTURE-scale
  seed=1 n=300 (0.23); the owner-facing number needs the full N>=1e5 draw.

- WP1 (Build 8d homogenize positive parity onto Schwinger): rerouted
  operator.py positive-parity wave branch. RENAMED
  _positive_parity_grid_with_fallback -> _positive_parity_grid (internal;
  only F_op/F_op_grid callers, no test refs to the name). New body is a
  DIRECT dispatch on gamma_prime=_mass_sheet_map(y,gamma,kappa)[2]
  (=gamma/(1-kappa)): gamma'>0 -> per-node _schwinger.f_schwinger with the
  IDENTICAL reduce/rotate/reconstruct copied verbatim from the old 7a
  fallback AND _saddle_grid (z_eig rotate, mass_sheet_phase, /lam); gamma'==0
  (exact, via `not gamma_prime>0.0`) -> legacy _grid_certified (sole legacy
  production exit). Deleted the try/except-CancellationError-then-Schwinger
  fallback structure entirely. Added w_array.ndim!=1 guard (matches
  _saddle_grid). Both F_op(scalar, 1-elem grid) and F_op_grid delegate here
  => single-intercept reroute for both.
- STEP2 refusal symmetry: gamma'>0 above-ceiling now raises
  SchwingerCertificationError (named), NOT the old CancellationError — this
  is the intended homogenization (positive parity == saddle arm behavior).
  No legacy fallback catch. gamma'==0 keeps CancellationError.
- STEP3 oracle alias: module-level `legacy_operator_oracle = _grid_certified`
  inserted right after _grid_certified def, with 'test-only oracle; NOT a
  production path (see Build 8d)' comment. No wrapper/logic.
- STEP4 verified intact (I did NOT touch _grid_certified/_fused_contraction):
  _SERIES_TOLERANCE=2e-12 module global, _fused_contraction.py_func reachable,
  half_sum still a param. Smoke: gamma'>0 F_op order_used=0 & matches direct
  Schwinger recon <1e-14; gamma'==0 order_used=9 (legacy); grid==scalar;
  w=70 gamma'>0 -> SchwingerCertificationError.
- Docstrings updated in-file (F_op, F_op_grid, _positive_parity_grid, module
  MACRO-SADDLE DISPATCH para): removed the now-false 'positive parity ...
  BYTE-IDENTICAL operator/1F1 branch' claim; documented gamma'=0 as sole
  legacy exit + Schwinger for gamma'>0 on both parities.
- DID NOT TOUCH: _saddle_grid (saddle arm), channels.py, select_branch,
  geometry branch dispatch, refusal vocabulary (no new exception classes).
- OWED to Test Dev (Build 8d): tests that pin positive-parity gamma'>0 to
  the LEGACY operator path WILL now go RED and need re-baselining with
  contract-flip witnesses (7b precedent) — NOT my defect, intended flip:
  (1) test_lensing_schwinger.py::PositiveParityBitFreezeTestCase — asserts
  F_op/F_op_grid return frozen literals AND diagnostics.order_used>0 for
  moderate-shear (gamma'>0) configs; both now false (Schwinger value within
  1e-10 of the frozen literal, order_used==0). Re-baseline literals to the
  Schwinger values + flip the order_used>0 assertion to ==0.
  (2) crown byte-identity pin (test_lensing_surrogate.py) and ratio-layer
  cache-determinism pin against the CURRENT positive-parity evaluator WILL
  flip (brief anticipates this); physics tolerances (RB-vs-brute, oracle
  1e-10) must hold. (3) The NEW Build-8d overlap-domain harness
  (Schwinger-vs-legacy_operator_oracle at 1e-10 on the certified overlap,
  refusal-decision identity both directions) + a dispatch mutation test are
  Test Dev deliverables — legacy_operator_oracle is the import handle.
  (4) UNVERIFIED: full lensing suite pass/fail at fixture scale (I only
  smoke-checked import+wiring+one config per branch; downstream runs it).

- WP1 (Build 8c multi-chart surrogate): rewrote
  cogwheel/lensing/surrogate.py into a flat multi-chart emulator — two
  FROZEN dataclasses (TubeChart in (gamma,u=sqrt(eta),theta,log w);
  FarFieldChart in (gamma,y1_eig,y2_eig,log w)), NO hierarchy;
  select_chart() is a plain guard-stack fn keyed on certified physical
  eta+image_count (theta only for cusp-window exclusion, F017). Single-npz
  save/load (chart{i}_* named arrays + one JSON provenance scalar, NO
  manifest); two load paths (package-data default via importlib.resources +
  explicit path override; data_registry fallback left as a TODO). 8a
  backward compat: legacy npz (no `n_charts` key) -> one FarFieldChart via
  _load_legacy_single_box; envelope()/in_domain()/from_engine()/grid attrs
  preserved. Smoke-verified: multi-chart round-trip coeff-exact, serve
  tube+farfield, cusp exclusion + gamma-guard fall-through, legacy load.
- BUG I introduced AND fixed same session: _normalize_refused(None) gave
  array(nan) shape () -> ValueError; None must short-circuit to
  np.empty((0,3)). (np.asarray(None,float) is NOT size 0.)
- channels.py: ONE additive field `caustic_theta: float` on
  ChangRefsdalGeometryPartition, populated `caustic_theta=float(caustic.theta)`
  in geometry_partition (the ONLY constructor; caustic already bound via
  geometry.nearest_caustic_point). Verified populated (2.74 rad sample).
  reconstruct_from_envelope consumed fields untouched.
- likelihood.py: _surrogate_coefficients now does may_serve early-out ->
  geometry_partition -> surrogate.serve(eta=caustic_distance,
  theta=caustic_theta, image_count=int(real_mask.sum())). Removed dead
  _SURROGATE_CAUSTIC_FLOOR const, _surrogate_region_image_count method,
  _surrogate_region_nimg attr (init+getstate pop+setstate+docstring). Kept
  kappa!=0->None guard (INS-8a-001). Intercept still guarded by
  `if amplification_surrogate is not None` => default None byte-identical.
  pyright None-narrowing warnings on serve/may_serve are pre-existing-style
  noise (HEAD had same on in_domain/envelope), not defects.
- WP-CS (Build 8c-cont census tool): NEW cogwheel/lensing/surrogate_census.py
  (importable core, pure compute) + scripts/census_lens_surrogate.py (thin
  arg-parse+run+json.dump CLI, no embedded logic). NO edits to
  surrogate.py/channels.py/likelihood.py/prior.py (all READ-ONLY). Verified
  read-only: parse OK, import OK, guard-predicate signatures match my call
  sites (positional _tube_serves/_farfield_serves, kw select_chart/serve),
  draw_samples yields gamma/m_lens_msun/y1/y2 cols. NO shipped surrogate .npz
  in tree yet => full end-to-end run UNVERIFIED (Test Dev builds fixture).
- Categorization technique: classify_fallthrough toggles ONE guard off on a
  FROZEN chart via dataclasses.replace (cusp_windows=() -> re-call
  _tube_serves for 'cusp-window'; refused_points=empty(0,3) -> re-call
  _farfield_serves for 'refusal-ball'), never re-deriving guard math. Priority
  gamma-guard -> dropped-sliver -> cusp-window -> refusal-ball -> out-of-box.
  Professor Q7 (near-cusp neighbor arc) lands in out-of-box naturally: with
  cusps relaxed the theta-range gate still fails.
- OWED to Test Dev (WP-CS): fallthrough_breakdown reports a 6th bucket
  `engine_refused` BEYOND the brief's 5 categories (samples where
  geometry_partition raised a named refusal BEFORE any surrogate guard — not a
  surrogate decision). Defensive partition assert is
  served + engine_refused + sum(5 cats) == N; the 5 guard cats partition
  (N - served - engine_refused). Hand-computed fixture counts must budget this
  bucket.
- INS-3-001 RESOLVED (verified, no new change by me): surrogate_training.train()
  now accumulates `all_dropped_slivers` across BOTH parities as flat [lo,hi]
  pairs and passes them to _build_provenance(box,config,charts,
  all_dropped_slivers), which writes provenance['dropped_gamma_slivers'] =
  [list(s) for s in ...]. save() serializes the whole provenance via
  json.dumps -> load() json.loads, so the field round-trips. Census
  _dropped_slivers_from(surr, None) reads it by default; _normalize_slivers
  unpacks `for lo,hi in ...` — shapes match. Runtime-verified end to end:
  _build_provenance -> json round-trip -> census read recovers
  ((0.98,1.02),(1.19,1.205)). The finding was written against the EARLIER
  state (see stale note below); threading was added since, so no re-edit.
- INS-1-001 is a Test Developer deliverable (test_lensing_surrogate_census.py)
  — NOT a Coder task; declined authorship (I authored the census module;
  code+its blessing must not share an author). Routed to Test Dev.
- (superseded) UNVERIFIED discrepancy (WP-CS): brief says read dropped_gamma_slivers "from
  the surrogate provenance field" but landed WP3 writes it only to the TRAINING
  REPORT (report['parities'][label]['dropped_gamma_slivers']), NOT
  surrogate.provenance. Handled defensively: _dropped_slivers_from reads
  provenance.get('dropped_gamma_slivers') (empty if absent) AND accepts an
  explicit `dropped_slivers` override; added public helper
  dropped_slivers_from_training_report(report) so the CLI (--dropped-slivers-
  report) stays thin. If a future WP adds slivers to provenance this path
  still works.
- lnL tiers (stage 4) are dependency-injected via lnlike_pair callable
  (default None -> stage skipped/returns None); CLI does NOT wire a likelihood
  (kept thin). assign_tier keys on gamma/eta ONLY (F017, never theta):
  gamma>1 OR gamma'>=0.5 OR eta<=CROWN_CAUSTIC_MARGIN -> strong_saddle else
  crown; rescued via injected best-effort predicate. Held-out eps uses FRESH
  engine_factory(w).evaluate (F002), currency max|Es-Ee|/max(max|Ee|,1e-6).

- WP-REG (Build 8c-cont registration): registered lens_amplification_surrogate
  as a first-class data product across 3 spec/tooling files, NO source touched.
  (a) DATA_CONTRACTS.yaml: new `lens_amplification_surrogate` artifact
  (format npz; producer scripts/train_lens_surrogate.py::main; consumers =
  likelihood.py::LensedRelativeBinningLikelihood,
  marginalized_likelihood.py::LensedMarginalizedExtrinsicLikelihood,
  surrogate_census.py::run; conventions units+array_axes), schema_version
  0.1.0->0.2.0 (MINOR). All function tokens find_symbol-verified BEFORE writing
  so check_data_contracts passes. (b) data_registry.yaml: added `package_data`
  storage_root (path cogwheel/data) + entries.lens_amplification_surrogate ->
  makes pipeline_graph.py report registry_path=yes (verified: `list` shows
  registry_path=yes, `trace` shows all 3 consumers). (c)
  regenerate_consumer_graph.py LOADERS += LensAmplificationSurrogate.load entry
  (idiom-matched, AST+import verified). sync_derived_docs.py --check RC=0.
- UNVERIFIED (WP-REG): CONSUMER_GRAPH.json NOT regenerated — ripgrep (`rg`) is
  not installed anywhere on this machine (not next to interpreter, not on PATH),
  so scripts/regenerate_consumer_graph.py dies at the FIRST loader
  (_files_importing needs rg), unrelated to my new entry. CONSUMER_GRAPH.json
  left UNTOUCHED; LOADERS edit is correct so post-gate doc-sync regenerates it
  once rg is available. This is an env/tooling gap, NOT a denied command.

- OWED to Test Dev: 8a test_lensing_surrogate.py references the OLD single-box
  API attrs (_real_coeffs/_knots/envelope_real/envelope_imag) — these moved
  onto charts[0]; public envelope()/in_domain()/save/load/from_engine kept,
  but any test scraping private single-box internals needs re-targeting at
  charts[0].real_coeffs / .knots. NEW multi-chart tube/farfield selection +
  sqrt(eta) fitting + serialization gates are Test Dev's to author.


- WP-B (Build 8b-levers): fused operator.py `_weight_vectors` +
  `_contract_grid` into ONE njit `_fused_contraction(table, z_powers,
  zbar_powers, abs_powers, half_sum, derivs_scaled, w_array,
  gamma_scaled, max_order, dim)`. Dispatch-only merge: the two loop
  nests are inlined VERBATIM (v/v_abs now internal per-call temporaries),
  so accumulation order is byte-identical. Proved bit-identity vs
  HEAD's two-stage pipeline: 1800 output arrays / 300 random trials,
  0 tobytes mismatches (extract HEAD funcs via git show, strip
  @numba.njit decorators, run .py_func-equivalent bodies — njit
  cache=True cannot exec from a string, must strip decorator). half_sum
  kept an ARG; _SERIES_TOLERANCE/_CONSECUTIVE_SMALL/_MIN_ORDER kept
  module globals referenced by name; .py_func exposed.
- OWED to Test Dev (F010 red-capability, test_lensing_batched_operator.py):
  the two BatchedContractionFalsificationTestCase gates patch the now-
  REMOVED `operator._contract_grid` / `operator._weight_vectors`
  py_funcs (~lines 721, 753) — must be re-targeted at
  `operator._fused_contraction.py_func` (series-tolerance test: patch
  fused->py_func + _SERIES_TOLERANCE; gather test: wrap fused with a
  zeroed-half_sum shim). Also add `_fused_contraction` to
  ORACLE_FORBIDDEN_NAMES (line ~173) and drop the two dead names.
  Until updated these two tests ERROR (AttributeError), not vacuous.
