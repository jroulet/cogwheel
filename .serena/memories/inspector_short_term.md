# Inspector Short-Term Observations

## 2026-07-21 — Build 8f review (five serving micro-levers)

Scope: uncommitted tree, worktree /home/tejaswi/Work/cogwheel-claude-dev.
Full python: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.

### VERDICT: PASS (no NEW code defects). Three carried non-code items open.

Code files changed: geometry.py (_companion_roots), likelihood.py
(_norm_term), operator.py (WP3 node-parallel Schwinger + WP5 census guard +
L_MAX comment), _pearcey_cusp.py (WP4 table hooks), new _pearcey_table.py,
new tests/scripts, DATA_CONTRACTS.yaml (0.1.0->0.2.0), data_registry.yaml
(1.0.0->1.1.0). SPEC.md NOT changed (still 0.16.0).

### Re-derived CORRECT this round
- WP1 geometry._companion_roots == numpy.roots: same Frobenius companion
  diag(ones(N-2),-1), A[0,:]=-p[1:]/p[0], numpy.linalg.eigvals; guard
  defers leading/trailing-zero + non-finite to np.roots. Bit-identical for
  the production quartic (det A!=0). dtype preserved via asarray.
- WP2 likelihood._norm_term bilinear re-association: ALL nine n** re-checked
  against the frozen reduce_pairs semantics (Q[p,q]=reduce_pairs(bp,r_p,rho_q)):
  n00=q0[0,0], n11=q1[1,0]+q1[0,1], n22=q2[1,1], n10=q1[0,0],
  n21=q2[1,0]+q2[0,1], n32=q3[1,1], n20=q2[0,0], n31=q3[1,0]+q3[0,1],
  n30=q3[0,0]. einsum 'mMdb,pmdb->pMdb' then 'pMdb,qMdb->pqdb'. FP order only.
- WP3 _schwinger_wave_grid_values MIRRORS f_schwinger byte-for-byte (read
  f_schwinger body @748-837 line-by-line): a=1-g',b=1+g', t_cap, log_t_cap,
  margin=_CANCEL_SCALE*w+_U_MARGIN_CONST, u_lo/mid/hi, _panel_count,
  _dd_gl_rule(_PANEL_ORDER), core call order (N then 2N with u_mid==log_t_cap),
  dd_complex_sub cert vs _CERTIFICATION_TOL, integral=r2[0]+r2[1]+i(r2[2]+r2[3]),
  _reconstruct, mass-sheet phase*f_pure/lam. njit prange PURE MAP, fastmath=OFF,
  no cross-node reduction. Classification partition in _saddle_grid /
  _positive_parity_grid matches OLD serial exactly (geometric: w>ceil&resolved;
  arm/ceiling-refuse: w>ceil&unresolved; batch: w<=ceil). Refusers=ceiling_refusers
  ∪ uncertified-batch; min(refusers) re-run through f_schwinger for authentic
  named exc = OLD first-refuser (lowest index) identity. _validate_inputs skipped
  in batch — harmless (grid nodes pre-validated). NOTE off-production nuance:
  pre-pass evaluates geometric nodes eagerly, so on a pathological multi-defect
  grid a higher-index geometric census defect (LensDomainError) can surface
  before a lower-index Schwinger refuser — both are named refusals -> -inf at
  posterior, so NOT a serving defect; never fires in production (resolved
  geometric censuses are non-degenerate). Covered by INS-3-003.
- WP4 PearceyTable OFF by default (_PEARCEY_TABLE=None -> _consult_pearcey(...,None)
  -> pearcey() byte-identical). load(): allow_pickle=False + SHA1 content-hash
  verify (raises ValueError on mismatch); use_pearcey_table catches
  OSError/ValueError/KeyError -> clears global, returns False. evaluate() returns
  None outside box / non-finite / non-finite remodulation. Internal artifact
  schema 0.1.0. Graph trace resolves (producer train_pearcey_table.py::main
  confirmed exists, 2 consumers cusp_amplification/use_pearcey_table).
- WP5 L_MAX==48 confirmed at runtime. _certify_geometric_census: (a) count in
  {2,4}, (b) sum((-1)**morse_index)==sign(detA)-1 (0 for det>0, -2 for det<0).
  morse_index=eigvalsh(hessian)<0 count -> sign(mu). Called once (find_images
  reused). Passes silently on resolved non-degenerate censuses = byte-identical.

### Tests run GREEN this round
- test_lensing_levers.py: 47 passed, 1 xfailed (70s).
- test_lensing_geometry.py: 13 passed (16s).
- test_lensing_schwinger.py: 34 passed (188s).
- test_lensing_fast_path.py + test_lensing_batched_operator.py: 38 passed,
  4 skipped (138s).
- test_lensing_saddle_channels.py: 13 passed (25s).
- imports clean; L_MAX=48; pipeline_graph trace pearcey_table resolves.

### OPEN carried findings (NOT code defects; unchanged since prior review)
- INS-3-001 (trivial, LIBRARIAN): SPEC.md microlensing paragraph does NOT
  mention the registered pearcey_table product (asymmetric with
  lens_amplification_surrogate). Off-by-default+byte-identical -> default
  narrative stays accurate. Needs one Librarian sentence + spec_version bump.
  STILL OPEN (SPEC.md unchanged, 0.16.0).
- INS-3-002 (informational, DRIVER/ARCHITECT): WP5 kept L_MAX=48 (certified
  overlap ceiling) rather than relaxing toward the census-(b) floor; brief's
  ~13.9% served-fraction payoff intentionally forgone. Reconcile floor ledger.
- INS-3-003 (informational): _certify_geometric_census is STRICTER than
  geometry._check_image_census (refuses degenerate fold/cusp censuses the
  latter permits); value-preserving only because production resolution gate
  routes degenerate censuses to wave branch. Direct public geometric_amplification
  call on a degenerate config now raises LensDomainError earlier — fail-fast,
  off-production exception timing/type shift only.

### Carry-forward from 8e (not in 8f scope, unverified)
- INS-2-001 fold arm leading-order; INS-2-002 RefusalContract RED;
  INS-2-003 SPEC line-54 stale; INS-2-004 cusp-window shrink inert.
