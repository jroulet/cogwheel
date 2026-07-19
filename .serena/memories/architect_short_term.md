# Architect Short-Term Observations

Build 6 (neg-parity S1) attempt-6 FINAL plan, fresh consults (2026-07-18e):
- Verified in tree: geometry.py (parity split+saddle images+two-lobe crit),
  _schwinger.py (f_schwinger, W_CEILING_SCHWINGER=60, _CERTIFICATION_TOL=3e-10
  raw-t N-vs-2N, SchwingerCertificationError, _measure_warm_cost), operator.py
  (_saddle_grid dispatch resolved&&w>60->geometric else Schwinger,
  _saddle_mass_sheet_map SEPARATE, F_op/F_op_grid parity dispatch, select_branch
  BYTE-FROZEN no saddle arg). All coherent; Prof found NO code defects.
- Simplifier: TRIM to ZERO Coder WPs. Verify-only WP = bureaucratic relay =
  Inspector's job (invariants readable from 6 fn bodies; S1 deviation documented
  inline in _saddle_grid). _measure_warm_cost -> Test-Dev diagnostic one-liner.
  FINAL PLAN = empty work_packages + domain_test_descriptions for Test Developer.
- Prof CORRECTIONS to prior staked tols: G2 oracle band -> [0.5,55] 1e-10 (was
  45; dd holds 1e-10 to w~64 so 55 well inside), (55,60] raw-t self-consist 3e-10
  ONLY. G3 oracle MUST take PHYSICAL-frame (kappa,gamma,beta,y) inputs (beta!=0,
  both y comps!=0, kappa!=0) -> implicitly tests eigenframe rotation sign; 1e-14.
  G6 = DIRECT geometric_amplification vs Schwinger (NOT dispatch; w=13 never hits
  geometric), verify w*delta_min>5 first; 2-img w=13 + 4-img w=20; 5e-4. G5 w=1e-5
  & 5e-4 both fine (drift 1.3e-4<<5e-4, no subtraction). G1 index sum -2 confirmed
  both sets {(1,1)},{(0,1,1,1)}; no Morse-2 since tr A=2lam>0.
- 3 supplementary probes Prof added: dispatch-AND falsification (w=50 resolved ->
  Schwinger path, _saddle_grid vs f_schwinger 1e-14); ceiling boundary
  (f_schwinger(60.0) succeeds, 60.01 raises); L_S y-independence @ fixed w.
- Measurement (brief-required, NOT gate): _measure_warm_cost ms/point -> FINDINGS
  + envelope-surrogate todo via post-gate Librarian doc-sync.


Build 6 (neg-parity S1) attempt-6 VERIFY plan, fresh consults (2026-07-18d):
- CODE DELIVERED+coherent in tree: geometry.py saddle work (macro_matrix
  parity split, _centered_source_images saddle case, two-lobe critical utils),
  _schwinger.py (f_schwinger, N/2N raw-t certify 3e-10, W_CEILING=60 hard refuse,
  njit fastmath=False+py_func, analytic h', mod2pi), operator.py (_saddle_grid,
  _saddle_mass_sheet_map SEPARATE from byte-frozen _mass_sheet_map, F_op/F_op_grid
  parity dispatch; geometric takeover INSIDE _saddle_grid = resolved AND w>60).
- select_branch LEFT byte-frozen positive-parity (saddle decision lives in
  _saddle_grid; select_branch only consumed by channels=Build7). ACCEPTED S1
  deviation (Prof+Simplifier); Coder verify-WP must CONFIRM+DOCUMENT the split,
  not "fix" select_branch.
- Plan = ONE Coder verify-only WP (seam pins + one-shot _measure_warm_cost) +
  domain_test_descriptions for 7 gates (Test Dev authors). Simplifier: LEAN/LEAN/
  TRIM-zero-coder. List pins explicitly so Coder checks invariants not re-reads.
- Prof STAKED gate tolerances: G1 sum(-1)^n==-2 exact, sets {(1,1),(0,1,1,1)},
  resid<1e-7, gamma=1.3. G2 split: w[0.5,45] 1e-10; (45,55] 1e-8; (55,60]
  N-vs-2N self-consist 3e-10 only; grid gamma'={1.05,1.3,2.0} x y_eig={(0,0),
  (0.3,0.2),(0.7,0.5)}. G3 2D rotated-contour PHYSICAL-frame oracle 1e-14 @
  (w=3,y=(.4,.3),a=-.3,b=2.3) [catches eigenframe rot D1]. G4 delay-diff+flux-
  ratio kappa-invariant 1e-14; mass-sheet F_op identity (kappa=.3 vs kappa=0
  rescaled, two indep calls) 1e-13 @ w>=10. G5 |F| vs 1/sqrt(g^2-lam^2) rel
  5e-4 AND Morse phase intercept -pi/2 abs 5e-4, BOTH @ w=1e-5 (no drift sub).
  G6 2-img geom-vs-Schwinger 5e-4 @ w=13 + ADD 4-img config 5e-4 + select_branch
  returns 'wave' probe. G7 full suite + byte-identical frozen-literal dispatch
  probe. Structural: F010 py_func falsification; refusal-symmetry named errors;
  D3 y-independence @ w=30 two y same tol.



Build 6 (neg-parity S1) FINAL plan, fresh consults (2026-07-18c):
- Tree verified: geometry.py saddle work present+coherent (macro_matrix guard
  split, _centered_source_images saddle case, two-lobe critical_point/
  nearest_caustic_point). find_images = alias to parity-agnostic
  find_images_quartic. operator._mass_sheet_map STILL hard-refuses saddle
  (not lam>|gamma|); _schwinger.py absent. Dispatch fork unbuilt.
- Prof CORRECTIONS to draft tolerances: (1) N-vs-2N 3e-10 gate measured on RAW
  t-integral BEFORE 1/Gamma(iw/2) prefactor (else folds in the e^{pi w/4} you're
  certifying against). (6b) oracle rel 1e-10 only to w<=55; 55<w<=60 assert ONLY
  N-vs-2N self-consistency 3e-10 (mpmath+engine mutual agreement degrades into dd
  band >~45). (6c) 2D anchor 1e-14 not 5e-15. (6d) observables 1e-14, direct
  F-vs-reconstruction 1e-15 not 3e-16. (6e) Morse phase abs 2e-4 not 5e-4.
  Confirmed: dd@accumulation + phase mod2pi (~46 wraps @w60); analytic h'(t);
  ceiling 60 INSIDE evaluator; select_branch saddle gate = resolved AND w>60;
  new SchwingerCertificationError(RuntimeError); census 200 idx sum -2 sets
  {(1,1),(0,1,1,1)}; deep |F| rel 1e-3; geom-branch 5e-4 @w=13.
- Simplifier: 3-WP split LEAN keep separate. WP3 add SEPARATE
  _saddle_mass_sheet_map (0<lam<=|gamma| guard), do NOT refactor byte-frozen
  _mass_sheet_map. cancellation_exponent stays saddle-refusing & UNTOUCHED —
  select_branch saddle path takes w directly, never computes L. Give select_branch
  optional saddle flag so default positive-parity call byte-identical (channels
  Build 7 unaffected). Timing probe diagnostic-only, never gates control flow.


Build 6 (negative-parity S1) RE-PLAN, fresh consults (2026-07-18b):
- Prof sign-off confirms all tolerances below. Ceiling w>60 = UNCONDITIONAL
  hard refuse INSIDE _schwinger (evaluator never returns for w>60); dispatch
  decides geometric takeover. Internal certify gate = paired N-vs-2N GL rel
  3e-10 (3x the 1e-10 external oracle target, conservative bound). dd carried
  through the RUNNING ACCUMULATION of GL-node sum (each node float64 O(1);
  1/Gamma(iw/2) prefactor + reconstruct phase applied at END via _reduced_phase
  mod 2pi). h'(t) MUST be analytic closed-form (Simplifier watch), never finite-
  difference. Mass-sheet reduce+rotate+reconstruct lives in operator.py (WP3);
  _schwinger takes pure eigenframe (w, y_eig, g') with a=1-g'(NEG), b=1+g'.
- Exception dissent RESOLVED: keep NEW SchwingerCertificationError(RuntimeError)
  in _schwinger (Prof authority) over Simplifier's move-CancellationError-to-
  geometry (rejected: touches byte-frozen operator path + broad import surface).
- WP split 3 (Simplifier lean): WP1 geometry finalize/verify+wire find_images
  (indep), WP2 _schwinger (indep), WP3 operator dispatch+select_branch (dep WP2).
  No DATA_CONTRACTS artifact touched (internal compute). Timing = WP2 bounded
  one-shot warm measurement (prices envelope-surrogate todo), NOT a gate.

Build 6 (negative-parity S1: saddle geometry + Schwinger wave branch) plan
checkpoint (2026-07-18):
- geometry.py saddle work ALREADY in tree (uncommitted, not test-verified):
  macro_matrix parity split (lam<=0 + parity-boundary named refusals), two-lobe
  branch-param critical_point/_caustic_source/_coarse_squared_distances/
  nearest_caustic_point, saddle _centered_source_images. Positive-parity paths
  byte-frozen via branch defaults. WP1 = finalize/verify + wire find_images
  centered-saddle route, fix defects. NOT a pure audit (Simplifier: lean, keep
  separate from operator dispatch).
- NEW _schwinger.py (WP2): exact 1D Schwinger dd quadrature. Prof mapping:
  a=1-kappa-gamma (NEG, e1 axis), b=1-kappa+gamma; y_eig = Re/Im(exp(-i*beta)*
  (y_scaled[0]+i y_scaled[1])); mass-sheet reduce then reconstruct
  (1/lam)exp[iw(ln lam/2 - kappa|y_scaled|^2/2)]. dd at accumulation +
  1/Gamma(iw/2) prefactor multiply ONLY (single y-indep channel L_S=pi w/4);
  REUSE _hyp1f1 dd phase-reduction mod 2pi (fatal at w~60, ~22 wraps). Certify
  via N-vs-2N Gauss-Legendre per panel, refuse >1e-11; ceiling W=60 hard refuse.
  ONE new exception SchwingerCertificationError(RuntimeError) (import cycle:
  _schwinger cannot import operator; RuntimeError base mirrors CancellationError
  so Build 7 except-clauses generalize).
- operator.py dispatch (WP3, dep WP2): parity check BEFORE _mass_sheet_map;
  saddle -> schwinger; positive-parity path BYTE-frozen; select_branch saddle
  gate = resolved AND w>60 (replaces L>L_MAX). NO channels.py (Build 7).
- Tolerances carry Prof authority: census 200+20 srcs index sum -2 sets
  {(1,1),(0,1,1,1)}; schwinger-vs-mpmath-dev-oracle 1e-10 (w<=50) grid incl
  g'=1.05,w<=60; 2D rotated-contour anchor 5e-15; mass-sheet 3e-16 + observables
  1e-14; deep |F|=1/sqrt(gamma^2-lam^2) rel 1e-3@w=1e-4 AND Morse phase intercept
  -pi/2 abs 5e-4; geom-branch 5e-4@w=13(dtau=0.385); F004 exact-boundary points
  (0.5/0.5,0.75/0.25,0/1.0); F010 py_func falsification for njit core.

Build 5 (lensed marginalization) plan checkpoint (2026-07-18):
- Professor: ship FULL extrinsic marg (CoherentScoreHM) directly; distance-only
  tier does NOT move wall-clock (per-proposal XPHM strain cost unchanged). HM
  (not QAS) respects the 22-only phase constraint structurally.
- Lensed timeseries seam = override `_get_dh_hh_timeshift`: per image a, modulate
  bin templates h_mpb by k0_a[b]*exp(i·2π·dt_a·f_b) (exp = exact timeseries shift
  via existing _d_h_weights; dt_a from _image_delays), contract with UNCHANGED
  _d_h_weights, SUM over images. Norm: multiply _h_h_weights by |F(w_b)|^2 where
  F_b = Σ_a k0_a[b] exp(i·2π·dt_a·f_b). Coherent score consumed UNCHANGED.
- Simplifier: A2 composition (internal LensedRelativeBinningLikelihood for
  `_amplification_coefficients`; wasted moment build is one-time) over mixin
  refactor of validated 1500-line likelihood.py. Reuse LensedPosterior as-is
  (already maps LensDomainError/CancellationError->-inf). New module
  cogwheel/lensing/marginalized_likelihood.py. Blob distance key = d_app (defer
  physical d_L=d_app*sqrt(mu_macro) to post-analysis). 2 new classes total.
- Refusals evaluated ONCE up front per proposal, propagate to -inf; never averaged
  inside QMC. Tolerances: full-marg |lnL_marg-oracle|<=0.3 nats (90th pct<=0.2),
  importance-sampling oracle (1e5 draws) not full grid; 8 seeded configs C1-C8;
  conditional draws = round-trip consistency (max lnL_full>=lnL_marg-0.3).
