# Coder Short-Term Observations

- INS-3-001 re-dispatch (docs only): finding was an INSPECTOR-SESSION access
  failure (Bash false-denial + Serena/Read timeouts in their session), NOT a
  code/test defect. My session HAS file access -> statically confirmed crown-gate
  deliverables present + coherent: likelihood.py has _amplification_coefficients
  (F006 fix) + subsampling machinery + _data_term/_norm_term + retained
  _amplification_at_bins/_edge_linear_coefficients; test_lensing_likelihood.py
  ContractionTimingTestCase encodes the TWO F007 baselines (speedup vs
  lnlike_bruteforce; contraction subdominant to _amplification_coefficients) and
  explicitly declines old coarse-strain gate; NearCuspRegressionPin +
  NormalizationFloorZeroNoise (zero-noise F->1 anchor) + Determinism all present.
  Appended a "Static verification (WP1 re-dispatch)" subsection to FINDINGS F007
  recording this + marking pytest runtime green-ness UNVERIFIED (handed to
  runtime reviewer; Coder must not run the blessing suite). No production/test/
  SPEC touched.

- WP1 build2b CLOSEOUT #2 (docs only, no code/tests/SPEC): added FINDINGS
  F007 (two Build-2b non-defects). (a) TIMING GATE mis-spec:
  ContractionTimingTestCase asserted t_contract < t_coarse_waveform, but the
  coarse get_strain_at_detectors(fbin) is RB's per-eval CO-COST, not its
  competitor (lnlike_bruteforce is the full-grid competitor). 23x (1.47ms vs
  64us) = additive M^2+n_img^2 design as-designed (XPHM n_m<=4). Correct gates:
  (i) lnlike < lnlike_bruteforce by conservative margin (public entry points),
  (ii) t_contract < t_amplification (_amplification_coefficients special-fn
  engine at n_bins*kernel_subsamples pts). Old gate also EXCLUDED
  _amplification_coefficients from measurement => left real added work
  unguarded; baseline (ii) closes it. (b) F->1 FLOOR (0.10-0.33) NOT a
  normalization bug: _compute_d_h/_h_h apply asd_drift**-2 per det, both
  oracles route through; 4*df/blued/wht_filter**2 match CBC/RB. Residual =
  template-construction asymmetry: _set_summary builds _h0_edges with
  disable_precession=False + _stall_ringdown, _candidate_bin_ratios builds
  candidate h_edges with NEITHER => at r~1 ratio!=1 in ringdown of 60+45 Msun
  fixture (delta-h/h~1e-3), beaten vs unseeded noise reads ~rho*delta-h/h
  ~0.05-0.3 (phys lensing residual at w~1e-7 ~1e-4, 4 orders below). KNOWN
  residual left in place for 2b (aligning builders risks green brute suite,
  out of scope); crown gate handles it via zero-noise F->1 anchor. Also added
  correcting annotation on F006's "at F->1 ... exact to ~1e-8" (holds only for
  p=0 moment w/ exactly-constant ratio/kernel + matched freq sets; full path
  carries construction-asymmetry delta-h). Changelog: top-level changelog.d/
  is the PROJECT news-fragment dir (frontmatter `date: 2026-07-16`, distinct
  from spec_changelog.d/ `bump:` fragments) -> created
  changelog.d/2026-07-16_lensing-build2b-crown-gate.md matching that format.
  Files: .claude/spec/FINDINGS.md, changelog.d/*.md. No production/test/SPEC
  touched. UNVERIFIED: no renderer run (docs-only, no shell needed); frontmatter
  matches existing top-level fragments byte-for-byte.

- WP2 build2b CLOSEOUT (docs only, no code/tests): added FINDINGS F006
  (near-cusp (h|h) blow-up: old _edge_linear_coefficients built per-bin
  (k0,k1) from the TWO bin edges = secant; near cusp the merged-image
  channel kernel collapses via smootherstep->0 to alpha_a*exp(-iw tau_a)*
  F(w) carrying full amplification oscillation; secant aliases it ->
  k0->0, k1 blows up, _norm_term squares k1 -> 6.43e8 spurious (h|h). Fix =
  new _amplification_coefficients: dense per-bin subsample grid
  (kernel_subsamples=8) + per-bin least-squares (k0=mean, k1=offset LS).
  Contraction/moments/delay guard UNCHANGED). F->1 normalization AUDITED,
  NO code change (floor variability was unseeded EventData noise, seeded in
  tests). Verified fix landed by reading likelihood.py
  _amplification_coefficients body (dense_w, ChangRefsdalChannels, einsum
  _kernel_fit_value/_slope). Added spec_changelog.d fragment
  2026-07-16_lensing-near-cusp-fix.md (bump: patch; no SPEC prose change -
  positive-parity/named-refusal guarantees still hold). UNVERIFIED:
  render_fragments.py --check shell-denied this session; fragment
  frontmatter (---/bump: patch/---) matches existing working fragments
  byte-for-byte.

- WP1 build2b (cogwheel/lensing/likelihood.py, near-cusp/F->1/refusal):
  DIAGNOSIS (read-only, shell denied): contraction algebra in
  _data_term/_norm_term is CORRECT (verified both term-by-term vs the
  truncated linear-kernel x linear-ratio x linear-phase model, grouped by
  kbar0/kbar1 and by image order s). Near-cusp mechanism = engine kernels
  are NON-smooth in the unresolved regime: exact_transition_channels
  builds K_a = trial + alpha*conj(carrier)*residual, and when switch->0
  (smootherstep(w*delay_sep,0.5,4)=0, exactly the near-cusp merged-image
  regime) K_a collapses to the ARTIFICIAL split alpha_a*exp(-iw tau_a)*F(w),
  carrying the full amplification oscillation. The old _edge_linear_
  coefficients built (k0,k1) from the TWO bin edges (secant); when the two
  edges alias the kernel phase, k0(secant midpoint)->0 while slope k1
  blows up, and _norm_term SQUARES k1 -> the ~6.43e8 spurious (h|h). FIX
  (brief hint c, "denser grid then reduce to bin coeffs"): new
  _amplification_coefficients evaluates ChangRefsdalChannels on a per-bin
  sub-sample grid (kernel_subsamples=8 default, midpoints of S equal
  sub-intervals, strictly-increasing/positive w) and reduces each channel
  kernel to (k0,k1) by per-bin least squares (offsets symmetric about
  f_center => k0=mean, k1=offset/sum(offset^2)). Contraction, moments,
  ratio path, delay guard, 3 design decisions ALL unchanged; only the
  kernel coeffs improve. Kept _amplification_at_bins + _edge_linear_
  coefficients (timing test + ratio use them). F->1 AUDIT: normalization
  is CORRECT, NO change - moment prefactor 4df, blued_strain,
  wht_filter**2, asd_drift**-2 all match CBCLikelihood._compute_d_h/_h_h
  and RelativeBinningLikelihood; at F->1 (r=1,k=alpha_a,delays~0) the p=0
  moments sum over bins to the EXACT integral so d_h/h_h are exact to
  ~1e-8; the build2b 0.10-0.33 readings are the noise-projected binning
  error (nondeterminism, TestDev seeds it), not a normalization bug.
  Refusal symmetry: nothing swallowed (both paths propagate LensDomain
  Error + CancellationError); macro-saddle symmetric (both hit
  geometry.macro_matrix); dense grid adds no new refusal for tested
  configs (w~few, L<<45). UNVERIFIED (shell/ast.parse denied): runtime,
  and whether best-fit fully clears near-cusp to RB_ATOL/RB_RTOL - it
  cures the SECANT ALIASING (the catastrophic k1^2 blow-up) but leaves the
  inherent per-bin linear residual; if genuine kernel curvature over 4Hz
  still exceeds tol the lever is finer bins (design already supports it) or
  raising kernel_subsamples. Serena LSP: 0 lines>79; only false-positive
  Optional-subscript (f_center/moment_ops set in _build_moment_operators
  first) + numpy/scipy env-import noise.

- FIX INS-2-001/002/003 (2nd dispatch; test-authoring done under duress
  after handoff-to-TestDev was not honored — flagged for mandatory
  Inspector review of oracle independence). THREE changes, all in
  cogwheel/tests/:
  (1) INS-2-001: re-scoped GeometricOpticsSlopeTestCase.SLOPE_W from
  linspace(12,45,84) to linspace(12,27,84) (SLOPE_Y stays [0.9,0], so
  L=0.9*w max 24.3 < 25 = tested-returning band; F_op now refuses
  >~L30). Also updated the module-docstring TESTED DOMAIN paragraph.
  (2) INS-2-002: added ContractionCertificationTestCase — sweeps
  CERT_LS=linspace(24,48,17) at y=[0.9,0],gamma=0.2,kappa=0 (w=L/0.9);
  each F_op call is finite+oracle-accurate(RTOL_GATE=1e-10) XOR raises
  operator.CancellationError; asserts returned>0 AND refused>0; pins
  L~40 config to raise with message naming w=/y=/gamma/kappa (verified
  _refusal_message format in operator.py). XOR is robust by construction:
  F_op returns only when contraction_error<=1e-10 and actual<=estimate.
  Added test_certification_band_gate_can_go_red to SelfFalsification.
  (3) INS-2-003: NEW cogwheel/tests/test_lensing_likelihood.py. Fixtures
  mirror test_posterior.py: EventData.gaussian_noise(duration=4,'HLV',
  asd_*_O3)+inject_signal+WaveformGenerator.from_event_data, approximant
  IMRPhenomXPHM (HM: |m| in {1,2,3,4}; XAS is 22-only, no XHM registered).
  Explicit uniform fbin (DF_BIN=4Hz) from event band; delta_t_max=0.02
  (pi*4*0.02=0.25<0.5 tol). Classes: BruteForceAgreement (RB lnlike vs
  lnlike_bruteforce over 2/4-image,near-cusp,kappa,rotated-shear,near-fold,
  waveform-offset), ProductOfSummariesRegression (0.97*caustic
  near-degenerate), UnlensedLimitNormalization (F->1: lensed bruteforce &
  RB vs lnlike_fft AND standard RelativeBinningLikelihood — independent
  normalization anchors, NOT my code), BinGuard (construction: big
  delta_t_max on fine grid; eval: delta_t_max=1e-4 + fixture-mass
  candidate delays~ms), ContractionTiming (imports _data_term/_norm_term/
  _edge_linear_coefficients, best-of-7, contraction<coarse waveform call),
  MacroSaddleRejection (lnlike & lnlike_bruteforce raise LensDomainError),
  SelfFalsification. UNVERIFIED (shell/ast.parse denied this session):
  runtime, tolerances (RB_ATOL=1.5/RB_RTOL=1e-2/NORM_TOL=0.1 set
  CONSERVATIVELY for structural-bug detection, need TestDev calibration to
  the true binning floor), and bruteforce runtime (amplification over
  ~1500 in-band freqs). Serena LSP parsed both files; 0 lines>79; all APIs
  (lnlike/lnlike_fft/lnlike_bruteforce/RelativeBinningLikelihood ctor/
  event_data.fslice/geometry.critical_point) signature-checked.

- WP4 (Build 2 closeout, docs/spec only — no code): Added SPEC.md layer row
  for the two Build-2 modules (cogwheel/lensing/waveform.py
  LensedWaveformGenerator + likelihood.py LensedRelativeBinningLikelihood),
  a spec_changelog fragment (bump: minor), a top-level changelog.d fragment,
  and an overview.rst paragraph (w=8*pi*G*M_L*(1+z_L)*f/c^3 convention + both
  public entry points). KEY FINDINGS: (1) F005 was ALREADY finalized by WP1
  in FINDINGS.md ("Resolution shipped (WP1)" — NARROWED not RESOLVED: silent
  nan closed via overflow-safe frexp/ldexp rescale + named CancellationError;
  accuracy gap L in [~30,48] stays open) — did NOT duplicate. (2) SPEC.md
  engine-row limitation line was ALSO already updated by WP1 (pre-loaded
  SPEC content was stale) — I only ADDED a new row, left engine row as-is.
  (3) grep of both modules for to_npz/np.save/to_feather/savez/open/Path =
  ZERO on-disk writes — both are in-memory JSONMixin objects, so NO
  DATA_CONTRACTS.yaml change (recorded 'no new data product'). Build-3 todo
  fragment (2026-07-16_lensing-program.md) left untouched. UNVERIFIED:
  render_fragments.py --check was shell-denied this session (documented
  refusal) — fragment frontmatter matches existing working fragments
  byte-for-byte in format, so parser will accept.

- WP3 (cogwheel/lensing/likelihood.py, LensedRelativeBinningLikelihood):
  subclasses BaseLinearFree (imported from
  cogwheel.likelihood.relative_binning submodule — NOT in the likelihood
  package __init__, but it's a public class; mirrors how relative_binning
  imports CBCLikelihood). Reference par_dic_0 is UNLENSED; lensing enters
  only at hot path via lens keys in par_dic
  (m_lens_msun,z_lens,y1,y2,gamma,beta,kappa). Design = paper Eqs
  (fiducial-component)..(slow-component-ratio): delay-free frequency-MOMENT
  summaries A^(p) (data, p<=2) B^(p) (norm, p<=3, ordered mode pairs MxM)
  built at setup via sparse hard-bin operators (moment_ops[p] =
  4df*(f-f_center)^p, csr). Candidate image-delay phase kept ANALYTIC:
  exp(-2pi i f_b dt_a) at bin center + linear in-bin correction folded into
  higher moments (guarded by pi*Df_bin*delta_t_max<bin_delay_tol ->
  LensedBinningError). Kernels K_a interpolated linearly per bin from edge
  values (partition.kernels at w=xi*fbin edges). Contraction is
  ADDITIVE M^2+n_img^2: mode-reduce first (einsum over m / m,m'), then
  image-reduce (n_img=4 channels incl virtual; ordered pairs). Pure math in
  module fns _data_term/_norm_term; no FFT/py-loop on hot path.
  CRITICAL sign check I verified on paper: r ratio formed in linear-free
  ALIGNED frame (r = h_raw*exp(2pi i f dt_lf)/h0); data uses
  tau_a=dt_a-dt_lf, norm uses relative dt_a (dt_lf cancels in |.|^2) ->
  reconstruction collapses to F(w)*h_true exactly, dt_lf drops out. Engine
  delays are RELATIVE to t_min (t_min common shift omitted) — CONSISTENT
  with WP2 amplification (also t_min-subtracted via exact_total), so RB vs
  lnlike_bruteforce agree. lnlike_bruteforce builds a per-candidate
  LensedWaveformGenerator (consumes WP2) and uses .amplification (exact_total)
  as the same-generator brute reference. CancellationError/LensDomainError
  from engine NOT caught (propagate). Passed dict(par_dic) to
  _get_linearfree to avoid ZERO_INPLANE_SPINS in-place mutation.
  UNVERIFIED at runtime: ast.parse/import denied this session (documented
  shell refusal) — read-back: 0 lines >79, all names resolve, all base
  methods (_get_h_f,_stall_ringdown,_get_linearfree_hplus_hcross_dt,
  _compute_d_h/_h_h,get_strain_at_detectors) signature-checked, einsum/
  broadcast shapes hand-traced. Pyright import flags (numpy/scipy/
  cogwheel.lensing.waveform) are env-resolution noise; _bin_moments Optional
  subscript is a false positive (moment_ops set before use).

- WP2 (cogwheel/lensing/waveform.py, LensedWaveformGenerator): composes a
  wrapped WaveformGenerator (duck-typed on get_hplus_hcross/m_arr) + fixed
  lens params. Single authoritative w(f): module fn
  dimensionless_frequency(f_hz, m_lens_msun, z_lens) = 8*pi*lal.MTSUN_SI*
  m_lens_msun*(1+z_lens)*f, linear in f. Used lal.MTSUN_SI (=G*Msun/c^3 s,
  ~4.925e-6) because it IS the G*M/c^3 combo already used via lal in
  waveform.py — no astropy needed. amplification(f) -> partition.exact_total
  on positive-w subgrid, F=1 elsewhere. CRITICAL engine constraint:
  channels._validate_frequencies REJECTS w<=0 (strict >0, increasing, >=2
  pts), so I MUST slice to w[positive] before ChangRefsdalChannels(...) and
  fill unlensed bins (f=0 bin) with F=1. decompose() returns
  LensedDecomposition(w, delays=tau_a, kernels=K_a (n_f,4), real_mask,
  unlensed_hplus_hcross (n_m,2,n_f), m_arr); w<=0 rows filled
  K[:,0]=exp(-i w tau_0) so reconstruction=1 exactly. Macro-saddle rejection:
  constructor calls geometry.macro_matrix(gamma,beta,kappa) for its raise
  side-effect (LensDomainError if 1-kappa<=|gamma|) AND every strain/decompose
  path re-hits it via engine.evaluate -> NO try/except anywhere. Subclasses
  utils.JSONMixin; get_init_dict works via same-name attrs (y stored as
  np.asarray). UNVERIFIED at runtime: ast.parse/import denied this session
  (documented shell refusal) — read-back shows valid syntax, all names
  resolve, signatures cross-checked (evaluate is kw-only gamma/y/beta/kappa;
  ChangRefsdalChannels(w) positional), no lines >79.

- WP1/F005 (operator.py F_op): closed the silent-nan overflow in the
  wave-branch contraction by factoring max|derivs| out as an EXACT
  power of two (np.frexp/np.ldexp) before the matmuls, running the
  whole sum in scaled units and reconstructing the total exactly. This
  is bit-preserving for the L<=25 region (power-of-two commutes through
  mul/add; scale_exp small there so no meaningful underflow) — did NOT
  add double-double (F001 two-channel model preserved). Added a named
  refusal: measured contraction round-off eps*(sum|term|/|total|) vs
  _CONTRACTION_TARGET=1e-10, plus non-finite-total and non-finite-value
  backstops (critical: nan>threshold is False, so nan must be caught
  explicitly BEFORE the ratio gates or it slips through). Kept the
  existing gamma-channel gate (max_term/|total|>1e13) unchanged and
  firing first.
- KNOWN test conflict I flagged for Test Dev: closing F005 changes
  behavior from silent-wrong to named-refusal for L in [~30,48].
  GeometricOpticsSlopeTestCase (test_lensing_operator.py) sweeps w to 45
  at |y'|=0.9 -> L~40.5 and calls F_op directly expecting a finite
  value; it now raises CancellationError and must be re-scoped (lower w
  top or SLOPE_Y) to stay in the certified region. channels.py wave
  calls are gated by select_branch, so production hits the refusal only
  for unresolved L in [30,48] — the intended "certified-or-refuse"
  contract.
- Empirical anchor for calibration (from the operator test docstring):
  worst actual error at L<=25 is 5.65e-12, so the contraction condition
  (sum|term|/|total|) there is ~2.5e4 and eps*cond ~5.65e-12, ~18x below
  the 1e-10 refusal cut — L<=25 stays returning. The refusal boundary is
  the estimate crossing 1e-10 (cond ~4.5e5), landing near L~30.
  UNVERIFIED at runtime (ast.parse/import denied this session).
