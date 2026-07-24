# Test Dev Short-Term Observations

## test_lensing_exterior_windows.py EXTENDED (Build 8h-b3 Specs 10-11) — 2026-07-23
- Added WholeInteriorSacrcTestCase(6) + WholeInteriorSacrcLiteralBarTestCase
  (1 xfail) + TubeByteIdentityTestCase(2) + 2 self-falsif -> full file now
  68 passed + 2 xfailed (was 58+1). Neighbors green: geometry 13, ghost
  36+1xfail. Full file 279s. conda python
  /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python. TEST-ONLY (no
  production edits).
- SPEC10 SACR-C interior vs far-field label. Config RHO_C=0.25, band_half
  0.03, half_rho 0.03, half_theta 0.15, ng4/nr5/nt5/w6, w[0.05,20], seed=1,
  5 held-out. Charts lru_cached (gamma,definition)->6 unique builds ~140s.
  eps via surrogate._evaluate_chart(chart,g,rho,th,eta=0.1,theta=0.0,log_w)
  DIRECT (bypasses serve guards, isolates label conditioning). SACR-C
  normalized by max|partition.envelope|, far-field by max|exact_total| with
  ch.farfield_envelope_from_partition(part). MEASURED (deterministic):
  g0.40 SACR-C 0.023/far 0.124 img=4; g0.65 0.072/33.2 img=4; g0.90
  0.100/2.49e11 img=2 (crown = degenerating-caustic EDGE, only 2 images!).
  GATES budget-independent: far-field-fails FAR_FAIL_FLOOR=1e-2 (all far>);
  SACR-C RELAX bar 1.5e-1 for 0.40/0.65, CROWN_BAR 1.5e-1 for 0.90 (0.100
  measured, order-of-mag milestone per Prof R4 NOT 1e-3); CONTRAST
  far/sac>2.0 at each g AND grows (crown>>0.40) = representational-win
  reachable-red. Literal 1e-3 UNREACHABLE at budget -> @expectedFailure
  tripwire (record_comparison BEFORE assert). NOTE SACR-C eps 0.023 at 0.40
  is ITSELF >FAR_FAIL_FLOOR(1e-2) so the FLOOR alone doesn't separate labels
  — the CONTRAST test is the real discriminator; self-falsif
  test_equal_labels sac/sac=1.0<2.0 proves it.
- SPEC10 Prof R4 NO cusp carve-out: cusp-aligned (theta_c=0) interior tile
  from_engine(INTERIOR_SACR_C) BUILDS clean + serves finite envelope (no
  CarrierDiscontinuityError). tau_c continuity oracle: INDEPENDENT engine
  grid of part.critical_source (2-vec) over tile nodes, max adjacent-node
  jump < surrogate._CARRIER_FLIP_FRACTION(=0.5)*_caustic_reach(g). Measured
  cusp tile g0.40 jump 0.3527 vs reach 1.0328 frac 0.3415 (<0.5, single
  basin). GUARD TEETH: surrogate._assert_carrier_continuity(grid,gamma_grid,
  (ng,nr,nt)) — synthetic all-zeros array passes; array with one axis-slice
  hopped 2*reach RAISES CarrierDiscontinuityError. GOTCHA: could NOT reach a
  real physical straddling tile — swept cusp-axis/diagonal/wide-theta/rho-
  span tiles, ALL build single-basin (0.5*reach flip bar is large; astroid
  basins smooth). So teeth demonstrated via SYNTHETIC flipped array fed to
  the production guard directly (deterministic, clean).
- SPEC11 tube byte-identity HARD FENCE. Load HEAD surrogate.py via
  subprocess git show HEAD:cogwheel/lensing/surrogate.py -> exec into
  types.ModuleType('surrogate_head_byteident') registered in sys.modules
  (lru_cache maxsize=1). HEAD imports (channels/geometry names) resolve to
  WORKING-TREE copies — fine, tube serve is pure spline interp, never calls
  engine, so isolates CHART+SERVE code from the (changed, separately-tested)
  engine. Synthetic deterministic TubeChart via _synthetic_tube_chart(module,
  scale): closed-form cos/sin envelope over (gamma,u=sqrt(eta),theta,log_w),
  np.moveaxis(...,3,0) to (n_w,n_g,n_u,n_t). Build under BOTH modules, wrap
  in module.LensAmplificationSurrogate([chart],{'kind':...}).serve(w,gamma=,
  y1=,y2=,beta=,eta=,theta=,image_count=). 30-query sweep max|diff|==0.0
  EXACT; also real_coeffs/imag_coeffs (NOT real_coeff) max|diff|==0.0. Self-
  falsif: HEAD chart scale=1+1e-6 -> served diff>0 (teeth). NOTE only
  surrogate.py/surrogate_training.py/channels.py/geometry.py/likelihood.py
  changed this build vs HEAD; end-to-end engine-driven tube build can't be
  byte-identical (engine changed) so test uses SYNTHETIC chart to fence the
  tube CODE, not engine numerics.
- API PINS: from_engine(*,gamma_range,rho_range,theta_c_range,w_range,
  n_gamma,n_rho,n_theta,w_nodes_per_decade,definition) needs n>=4 on cubic
  axes; INTERIOR_SACR_C stores partition.envelope, else
  farfield_envelope_from_partition. surrogate._log_w_grid(w_range,wnpd),
  _caustic_reach(g), _from_caustic_fixed(g,rho,theta_rad). TubeChart.
  from_values(*,gamma_grid,u_grid,theta_grid,log_w_grid,envelope_real,
  envelope_imag,image_count,parity,eta_floor,eta_max,cusp_windows). serve
  returns (E,served,definition). Plots -> output/{sacrc_interior_label_
  contrast,tube_byte_identity_diff}.png.


## test_lensing_exterior_windows.py EXTENDED (Build 8h-b3 Specs 7-9) — 2026-07-23
- Added 3 classes + 3 self-falsification tests to existing suite -> 58 passed
  + 1 xfailed (was 42+1). Neighbors green: test_lensing_geometry 13,
  test_lensing_ghost 36+1xfail. conda python
  /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python. Test-only (no
  production edits). Full suite runtime ~137s.
- SPEC7 ReprovisionNodeCountTestCase (already present from prior shard, kept).
  Added self-falsif test_loosened_reprovision_bar: loosening bar 1e-3->1e-2
  lowers min-accepted n 4->2, eps(2)=5e-3>1e-3 (teeth). REPROV_EPS_CURVE
  crosses bar between n=4(8e-4 clears) and n=3(2e-3 fails); N_rec=4.
- SPEC8 InteriorDirectionalAdmission (S2-1/WP6). Fixtures: BAND(0.45,0.55)
  mid0.5, reach=_caustic_reach(0.5)=sqrt2=1.4142, eta_max0.05.
  adm=st._interior_admission(BAND,1,reach,cfg). admits((rho,theta),(half_rho,
  half_theta)) uses CAUSTIC-FIXED (rho,theta), rho_outer=rho+half_rho probed.
  *** ANISOTROPIC GAIN: rho=0.40 at theta=0(fat, cusp axis, dir radius 0.5285)
  admits True + find_images=4; at theta=45(thin diag, radius 0.322) admits
  False + find_images=2. old isotropic old_admit_rho=(inradius0.5000-eta_max)/
  reach=0.318 < 0.40 -> old inscribed disk WRONGLY rejected the gain (band-edge
  waste). SWEEP theta{15,30,60,75,105}: rin=0.5*bnd admits True + 4imgs ALL
  band gammas; rout=1.10*bnd admits False + tightest-gamma(argmin r_caustic)
  find_images=2. TUBE SHELL: rho=0.45 theta=0 radially interior(0.45<0.5285) &
  find_images=4 BUT nearest caustic 0.0331<eta_max0.05 -> admits False (keys
  off NEAREST dist not radial gap). Tiles: st._farfield_interior_tiles(1.0,5,
  admission=,cusp_angles=st._cusp_source_angles(0.5,n)) -> 20 tiles, 4 cusps
  {-pi/2,~0,pi/2,pi}. Self-falsif: dataclasses.replace(adm, rho_boundary=const
  inradius/reach=0.3536) flips fat 0.40 admit True->False.
- SPEC9 SaddleLobeAdmission (S2-2/WP7). gamma=1.5>1 => TWO deltoid lobes.
  cfg=dataclasses.replace(TrainingConfig(),eta_max=0.02) (default 0.05 gives
  saddle_lobes_zero_admission: centroid nearest-caustic ~0.030<0.05). lobes=
  st._saddle_lobe_admissions((1.4,1.6),cfg); centroids ~(-1.579,0) & (1.579,0).
  lobe.admits(center,half) uses LOBE-LOCAL (rho_lobe,theta_local): helper
  _lobe_local inverts via lobe._r_deltoid. centroidA admits A True / B False;
  origin(corridor) admits both False. Morse via _signed_morse_sum(gamma,src)=
  (n_images, sum sign(magnification)) using geometry.macro_matrix+find_images+
  magnification (INDEPENDENT oracle): centroid=(4,-2), origin=(2,-2) [4->2
  across caustic, neg-parity sum=-2 both]. winding: st._winding_number(loop-
  centroid) ±1 self / 0 other for every band loop. Tiles per lobe: st.
  _lobe_cusp_source_angles(1.5,st._SADDLE_LOBE_CENTERS[k],centroid,n)=3 cusps;
  st._lobe_interior_tiles(lobe,cusps,5) -> 26/16 tiles. Self-falsif:
  mock.patch.object(st,'_winding_number',return_value=0.0) flips centroidA
  admit True->False.
- STRADDLE gotcha: raw near-zero cusp (-1.2e-31) vs its (angle+pi)%2pi-pi
  remapped tile edge differ by ~1e-16 -> naive strict straddle miscounts 1
  (interior) / 2 (lobe1). Use TOL_STRADDLE=1e-9 in _straddles_ray (worst real
  overlaps 1.2e-31 & 9.5e-17, far below tol). lobe0 genuinely 0 straddle.
- API PINS: geometry.r_caustic(gamma,theta,*,kappa=0.0,n_sample=720);
  macro_matrix(gamma,beta=0,kappa=0); find_images(source,matrix);
  magnification(image,matrix). st._caustic_inradius(g,parity,n)->(inradius,
  encloses_origin). surrogate._from_caustic_fixed(gamma,rho,theta_rad),
  _to_caustic_fixed(gamma,y1,y2), _caustic_reach(gamma). _InteriorAdmission &
  _SaddleLobeAdmission are frozen(eq=False) dataclasses -> dataclasses.replace
  works for mutation. insert_after_symbol/insert_before_symbol param is
  name_path NOT name_path_pattern. 3 plots -> output/{interior_admission_map,
  saddle_lobe_admission_map,reprovision_eps_vs_wnodes}.png.

## test_lensing_exterior_windows.py EXTENDED (Build 8h-b3 Specs 4-6) — 2026-07-23
- Added 3 classes to existing suite -> 42 passed + 1 xfailed (was 25).
  Neighbors green: test_lensing_geometry 13, test_lensing_ghost 36+1xfail.
  conda python /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.
- SPEC4 MidWindowGhost (ghost helpful/harmful). E=F-ppGO obtained via
  ch.farfield_envelope_from_partition(part, ch.FARFIELD_KERNEL_SUM) (==F-ppGO,
  tau_c=0). Oracle G from geometry.ghost_kernel: G=contr.kernel*exp(1j*w*
  contr.delay) — BIT-IDENTICAL to ch.farfield_ghost_term (verified formula).
  HELPFUL fold-annulus gamma=0.4, 45deg, rho{1.4,1.6}, w[3,40]: gate
  w_min*Im tau_c >=2 PASSES (3.38/4.57), |G|/maxF in(1e-3,0.1). HARMFUL cusp
  gamma=0.4 rho=1.15 theta{0.2,0.5,1.0} w[3,20]: gate ~0.01-0.08 REFUSES,
  production E-G grows residual 1.73-1.82x (>=1.5). *** KEY FINDING: at
  gate-PASSING gamma=0.4 configs production's subtracted ghost is
  ANTI-ALIGNED — E-G INFLATES ~2x (sub ratio 2.06) while E+G shrinks ~3x
  (add 0.34). Sign FLIPS with gamma: gamma=0.2 E-G helps (0.27) but gate
  refuses there (<2). So gate & ghost-sign are internally inconsistent this
  build; spec's literal helpful "E-G<=|E|/3 at gate-pass" is UNMET ->
  carried as @unittest.expectedFailure tripwire (xfails now, flips RED
  unexpected-success when production fixes sign; record_comparison BEFORE
  the failing assert). Companion PASS test pins add<0.5 & sub>1.5.
- SPEC5 TagContract. Tags: FARFIELD_DIFFRACTIVE='farfield_diffractive_bare_
  total', FARFIELD_KERNEL_SUM='farfield_full_kernel_sum'(=v2), FARFIELD_
  KERNEL_SUM_MINUS_GHOST='farfield_kernel_sum_minus_ghost', INTERIOR_SACR_C=
  'interior_sacr_c_envelope'. ch.KNOWN_FARFIELD_DEFINITIONS=3 far-field;
  surrogate._KNOWN_ENVELOPE_DEFINITIONS=all4. Matched serve err: diffractive
  0.0, kernel_sum 2.4e-15 (<1e-3). MINUS_GHOST route needs envelope+ghost on
  gate-pass config (gamma=0.5 _eigenframe_source(1.5,40) w[1.5,40]).
  CROSS-SERVE (Prof R3): diffractive<->kernel_sum both dirs err=32.04 (>>1e-3,
  gate TAG_MISMATCH_FLOOR=1e-2). FAIL-FAST: build multi-chart npz via
  surrogate._chart_to_npz(chart,i) merged into one dict; _chart_from_npz(data,i)
  preserves each tag. Tamper meta json (json.loads(str(data['chart0_meta'])),
  set 'bogus_v1' / pop key) -> _chart_from_npz raises ValueError. PROVE before
  numerics: mock.patch.object(surrogate.FarFieldChart,'_assemble') spy ->
  call_count==0 on bad tags, ==1 on good control (_validate_farfield_definition
  runs BEFORE FarFieldChart._assemble in _chart_from_npz). Helper
  _make_farfield_chart(tag, n=4): FarFieldChart.from_values(gamma_grid,rho_grid,
  theta_c_grid,log_w_grid,envelope_real,envelope_imag,image_count=2,parity=1,
  envelope_definition=tag).
- SPEC6 FixedWindowContainment. box=st.PriorBox.from_prior_classes();
  cfg=st.TrainingConfig(); reach=surrogate._caustic_reach(0.5); exclusion_rho=
  1+cfg.eta_max/reach=1.0354; band=(0.4,0.6); rho_outer=box.y_reach/reach=2.121.
  st._farfield_region_window(box,1,band,exclusion_rho,rho_outer,reach,None,None,
  cfg) -> window=(0.941,19.333) action='keep'. st._farfield_window_contains_
  draws(box,window,tol=1e-12) -> contained True, max_subset_violation 0.0,
  n_overlap 8 (clip-by-construction). Independent per-draw recompute via
  st.dimensionless_frequency(f,mass,0.0). NO-STRATA: mock.patch.object(st,
  '_mass_strata') & ('_stratum_w_range') both call_count==0 (region_window
  confirmed to call _farfield_region_w_floor/_upper_w_cap/_apply_ppgo_trim,
  NOT strata). Tile loop st._farfield_tiles(exclusion_rho,rho_outer,5)=25
  non-empty. BYTE-IDENT no-op: st._apply_ppgo_trim(rng,None,None)==(rng,'keep').
  SELF-FALSIF: raw unclipped band violates window (w_hi-w_trust>>tol) proving
  clip does real work.
- 3 new plots output/exterior_windows_{ghost_overlay,tag_routing,
  containment_margin}.png. TEST-ONLY change (no production edits) -> cannot
  regress other suites.

## test_lensing_exterior_windows.py NEW (Build 8h-b3 S1-1..S2-3) — 2026-07-23
- New suite `cogwheel/tests/test_lensing_exterior_windows.py`, 25 tests
  green. Certifies 3 Architect specs: caustic-fixed exterior tiler +
  notch exclusion + reach parity (Spec1); w-windowed exterior label seam
  reconstruction + ghost gate (Spec2); diffractive-bottom bounded object
  (Spec3). conda python /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.
- FIXTURES (gamma=0.5, positive-parity astroid, scalar reach=sqrt(2)=1.4142):
  exclusion_rho=1+eta_max/reach=1.03536 (eta_max=0.05, from
  st.TrainingConfig().eta_max). NOTCH: 20deg, mag=1.05*r_caustic(0.5394)=
  0.566 -> rho_scalar=0.40<exclusion, 2-image exterior (physically outside
  DIRECTIONAL caustic but scalar-rho<1) -> NOT admitted. FOLD (Spec2/3):
  rho=1.2,30deg -> source=(1.4697,0.8485), maxF=1.373, n_real=2,
  w_floor=0.3824. Diffractive |obj|/maxF band [0.86,0.92] in (0.3,3);
  kernel-sum FOIL blows to 31.9>3 (upper bound teeth). recon err diffractive
  0.0 / kernel-sum 2.4e-15 (<<1e-3). Seam |F_diff-F_kernel|@w_floor=0.0.
- GHOST GATE is w_min*Im tau_c>=2.0 (=ch._FARFIELD_WINDOW_RADIANS=RHO_END/2).
  For fold configs it's UNREACHABLE on a low-w grid: full-grid gate ~0.032,
  and even mid-band (w_min=w_floor) max is ~0.66<2 across rho1.05-1.5. So
  minus-ghost label REFUSES (GhostDomainError) at the fold — that IS why the
  seam has no step (mid band = plain kernel sum). To get a PASS use a HIGH
  w_min grid: rho=1.5,40deg, w in [1.5,40] -> Im tau_c=2.007, gate=3.01>=2,
  |G|/maxF=0.0101 (O(1e-2)), minus-ghost recon err 0.0.
- REACHABLE-RED: (Spec1) serve rho via geometry.r_caustic (directional)
  instead of surrogate._caustic_reach -> disagree by O(1) (1.5 vs 3.93);
  (Spec2) mock.patch.object(ch,'_FARFIELD_WINDOW_RADIANS',0.01) flips fold
  ghost from refused->admitted, |G|/maxF up to 19.8>>1e-3; wrong switch tag
  (reconstruct diffractive envelope with KERNEL_SUM) breaks recon>1e-3.
- ORACLES independent: partition.exact_total (engine path != label algebra);
  geometry.r_caustic (directional != scalar reach); geometry.ghost_kernel
  Im tau_c predicts gate outcome without the channels wrapper.
- API PINS: ChangRefsdalChannels(w).reset(); .evaluate(gamma=,y=(y1,y2))
  [y keyword, beta/kappa default 0]. farfield_envelope_from_partition(part,
  defn) + reconstruct_farfield(w,env,delays,saddle_kernels,real_mask,defn).
  For minus-ghost serve, re-add farfield_ghost_term(w,source,matrix) to env
  BEFORE reconstruct. st._farfield_tiles(rho_inner,rho_outer,n) returns
  ((rho_c,theta_c),(half_rho,half_theta),i,j); half_theta=pi/n; []-empty when
  rho_outer<=rho_inner. surrogate._to/_from_caustic_fixed round-trip EXACT
  (drho==dtheta==0.0). 3 plots -> output/exterior_windows_{admission_map,
  seam_recon,diffractive_bound}.png.
- NEIGHBOR DRIFT (report-only, NOT mine): test_lensing_farfield_envelope.py
  12 failed+40 errors — production _subdivide_farfield_tile() renamed kwarg
  exclusion_radius->exclusion_rho in this uncommitted build; that sibling
  suite (other run's) passes exclusion_radius= -> TypeError. Unrelated to my
  additive-only new file. Did not touch it.

## test_lensing_ghost.py EXTENDED (WP1 shard 3) — 2026-07-23
- Added GhostGuardTestCase (4) + RealImageByteIdentityTestCase (4) + 2
  self-falsification -> 36 passed + 1 xfailed (was 26+1). Neighbor
  test_lensing_geometry.py still 13 green. conda python:
  /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.
- NEAR-FOLD DET GUARD: |det H_c| floor 1e-8*(1+||A||_F)^2 is UNREACHABLE
  via a physical caustic approach — det ~ 0.79*sqrt(eps) so floor needs
  eps~6e-15, but ghost position error ~sqrt(machine eps)=1.5e-8 dominates
  first. INSTEAD reach it by handing _ghost_kernel a REAL critical-curve
  point as x_c: the exact fold IS the critical curve where det(hessian)=0
  to machine precision, so |det H_c|~1e-16<<floor while Re(z)=|x|^2>0
  clears the first guard. geometry.critical_point(gamma,theta).image ->
  x_c, .source -> source. Sweep thetas {0.4,0.6,0.9,1.2} gamma=0.2.
  Message asserts contain 'det' & 'near-fold'.
- NEG Re(z) guard: synthetic x_c=[1j,0] (z=-1) etc. reach FIRST guard
  directly (fires before amplitude formed); message contains 're(z)'.
- GhostDomainError IS-A LensDomainError IS-A ValueError; tested caught as
  family base too. Public ghost_kernel also refuses on-caustic source.
- MUTATION reachable-red: set geometry._GHOST_DET_FLOOR=0.0 in try/finally,
  near-fold _ghost_kernel returns max|kernel|~3.3e97 (finite but GARBAGE,
  not NaN/inf) instead of raising -> gate on >1e30. Restore floor after.
- BYTE-IDENTITY: HEAD copy of geometry.py via git show HEAD:<path> into a
  REAL temp .py file (numba njit(cache=True) needs a real file locator) +
  functools.lru_cache(maxsize=1) so numba compile runs once. geometry.py
  is self-contained (no cogwheel relative imports) so loads standalone.
  Battery gamma{0.2,0.4,0.6} x 10 sources spanning inside(>=4 imgs) &
  outside(2 imgs) caustic; find_images/delay/magnification/morse_index/
  image_kernel all max|diff|==0.0 (assertEqual to 0.0), census tuple +
  signed Morse sum exact. Assert battery has both regimes present.
  Self-falsification: np.nextafter one-ulp perturbation makes gap>0.

## test_lensing_ghost.py EXTENDED (WP1 shard 2) — 2026-07-23
- Added 3 spec cases to existing suite -> 26 passed + 1 xfailed (was 16);
  neighbor test_lensing_geometry.py still 13 green. Full conda python:
  /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.
- DECAYING-MEMBER SELECTION: sweep gamma{0.2,0.4} x rho{1.2,1.5,1.8} x
  angles{25,45,70,110,135}deg. New oracle `oracle_ghost_members` (added to
  _ORACLE_FUNCTIONS + AST guard) returns BOTH conjugate members; verified
  exactly 2 members with equal-&-opposite Im tau, production always picks
  the +Im (argmax) one, matches oracle decaying member to TOL_TAU_REL and
  differs from growing by >1e-3.
- ON-AXIS PURE-OSCILLATION: LITERAL spec (|Im tau_c|<1e-10 finite) is
  UNREACHABLE in landed code — a root with Im u < root_tolerance(3e-7)
  DECLASSIFIES to a real image (GhostDomainError "no ghost pair") before
  Im tau hits 1e-10, and EXACTLY on-axis the diagonal source-frame matrix
  collapses reconstruction onto u=a22 (GhostDomainError). So: (a) honest
  achievable LIMIT test at rho=1.5 (past cusp, |amp|~1.12 O(1)) angles
  {1e-3,1e-4,1e-5} — kernel/delay/pos/amp all finite, Im tau>0, carrier
  |exp(iw tau)|<=1+1e-12 (no spurious growth) & monotone -> 1; (b)
  exactly-on-axis asserts GhostDomainError; (c) LITERAL contract kept as
  @expectedFailure (xfails now via the raise, xpasses when a future build
  supports on-axis). NOTE near-axis at rho=1.1 kernel BLOWS UP (near cusp)
  — must use rho>=1.5 for finite-limit fixture.
- FAR-FIELD rho=4: gamma=0.4, angle=pi/4 gives Im tau=10.478~10.5; band
  linspace(0.5,1.2,24) reproduces Architect anchors max|E_ff|=2.09e-3,
  max|C|=7.14e-4 (<1e-3), ratio 0.34 (<0.5). Envelope test: |C|/|kernel|
  == exp(-w Im tau) to rtol 1e-10 (the decay IS the envelope). Plot ->
  output/ghost_far_field_rho4_envelope.png (semilogy).
- SELF-FALSIFICATION added: growing conjugate |kernel|*exp(+w Im tau) max
  ~3e5 >1e3 (blows up) while decaying <1e-3; argmin rule picks Im<0 member.
  Both prove selection is load-bearing.

## test_lensing_ghost.py (WP1 ghost-kernel machinery) — 2026-07-23
- New suite `cogwheel/tests/test_lensing_ghost.py`, 16 tests green;
  neighbor `test_lensing_geometry.py` still 13 green (no regression).
- Independent oracle reaches ghost tau_c/det H_c ONLY via numpy.roots +
  hand-rolled quartic/frame/clog tau/FD det — NEVER touching geometry's
  ghost helpers. AST guard (`_forbidden_names_in`) walks ast.Name.id /
  ast.Attribute.attr against FORBIDDEN_ORACLE_NAMES; positive controls
  (tainted nested oracles calling geometry.delay / bare `ghost_kernel`)
  flip it red. GOTCHA: inspect.getsource of a NESTED (indented) function
  breaks ast.parse with IndentationError — must textwrap.dedent BEFORE
  parsing (module-level oracle funcs are fine, but self-falsification
  controls are nested). Fixed by dedent in the helper.
- Engine E_ff anchor: |E| = |exact_total - ppgo| t_min-demodulated on a
  linspace(0.6*w,w,24) grid (engine needs >=2-pt increasing +ve w),
  |C| = |ghost.kernel * exp(1j*w*(tau_c - t_min))| (carrier incl.
  exp(-w*Im tau_c) decay). Reproduces Architect anchors: gamma0.2/w8.5 and
  gamma0.4/w3.3, gate abs(|C|/|E|-1)<0.10 & |arg|<3.5deg — LOOSE by design
  (residual R/E ~4-6%); tight correctness is the AST-guarded oracle.
- Oracle accuracy measured: tau rel 0.0, recon x_c.x_c==1/u_c ~4.5e-13,
  amp-vs-analytic rel ~1.5e-16, amp-vs-FD rel ~5e-8 / arg ~1.2e-8
  (Richardson-extrapolated central FD, h=1e-4*|x_c| floored at 1e-5).
- Morse double-count guard: multiplying amp by exp(-0.5j*pi) must push
  arg past TOL_AMP_FD_ARG; wrong-log-branch (tau off by -pi*i) must push
  tau rel past TOL_TAU_REL AND flip Im<0. Both are self-falsification tests.
- Diagnostic overlays written: output/ghost_anchor_overlay_gamma0.2.png,
  ...gamma0.4.png (|C| vs |E_ff| across anchor w-band).
- Python path (macOS one in role header is WRONG): use
  /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.
