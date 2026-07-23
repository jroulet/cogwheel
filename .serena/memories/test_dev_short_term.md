# Test Dev Short-Term Observations

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
