# Test Dev Short-Term Observations

- build2c test_lensing_operator.py: the suite ALREADY EXISTS, complete and
  in-house-idiom (independent mpmath amplification oracle at ORACLE_DPS=50 built
  from mpmath.hyp1f1 + integer-ladder shear operator — shares no code/substrate
  with production dd Kummer kernel; anti-vacuity OperatorTestCase.tearDown;
  SelfFalsificationTestCase with 6 red-proofs; ALL-CAPS #: constants; module
  docstring justifying RTOL_GATE=1e-10 and the L<=25 tested domain). Verified
  the operator.py API the suite calls is present & signature-matched via Serena
  find_symbol: F_op(w,y,gamma,*,beta,kappa,max_order)->(complex,
  OperatorDiagnostics); OperatorDiagnostics frozen dataclass fields
  order_used/converged/estimated_relative_tail/cancellation_ratio; constants
  RHO_START/RHO_END/L_MAX/_CANCELLATION_REFUSAL; CancellationError; free fns
  cancellation_exponent/select_branch/geometric_amplification. channels.select_
  branch/RHO_END/RHO_START are the SAME objects (BranchGateTestCase.
  test_thresholds_have_one_home) — WP1 audit-relevant.
- WP1 (channels.py _channel_switch neighbourhood) and WP2 (likelihood.py nsub
  8->2) do NOT touch operator.py; the operator suite is a REGRESSION GUARD for
  the geometric/wave branch gate + F005 certified-or-refuse contract. It should
  stay green under WP1+WP2 unchanged.
- RUNTIME GREEN CONFIRMED (build2c): after owner reconfirmed the bare-denial
  artifact, re-issued via Serena execute_shell_command. KEY GOTCHA: project root
  / Serena cwd is the WORKTREE /Users/tejaswi/Work/cogwheel-claude-dev, NOT
  /Users/tejaswi/Work/cogwheel (running from the latter -> "file not found").
  From the worktree root: `python -m pytest cogwheel/tests/test_lensing_
  operator.py -v` => 21 passed, 69 subtests passed in 83.68s. This exercises the
  REAL post-WP1/WP2 channels/geometry/_hyp1f1 (operator imports them);
  BranchGateTestCase.test_thresholds_have_one_home green => channels still shares
  operator's select_branch/RHO_END/RHO_START (WP1 threshold-home invariant
  holds). F005 certified-or-refuse band + geometric-slope + mass-sheet +
  cancellation-refusal all green.
- Shell-gate note: the BARE "user doesn't want..." denial is transient (owner-
  confirmed 2026-07-16); re-issue ONCE. A denial WITH a reason binds — hit one:
  Bash was refused with "USE SERENA for shell commands" (only git/gh/conda/brew/
  read-only sys cmds may use Bash directly); pytest must go through Serena
  execute_shell_command. Sibling-suite regression run (channels/geometry/hyp1f1
  standalone) got the bare denial twice incl one re-issue -> left UNVERIFIED
  standalone, but they ran fine AS IMPORTED by the operator suite.
- test_lensing_waveform.py: AUTHORED FRESH (overwrote a prior stub).
  Tests the waveform LAYER (LensedWaveformGenerator), not F_op internals.
  Base WaveformTestCase w/ anti-vacuity tearDown; SelfFalsificationTestCase;
  independent freq oracle = literature MTSUN_LIT=4.925490947641266e-6 (NOT
  lal.MTSUN_SI). Spec1 MacroSaddleControl: IN_BAND y=(0.30,0.10),gamma=0.10,
  kappa=0 (gamma_eff=0.10) certifies at order-42; BAND_EDGE = old
  mis-specified control y=(0.5,0.25),gamma=0.25,kappa=0.5 (gamma_eff=0.5)
  refuses w/ operator.CancellationError at w=[30,40,50] (L=w*0.79<48 wave,
  w*sqrt(s)<60). Refusal confidence HIGH: documented large-shear
  (gamma'=0.4,L=10,w=40) already refuses at max_order=42 (tail 1.27e-4);
  BAND_EDGE deeper -> refuses at least as easily. Spec2 UnlensedFloor:
  FLOOR_CONFIG y=(0.30,0.10),gamma=0.10; masses [1e-12(excl),0.1,0.3,1,2,4],
  f0=100Hz z=0 -> w0=1.2378e-2*M; assert |F|-1 strictly increasing in w,
  in (0,0.5), floor[0]<1e-2, floor[-1]/floor[0]>5 (QUALITATIVE thresholds).
- GOTCHA spec2: tiny-w point (w~1e-14) does NOT reliably look broken (may
  return clean ~1e-14 as series converges ~order9 before coeff gamma/(2w)
  overflows ~order27, OR raise). So exclusion test asserts only the DOMAIN
  CUTOFF mechanism (w0<W_FLOOR_CUTOFF=1e-3 and mass-filter drops it), not
  visible breakage. Deferred small-w gap cited in docstring; no dedicated
  ticket in FINDINGS.md so referenced F005 as containing contract.
- SHELL GATE FULLY CLOSED this session: bare denial hit 5x (incl re-issues)
  on py_compile AND pytest via serena execute_shell_command (cwd= param,
  no cd) AND Bash. Suite delivered UNVERIFIED-AT-RUNTIME. All API calls
  verified read-only via find_symbol. NEXT run: execute `python -m pytest
  cogwheel/tests/test_lensing_waveform.py -v` from worktree root; calibrate
  (a) BAND_EDGE refusal at w in {30,40,50}, (b) floor magnitude/monotone
  thresholds.
