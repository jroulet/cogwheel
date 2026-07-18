# Test Dev Short-Term Observations

## Build 3f WP2 likelihood suite (test_lensing_likelihood.py, 2026-07-18)

Fixed the WP1 breakage + added the likelihood-layer SACR-C gates my run
owns (GATE 3 + STRUCTURAL/TIMING; GATE1/5 re-checked THROUGH production
methods). Full suite GREEN: 29 passed, 1 xfailed (exit 0). 3 new PNGs.
Only edited test_lensing_likelihood.py (git confirms).

FIX: existing NearCuspRegressionPinTestCase died with
`channels.py:923 TypeError: _real_only_channel_switch() takes 3 pos args
but 4 given` — WP1 added critical_delay 4th arg to _channel_switch while
this suite's mutation monkeypatch was still 3-arg. Added the 4th param
(del critical_delay # inert: buggy real-only rule ignores tau_c). Same
root cause as the STILL-BROKEN test_lensing_channels.py
RealOnlyNeighbourFalsificationTestCase (5 fail+4 err) — that suite is
another run's; I did NOT touch it (scope).

NEW gates (all through production likelihood methods, over W=geomspace(
0.3,30,506); mass IRRELEVANT to LOO placement — _envelope_loo_nodes uses
only gamma/beta/kappa/y1/y2 + the w grid, so I pass a 5-key lens dict +
w window directly, no mass):
- GATE 1 (production layer): reconstruct F AT the LOO nodes (envelope
  engine-exact there => only telescoping carrier algebra, no interp) vs
  exact_total <=1e-13; measured ~2e-16 all anchors. (Dense-1e-13-with-
  exact-envelope is gauge-suite territory.)
- GATE 3: production LOO eps=max|dF|/max|F| on dense truth <1e-3
  (measured 1.8e-4..8.9e-4) + N<=48 (measured 26/28/26/28/32). Records
  cached once in setUpClass; 5 test methods assert on them + bar/summary
  plots.
- GATE 5: deep-band W=geomspace(1e-12,1e-8,40), gamma=0.2/kappa=0, |F|
  vs LITERAL 1/sqrt((1-k)^2-g^2)=1.020621, rel 7.9e-9<1e-6 AND flat
  7.9e-9. N=8 (seed only — envelope flat, LOO adds nothing).
- STRUCTURAL/TIMING: node-count config-independence (spread<=ceiling/2)
  + public speedup lnlike vs bruteforce best-of-5 (~47x). 18ms warm
  lnlike ceiling = @expectedFailure (measured 29.5ms; engine 1F1 ~89%,
  out of scope — my memory predicted this; matches fast-path xfail
  precedent). xfailed cleanly.
- SELF-FALSIFICATION: 1e-2*max|F| bump on interior envelope node ->
  eps 9.9e-3>1e-3 (GATE3 red) AND at-node identity breaks (GATE1 red).

KEY TECHNIQUE: assemble F_recon = sum_a exp(1j w tau_a) K_a from the
SHIPPED _reconstruct_kernels output (module-documented channel-sum
identity) — independent of reconstruct_from_envelope's own `total`
return, so the error isolates the kernels the likelihood contracts.
Oracle = fresh ChangRefsdalChannels(w).evaluate().exact_total (untouched
engine). _reconstruct_kernels(dense_w, coarse_w, env_nodes, partition)
needs dense_w within [coarse_w.min,max]; seeding LOO with the same 506-pt
grid makes endpoints coincide (no extrapolation).

NEIGHBORS: gauge+operator 68 passed (regression anchors green incl
MacroMagnificationLimit, F008 crossing/label-continuity). channels
5f+4e PRE-EXISTING (WP1 4-arg drift), not mine.

## Build 3f SACR-C gauge suite (test_lensing_gauge.py, 2026-07-18)

Extended the committed _gauge algebra suite with a SACR-C end-to-end
reconstruction layer covering the Architect gates. Full suite GREEN:
46 passed in ~6s (exit 0). 5 diagnostic PNGs in cogwheel/tests/output/.

GATES IMPLEMENTED (all at the gauge/channels reconstruction layer, no
likelihood.py needed):
- GATE 4 (PRIMARY): |S_a H_a| <= 2 at fold+cusp crossings, eta=+/-0.002
  both sides. Measured worst 1.21. Fixture _crossing_saddle_switch built
  from geometry/operator/_gauge ONLY (AST guard forbids channels names).
  Non-vacuity control: bare |H|~1e8 near caustic (switch does real work).
- GATE 1: envelope_total vs partition.exact_total, rel<=1e-13 on 5
  anchors; measured ~2e-15 (telescoping bit-exact).
- GATE 2: hindsight greedy on E(w), F-reconstruction-error driven
  (CubicSpline in ln w + _gauge.envelope_total), N<=26; measured 17-21.
- GATE 5: deep-band macro limit vs LITERAL 1/sqrt((1-k)^2-g^2) at
  w=1e-12..1e-8, rel<1e-6 AND flat plateau (ptp/mean<1e-6). |F|=1.020621.
- F001: SYNTHETIC large delays (up to 499.91, w up to 30 -> w*tau~1.5e4
  rad) vs PURE-mpmath oracle (dps=50), rel<=1e-10; measured ~5e-13. Real
  anchors only reach ~6 rad so synthetic delays are REQUIRED to hit the
  "thousands of radians" band.

KEY DECISIONS:
- GATE 3 (production LOO placement) + STRUCTURAL/TIMING (lnlike speedup)
  are likelihood.py-layer -> left to test_lensing_likelihood.py (another
  run). My suite owns the reconstruction-layer gates only. Noted in report.
- mpmath oracle _mpmath_envelope_total is pure-mpmath (AST guard forbids
  _gauge/channels/np/exp) -> F002 oracle independence.
- Self-falsification class: S=1 -> |SH|~1e8 (GATE4 red); 1e-6 envelope
  perturb -> breaks 1e-13 identity (GATE1 red); w*tau~1e12 -> mpmath
  disagreement 1.3e-5 (F001 red). All three go red as designed.

CORROBORATION/PITFALL: engine ChangRefsdalChannels requires >=2 freq
points (GATE 5 tiny-w must be a >=2-pt grid, not scalar). Window
[0.3,30] chosen: w>=50 trips CancellationError on 3 of 5 anchors.

NEIGHBOR REGRESSION (NOT MINE): test_lensing_channels.py now has 5 fail
+4 error, ALL in RealOnlyNeighbourFalsificationTestCase, from production
WP1 adding critical_delay 4th arg to channels._channel_switch while that
suite's 3-arg _real_only_channel_switch monkeypatch wasn't updated. I did
not touch that file (git confirms). Pre-existing drift owned by the
channels-suite run; scope discipline -> left untouched.


## Build 3e WP1/WP2 envelope suite (test_lensing_envelope_reconstruction.py, 2026-07-18)

WP1 (ChangRefsdalChannels.transition_envelopes) + WP2 (envelope+carrier
reconstruction in _amplification_coefficients) were BLOCKED / never landed
(see coder_short_term). So the 9 Architect specs mostly target a phantom
`transition_envelopes` API. Rather than fabricate tests against a
non-existent method (forbidden + unverifiable oracle), wrote an honest
integration/CONTRACT suite. GREEN: 10 passed, 1 xfailed (exit 0). Neighbor
suites test_lensing_gauge + test_lensing_channels = 55 passed (no regress).

WHAT IT GUARDS:
- API BOUNDARY: @expectedFailure asserting hasattr(ChangRefsdalChannels,
  'transition_envelopes') -> xfail today, flips to xpass/RED the moment WP1
  lands (auto-detects the future implementation). Plus positive contracts:
  channels public API == {evaluate,evaluate_path,reset,w};
  _amplification_coefficients source still 'return delays,k0,k1,partition'
  and lacks 'transition_envelopes' (Build-3d contract intact).
- SPEC 7 (only genuinely-uncovered spec-mapped invariant): large-phase
  carrier accuracy of shipped _gauge.reconstructed_total vs an independent
  pure-mpmath oracle (mpmath.mpc(cos,sin), dps=50) at CROWN scale
  (w up to 2000, delays to 3.4 -> w*tau ~ few thousand rad). Existing
  gauge/channels suites only reach w*tau<=36, so this band was untested.
- SELF-FALSIFICATION: unreduced float64 exp(-i*w*tau) loses digits at
  extreme phase; mod-2pi (mpmath.fmod) recovers <1e-11.
- ORACLE INDEPENDENCE: AST guard forbids oracle referencing _gauge/
  channels/np/exp; asserts it actually uses mpmath.

KEY LESSON (numpy range reduction): np.exp(1j*x) does ACCURATE large-arg
range reduction. Float64 phase-loss demos FAIL if w and tau are
exact-power-of-ten (product < 2^53 is exactly representable -> no error).
Must use IRRATIONAL-scaled factors: EXTREME_W=pi*1e6, EXTREME_DELAY=e*1e6
so the PRODUCT carries >53 bits and rounds by ~(w*tau)*eps ~1e-3 rad. The
precision loss is in the MULTIPLICATION w*tau, not in exp's reduction.


## WP1 fast-path suite (test_lensing_fast_path.py, 2026-07-18)

Rewrote the WP1 suite (6 Architect specs, kink-aware segmented kernel
interpolation). Full suite GREEN: 21 passed, 7 xfailed (exit 0).
Fixture: UNEQUAL-mass m1=35/m2=25 non-spinning, single det 'H', uniform
fbin DF_BIN=4, DELTA_T_MAX=0.02, seed 20260717 (equal-mass zeros 2 of 4
by_m harmonics -> 0/0 RB NaN; premise repair, not tolerance).

FINDINGS (WP1 shortfalls, encoded as unittest.expectedFailure so suite
stays green + self-corrects to unexpected-success when fixed):
- specs 1&2 RAW-KERNEL recon on shipped reduced grid: 3.6e-2..2.6e-1 null
  -safe, 35-260x above <1e-3 Build-3b gate. Converged 400-node GLOBAL
  spline hits 1.4e-7..2.2e-5 (positive control green) -> reduced node
  budget is the sole shortfall, oracle/method sound.
- spec 3 SEGMENTATION NOT load-bearing: at every crown kink the single
  global spline is 12-100x BETTER than segmented (mutation doesn't ring).
  Per brief contingency -> node budget should be re-derived.
- spec 5: crown reduces only 3.57x (<4x target; near-cusp/near-fold/
  sheared hit 4.5-6.2x). well-sep draws MORE nodes (30) than crown (28)
  -> monotonicity inverted (adaptive budget tracks delay-spread/beat).
  crown warm lnlike ~18.8ms > 15ms ceiling. Speedup 77x (>>2.5x, green).
- spec 4: near-fold RB-vs-brute 1.76nat > 1.5 gate (observable bitten by
  raw-kernel regression); crown/near-cusp/well-sep/sheared <=1.07nat green.

CORROBORATION: pre-existing test_lensing_likelihood.py (owned by another
run) ALSO fails near-cusp RB-vs-brute 2.35>1.5 -> independent confirmation
of the WP1 RB regression. Did NOT edit it (scope discipline).

TECHNIQUE NOTES:
- Off-grid raw-kernel probing must reconstruct via the SAME per-segment
  not-a-knot algorithm as production (searchsorted breakpoints + per-seg
  CubicSpline real/imag), then PROVE fidelity by reducing to k0/k1 via
  _kernel_fit_value/_kernel_fit_slope and asserting bit-equality vs
  like._amplification_coefficients (maxdiff 0.0). Closes "am I testing
  production?" gap for an internal path that only exposes reduced coeffs.
- Oracle = fresh ChangRefsdalChannels(w).evaluate at probe w (deterministic
  initial labeling matches production's fresh coarse-grid instance ->
  channel labels align; F002/F010 independent of the spline under test).
- ChangRefsdalChannels(w) requires strictly-increasing positive w:
  np.unique(np.sort(probes)) before evaluate.
- expectedFailure is the clean stdlib mechanism for "leave a plan
  -anticipated production shortfall RED without failing the build";
  pair each with a green converged positive control for non-vacuity.
