# Test Dev Short-Term Observations

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
