# Professor short-term (2026-07-22) — Build 8g-b far-field envelope redefinition REVIEW

Inference review of Build 8g-b (far-field surrogate LABEL redefined from
`partition.envelope`/SACR-C τ_c-demodulated to the full post-geometric-optics
remainder E_ff = F - Σ_{a real} H_a e^{iwτ_a}; switch=1 for real, critical_delay=0).

## Verdict: PASS
- Ran `cogwheel/tests/test_lensing_farfield_envelope.py` (py3.10
  gw_detection_ias_310; cogwheel_310 absent on this box). 42 tests + 9 subtests
  PASS in 180s. All 6 diagnostic PNGs written to cogwheel/tests/output/.
- Threshold constants verified against my own 8g-b rulings: OLD_JUMP_MIN_RATIO=100,
  NEW_ENVELOPE_MAX=5e-3, NEW_CONTINUITY_MAX=1e-3, MACHINE_REL_TOL=1e-12,
  SERVE_MIRROR_TOL=3e-3, FARFIELD_EPS_GATE=1e-3. gamma=0.0387,y1=1.3,
  Y2_SWEEP linspace(1.10,1.50,33) lands on 1.250 & 1.275. All match.
- Independent numeric probe (2 partition builds): reconstruction rel err = 0.000e+00
  (telescoping range-reduced carriers, Q2/Q3 confirmed exact, not just <1e-12);
  lobe-flip invariance rel dev = 0.000e+00 while critical_delay gap=2.56 and OLD
  envelope moves 0.58 => the lobe DOF is provably GONE. Root-cause fix confirmed.
- Tests are self-falsifying: SelfFalsificationTestCase + NewGateSelfFalsification
  feed the OLD label / corrupted E_ff into the SAME gates and assert red — gates
  have teeth (no decoration). Good practice.
- max|E_ff|=1.2e-2 at on-diagonal (1.3,1.3) over w∈[1,60] is NOT a violation: that
  point is near the caustic and low-w E_ff re-acquires near-critical oscillation
  (my Q4 caveat); it is not gated by the exterior-sweep 5e-3 ceiling.

## Operator-deferred (out of my turn budget, correctly)
- Full posterior sampling / real-data PP is the operator ship gate; I did not run it.
- Did NOT re-run the full modified surrogate/census/training suites (potentially
  slow); tube byte-identity + gate-currency-mutation are covered inside the
  far-field file and passed. If a regression review is wanted, run those three fast.

Related: mem:professor/microlensing_chang_refsdal (SACR-C; τ_c jumps between lobes
were the flagged known risk — this build is that risk being closed).
