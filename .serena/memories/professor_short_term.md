# Professor short-term: Build 3g ratio-layer inference review (2026-07-18)

## Verdict: PASS
Reviewed `cogwheel/tests/test_lensing_ratio_layer.py` (Build 3g ratio/heterodyne
acceleration over SACR-C direct path). Env: `cogwheel-newlal` (py3.10) — note the
IAS server has NO `cogwheel_310`; `.env` routes to `cogwheel-newlal`.

## Results
- Ratio-layer suite: 18/18 PASS in 38s. Neighboring fast suites
  (fast_path+likelihood+channels): 62 passed / 1 xfailed / 0 fail in 87s.
- Spec3 ratio-vs-bruteforce (inherited RB tol max(1.5,1e-2|bf|)): all 6 within tol.
  crown delta 1.077/tol 1.803 (hardest — 4 near-degenerate images), near_fold
  0.917/1.5, sheared_sw 0.814/5.76, rotated(beta=0.7) 0.541/1.5. These ~1-nat
  gaps are INHERITED RB-vs-bruteforce error (present in direct path too), NOT
  introduced by the ratio layer — confirmed by the ratio-vs-direct identity
  (1e-9) and perturbed (median<0.15) gates.
- Spec7 timing: warm best-of-5 9.809 ms, bruteforce 1401 ms, speedup 142.8x,
  ratio node count 8 (config-independent, <=20). Meets even the brief's 10 ms.
- Spec8 deep-band macro limit: closed form sqrt(mu_macro)=1.49755 reproduced
  through ratio path to <1e-6 (independently recomputed 1/sqrt((1-k)^2-g^2)).
- Refusal symmetry: macro_saddle(g=0.5,k=0.6) -> 1-k=0.4<=|g| correctly
  LensDomainError all 3 paths; cancellation(g=0.405,k=0.57) gamma_eff~0.94 >>
  cert edge -> CancellationError all 3. Verified parity boundaries by hand.

## Deviations judged ACCEPTABLE (both documented in test docstring)
1. Envelope identity gate 1e-9 (not brief's 1e-13): candidate 8-node seed grid vs
   fiducial LOO grid reproduce the engine envelope/critical_delay only to ~1e-11
   cross-grid; 1e-9 is still 7 orders below _LOO_STOP=4e-3, and self-falsification
   proves it still catches a 1e-6 spurious carrier. lnlike identity still meets 1e-9.
   Physically sound — this is grid-reproducibility floor, not interpolation error.
2. Absolute 10 ms ceiling -> machine-calibrated MS_CEILING=0.5s; HARD gates are
   speedup + node count (machine-independent). Brief itself deferred the 10 ms to
   Professor/operator. Measured floor reported. Fine.

Heavy full-sampling/PP validation is operator-deferred (out of turn budget).
