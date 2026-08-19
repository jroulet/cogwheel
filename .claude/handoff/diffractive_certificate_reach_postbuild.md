# Post-build driver steps — diffractive_certificate_reach (measured at 530e397)

## 1. Order scan {8,12,16,24} — CONVERGENT, order raised to 16

Engine-free honest ceiling (running-max N/2N, 40-pt log grid) over 40
residual draws/region from `demand_census_post_born_10k.json`:

| region | order | whole-band | any-cov | median ceiling |
|---|---|---|---|---|
| exterior | 8 | 10% | 75% | 0.54 |
| | 12 | 25% | 95% | 1.09 |
| | 16 | 30% | 100% | 1.78 |
| | 24 | 50% | 100% | 3.75 |
| tube | 8 | 0% | 15% | 10.87 |
| | 16 | 7.5% | 60% | 0.83 |
| | 24 | 12.5% | 85% | 1.00 |
| wedge_interior | 8 | 0% | 10% | 9.99 |
| | 16 | 0% | 22.5% | 5.57 |
| | 24 | 5% | 40% | 5.12 |

Coverage rises monotonically with order in every region (series is
convergent, NOT asymptotic). DECISION: raise `_DEFAULT_MAX_ORDER` 8 -> 16.

## 2. Serve-route census at order 8 vs 16 (3k draws, seed 0, engine-free)

- order 8:  engine_residual 1118/3000 = **37.27%** (diffr_analytic 335,
  born_analytic 105)  — already down from the pre-fix 44.62% (10k) by the
  first-breach certificate alone.
- order 16: engine_residual 984/3000 = **32.80%** (diffr_analytic 455,
  born_analytic 119).
- Delta = **4.47 pct points** >= the brief's 2-3 pct decision bar.

## 3. Engine-at-ceiling honesty spot check (30 points, gamma 0.1-0.3 x beta
{0,0.7} x {0.9,0.95,1.0}*w_low)

Worst rel-err = 1.000e-4 exactly at w_low (within the 1% estimator
allowance); all interior points <= bar. Whole-band top-of-band pin
(gamma=0.03, band top 40): rel-err 6.8e-6.

## 4. Order-16 correctness (verified against the exact engine)

- gamma=0.1 re-crossing converges away: ceiling 12.1 -> 40.9. The order-8
  non-monotone witness premise (breach at 12.5) is gone at order 16.
- Optimistic gammas {0.4, 0.5} now admit HONEST ceilings (3.88 / 1.30),
  engine-verified.
- Band independence holds at order 16 (drift 0.0e+00).

## HOT-PATH NOTE (driver finding, NOT a code change)

The diffractive certificate is computed per-proposal in the F070 fallback
(`_low_w_diffractive_serve`, called from `_amplification_coefficients`
before the fiducial dispatch). The hot path NEVER compares against the exact
engine for routing decisions — engine calls are pure serving. But the
certificate's fine scan IS a per-proposal cost for the ~2.8% low-gamma
whole-band population. A snapped-fiducial-key memoization was considered and
REJECTED as unsafe: the lens params are sampled, so snapping can over-certify
the actual lens's ceiling. The correct form is the campaign's trained
artifact (7b): the routing decision tree is saved with the charts and the
lnlike consumes it. This is already the queued work.
