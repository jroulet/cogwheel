---
date: 2026-08-19
bump: patch
---

`diffractive_w_low` (Rung P, `_diffractive.py`) now serves to the honest
first-breach ceiling: the N/2N tail ratio is NOT monotone in `w` near the
crossing (breach → dip → re-breach; a real order-8 truncation phenomenon,
engine-confirmed, vanishing at order 16+), so the certificate was over-
certifying (gamma=0.1 returned 13.9 when the first breach is 12.1, up to +78%
in `w`). `_rootfind_w_high` is now a two-tier running-max scan (coarse 5% far
under the bar, fine 0.2% near it) returning the largest `w` whose RUNNING MAX
clears `CERTIFICATION_BAR`; `_diffractive_bottom_ceiling` takes keyword-only
`w_lo`/`w_hi`; and the nested c3/Born/census band-split compositions
whole-band-certify correctly (bottom full, host empty) instead of regressing
to engine-host. Census and likelihood consumers thread the band verbatim.

POST-BUILD ORDER SCAN (driver, measured on the residual-demand fixture set):
the series is CONVERGENT, not asymptotic — coverage rises monotonically with
truncation order in every region (exterior 100% any-coverage at order 16+).
A 3k-draw serve-route census at the order-8 baseline (engine_residual 37.3%)
vs order 16 (32.8%) shows a 4.5 pct-point drop, above the 2-3 pct decision
bar, so `_DEFAULT_MAX_ORDER` was raised 8 → 16. At order 16 the gamma=0.1
re-crossing converges away (ceiling extends 12.1 → 40.9) and the previously
refused optimistic gammas {0.4, 0.5} admit HONEST ceilings (verified against
the exact engine, worst rel-err 1.000e-4 at the ceiling). Engine-at-ceiling
spot check passes at all gammas (worst 1.000e-4, within the 1% estimator
allowance). Engine-free: all scans/spot-checks use the shipped verifier.
