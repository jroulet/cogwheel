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
