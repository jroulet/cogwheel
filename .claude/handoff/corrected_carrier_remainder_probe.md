# Corrected Born carrier remainder probe — measured against the exact engine

Date: 2026-08-18. Driver-side measurement, read-only on production.
Question: can the corrected carrier
`F_cc = born_lead_carrier * (1 + a0/q2r + 1j*(w/2)*b1/q2r)`
serve the far field under the physical prior at the 1e-3 c3-currency bar?

**Verdict: NO — not at this order and not at ANY series order.** The true
remainder is not the next series term: it is floored by the omitted SECOND
image's oscillatory beat, amplitude `sqrt(|mu2|/|mu1|)` (exact image census),
which is w-independent and 29x–1000x above the bar across the entire
Born-gate census population. Coverage of the 1461 Born-gate draws: **0.00%**.
Go straight to residual charts; for the high-w end the natural carrier under
those charts is the TWO-image geometric-optics sum, measured here at
2e-4–2e-3 by w = 60 where both Born carriers are 100%+ wrong.

## Method / cost

- Harness mirrors `cogwheel/tests/test_lensing_born_certificate.py`'s
  accuracy sweep: oracle `operator.F_op(w, y, gamma)` (absolute Fermat frame,
  no demodulation), carriers from `chang_refsdal/_born.py`.
- Harness validated two ways: (i) reproduces the shipping test's fixture
  accuracy (measured rel_lead max 1.96e-4 at gamma=0.3, |y|=80 vs the test's
  ~2e-4); (ii) `F_cc` is byte-identical to production
  `born_amplification` on positive parity (measured diff exactly 0.0 at
  three points).
- Points: 50 total, all verified 2-image via `geometry.find_images`.
  - 32 saddle far-field: gamma in {1.05, 1.2, 1.4, 1.6} x rho_lobe in
    {2, 5, 10, 20} x 2 lobe-local angles, placed with the authoritative lobe
    gauge (`surrogate_training._lobe_caustic_points` +
    `_directional_lobe_boundary` + `surrogate._from_lobe_fixed`).
  - 18 positive-parity beyond-box: gamma in {0.1, 0.3, 0.5} x caustic-gauge
    rho from 2 to the prior edge |y|<=3 x 2 angles. NOTE: for gamma >= ~0.6
    the class is EMPTY inside the prior (`caustic_rho(gamma, 3.0) <= 2`), so
    gamma 0.7/0.9 contribute no points.
- w grid: 10 log nodes on [0.03, 60]; all 500 nodes engine-certified (zero
  Schwinger refusals).
- Engine calls: 500 (main sweep, 40.7 s wall) + 20 (shipping-fixture
  validation) + 20 (two-image GO supplement) = **540 total** (budget 600).
- Census: the 1461 Born-gate draws recovered as `route == 'born_analytic'`
  records of `demand_census_post_c3_regate_10k.json` (their min
  `est(w_hi)` = 0.019493 exactly reproduces the commissioning measurement
  0.0195).

## Result tables

### Relative error and improvement factor (500 nodes)

| population | n | rel_lead (25/50/75/max) | rel_cc (25/50/75/max) | impr = rel_lead/rel_cc (25/50/75/max) |
|---|---|---|---|---|
| saddle | 320 | 3.2e-2 / 1.5e-1 / 6.5e-1 / 18.3 | 4.2e-2 / 1.4e-1 / 1.2 / 1.0e3 | 0.19 / 0.91 / 1.31 / 14.7 |
| positive | 180 | 7.4e-2 / 1.7e-1 / 8.5e-1 / 2.6 | 8.5e-2 / 1.5e-1 / 8.4e-1 / 124 | 0.60 / 0.90 / 1.40 / 5.8 |

**Nodes admitted at rel_cc <= 1e-3: 0 of 500** (also 0 of 500 for the lead;
best node anywhere: rel_cc = 3.24e-3, rel_lead = 3.53e-3, both at the single
most favorable point sampled — saddle gamma = 1.05, rho_lobe = 20,
q2r = 1.6e4).

Improvement is regime-structured, not uniform:
- High-w end (w*b1 term dominates the lead's error): impr 5–15x — the
  correction removes exactly that term, then lands on the beat floor.
- Low-w end (w·dtau < ~2): impr median 0.48 (saddle) — the a0 term ACTIVELY
  HURTS because it breaks the exact w->0 limit F -> sqrt(mu_macro):
  measured rel_cc(w->0) = (0.79–2.4) x |a0|/q2r, median 1.47 (163 nodes).

### The floor is the omitted second image's beat

At nodes where the series part is negligible (est <= 0.05), rel_cc flattens
at a w-independent floor. Against the exact image census amplitude ratio
`amp = sqrt(|mu2|/|mu1|)`:

- floor(min over w-grid) / amp: 0.14–0.90 across all genuine far-field
  points (median 0.42; the grid samples the beat at quasi-random phase, so
  the min sits below the envelope). The only ratios > 1 are close-in points
  where |a0|/q2r > 0.2 (series divergent anyway).
- worst-phase envelope K_env = max rel_cc / amp on est<=0.05 nodes:
  median 1.10, p90 1.57, max 2.98.

Spot check at the best point (saddle gamma=1.05, q2r=1.6e4): certificate
est = 6.2e-4 but measured rel_lead = 4.3e-3–7.1e-3 = amp_ratio 6.3e-3 times
O(1). **The certificate's est is therefore NOT a sound remainder bound once
it drops below the beat amplitude — it bounds only the smooth series part.**

### Empirical admission boundary

There is no useful (w, q2r) boundary: the beat floor is w-independent, so
trimming the band cannot rescue a point. The measured admission region is

    rel_cc <= 1e-3  <=>  K_env * sqrt(|mu2|/|mu1|) <~ 1e-3   (necessary)
                         AND  ~est(w)^2 series term <= bar    (secondary)

i.e. essentially the SAME ultra-far geometries where the lead-only carrier
already sits at ~2e-4 (|y| >= 80 fixtures: rel_cc max 1.5e-4 vs rel_lead
2.0e-4 — a 1.1–1.4x cosmetic gain). Inside the prior (|y| <= 3 positive,
lobe-gauge saddle) the region is EMPTY.

### Saddle vs positive parity

No saddle-specific pathology: a0/b1 parity-agnosticism holds empirically.
Both parities show the same beat-floor mechanism and the same improvement
structure; the saddle's numbers are set by its geometry (q2r, amp_ratio),
not its parity. First saddle accuracy measurement of this carrier: at the
far lobe-gauge edge (gamma=1.05, rho_lobe=20) rel_cc = 3.2e-3–6.4e-3, i.e.
even the best saddle far-field point in the class fails the bar by 3x.

### Coverage of the 1461 Born-gate census draws

Per-draw exact image census + closed-form factors, over each draw's own
band [w_lo, w_hi]:

| quantity | min | p5 | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| amp_ratio = sqrt(mu2/mu1) | 2.87e-2 | 5.96e-2 | 9.06e-2 | 1.37e-1 | 2.42e-1 | 1.0 |
| est(w_hi) | 1.95e-2 | — | 2.12e-1 | 5.47e-1 | 2.54 | 1.2e4 |

- Corrected-carrier coverage (beat-only, K=1 — most optimistic): **0.00%**.
  With measured K_env=1.57 and the est^2 series term added: **0.00%**.
- Lead-only baseline at 1e-3 (or at the 5e-5 certificate bar): 0.00%.
- The minimum beat amplitude in the population (2.87e-2) exceeds the bar by
  29x; the median by 137x. No w-band restriction changes this.

### Supplement: two-image GO carrier (20 engine calls)

`F_2GO = sum_i sqrt|mu_i| exp(1j*w*tau_i - 1j*pi*n_i/2)` at four
census-representative points (q2r 8–90, both parities):

| point | w=0.3 | w=1.1 | w=4.2 | w=16 | w=60 |
|---|---|---|---|---|---|
| saddle g=1.2 q2r=90 | 1.3e-1 | 8.1e-2 | 2.6e-2 | 2.1e-3 | 5.4e-4 |
| saddle g=1.4 q2r=24 | 2.4e-1 | 1.3e-1 | 6.3e-2 | 9.7e-3 | 2.1e-3 |
| positive g=0.5 q2r=29 | 8.4e-2 | 2.8e-2 | 7.0e-3 | 1.6e-3 | 4.9e-4 |
| positive g=0.3 q2r=8 | 2.7e-2 | 1.5e-2 | 2.6e-3 | 5.2e-4 | 1.9e-4 |

The 2GO error decays ~1/w (per-image wave correction) and crosses the 1e-3
bar at w ~ 15–60 exactly where both Born carriers are 50–500% wrong. The
far-field demand splits cleanly: low w belongs to the diffractive rungs
(already served), high w to a two-image carrier + a SMOOTH residual — i.e.
residual charts over a 2GO carrier, not any single-image Born series.

## Verdict (one paragraph)

The corrected Born carrier is **not serve-worthy at any order**. Its measured
remainder is dominated not by the next series term but by the omitted second
image's oscillatory beat — amplitude sqrt(|mu2|/|mu1|), a separate stationary
point that no polynomial correction in 1/q2r can represent — and that beat
exceeds the 1e-3 bar at every one of the 1461 Born-gate census draws (min
29x over bar, median 137x) and at every one of the 500 probe nodes across
both parities. The a0/b1 correction buys a median improvement factor of 0.9
(i.e. nothing), helps 5–15x only at the high-w end where it then lands on the
same beat floor, and actively degrades the low-w end (a0 breaks the exact
w->0 limit; error ~1.5x|a0|/q2r). Skip corrected-carrier serving entirely and
go straight to residual charts; the measured 1/w decay of the two-image GO
carrier (2e-4–2e-3 at w=60 on census-realistic geometries) marks that as the
right carrier for the high-w far field, with the diffractive rungs keeping
the low-w end.

## Artifacts

- /tmp/cc_probe.py, /tmp/cc_probe_results.json (500-node sweep, 40.7 s)
- /tmp/cc_analyze.py, /tmp/cc_validate.py, /tmp/cc_floor.py,
  /tmp/cc_census2.py, /tmp/cc_2go.py (analysis; no production edits)
