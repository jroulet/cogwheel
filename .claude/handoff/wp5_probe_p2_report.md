# WP-5 Acceptance Report — Probe P2 + ghost-rung error sweep (REPORT ONLY)

**Date:** 2026-08-13
**Scope:** Acceptance evidence for the exterior ppGO+ghost serving rung
(`operator._ghost_ppgo_amplification`, served via `operator._uniform_arm_value`;
ghost from `geometry.ghost_kernel`). No source edits; no committed test.
**Probe:** `/tmp/ppgo_cert/wp5_probe_p2.py` → `/tmp/ppgo_cert/wp5_probe_p2.json`,
log `/tmp/ppgo_cert/wp5.log`. 45 oracle points, ran to `DONE`.

## Method

- **Grid (Professor's P2):** `gamma ∈ {0.3, 0.5, 0.7}`,
  `|y|/rc ∈ {1.05, 1.10, 1.15, 1.25, 1.40}` (rc = caustic reach = `1/caustic_rho`,
  so `|y|/rc == rho`), `w ∈ {65, 100, 150}`, `theta = 0.6 rad` (canonical generic
  off-axis angle; off a principal axis so `Im tau_c > 0` and the ghost does not
  degenerate).
- **Served value:** the SHIPPING rung `operator._ghost_ppgo_amplification(w, y, gamma)`
  (returns `None` when either config gate fails or the geometry refuses).
- **Oracle (F069-safe):** `oracle.exact_total` = `_schwinger.f_schwinger` + mass-sheet
  reconstruction, validated to 1.4e-16, in the SAME absolute frame as
  `geometric_amplification` — **no `t_min` pairing** (memory `pair-frames-before-scoring`;
  NOT scored against `F_op` or `channels.exact_total`).
- **Gates under test (frequency-independent):** decay `Im(tau_c) ≥ 0.4`
  (`geometry._GHOST_DECAY_IM_THRESHOLD`) AND separation
  `min|x_a − x_c| ≥ 0.7` (`geometry._GHOST_SEPARATION_MIN`), taken verbatim from the
  rung. `rel_err = |served − oracle| / |oracle|`; arm bar = 1e-2.

## TABLE 1 — config-level gates (frequency-independent)

| gamma | \|y\|/rc | n_img | Im tau_c | min\|xa−xc\| | decay≥0.4 | sep≥0.7 | admit |
|------:|-------:|:-----:|---------:|-----------:|:---------:|:-------:|:-----:|
| 0.3 | 1.05 | 2 | 0.3246 | 2.0239 | False | True | **False** |
| 0.3 | 1.10 | 2 | 0.3667 | 2.0854 | False | True | **False** |
| 0.3 | 1.15 | 2 | 0.4110 | 2.1487 | True  | True | True |
| 0.3 | 1.25 | 2 | 0.5058 | 2.2805 | True  | True | True |
| 0.3 | 1.40 | 2 | 0.6634 | 2.4889 | True  | True | True |
| 0.5 | 1.05 | 2 | 0.8433 | 2.2036 | True  | True | True |
| 0.5 | 1.10 | 2 | 0.9430 | 2.2888 | True  | True | True |
| 0.5 | 1.15 | 2 | 1.0474 | 2.3757 | True  | True | True |
| 0.5 | 1.25 | 2 | 1.2705 | 2.5542 | True  | True | True |
| 0.5 | 1.40 | 2 | 1.6404 | 2.8309 | True  | True | True |
| 0.7 | 1.05 | 2 | 2.1572 | 2.7263 | True  | True | True |
| 0.7 | 1.10 | 2 | 2.3910 | 2.8499 | True  | True | True |
| 0.7 | 1.15 | 2 | 2.6356 | 2.9744 | True  | True | True |
| 0.7 | 1.25 | 2 | 3.1576 | 3.2256 | True  | True | True |
| 0.7 | 1.40 | 2 | 4.0221 | 3.6063 | True  | True | True |

All 15 configs are 2-image (exterior), as expected for `rho > 1`.

**The separation gate never binds on this grid** — `min|x_a − x_c|` ranges 2.02–3.61,
far above 0.7 everywhere. The **decay gate is the sole active discriminator**: the only
refusals are the two lowest-`Im tau_c` configs (gamma=0.3, rho=1.05 and 1.10, with
`Im tau_c` = 0.3246 and 0.3667 < 0.4). At gamma=0.5 and 0.7 even `rho=1.05` admits
because `Im tau_c` is large (0.84, 2.16). The admit/refuse split tracks `Im tau_c`,
**not the rho label** — consistent with the frequency-independent decay physics.

## TABLE 2 — per-(config, w) served relative error vs f_schwinger oracle

| gamma | \|y\|/rc | w | w·Im tau_c | served | rel_err |
|------:|-------:|-----:|-----------:|:------:|--------:|
| 0.3 | 1.05 | 65  | 21.097 | REFUSED | — |
| 0.3 | 1.05 | 100 | 32.457 | REFUSED | — |
| 0.3 | 1.05 | 150 | 48.685 | REFUSED | — |
| 0.3 | 1.10 | 65  | 23.838 | REFUSED | — |
| 0.3 | 1.10 | 100 | 36.673 | REFUSED | — |
| 0.3 | 1.10 | 150 | 55.010 | REFUSED | — |
| 0.3 | 1.15 | 65  | 26.715 | ppGO+gh | 9.767e-07 |
| 0.3 | 1.15 | 100 | 41.100 | ppGO+gh | 2.600e-07 |
| 0.3 | 1.15 | 150 | 61.650 | ppGO+gh | 1.030e-07 |
| 0.3 | 1.25 | 65  | 32.876 | ppGO+gh | 1.455e-06 |
| 0.3 | 1.25 | 100 | 50.578 | ppGO+gh | 5.284e-07 |
| 0.3 | 1.25 | 150 | 75.867 | ppGO+gh | 1.861e-07 |
| 0.3 | 1.40 | 65  | 43.119 | ppGO+gh | **1.977e-06** |
| 0.3 | 1.40 | 100 | 66.337 | ppGO+gh | 3.272e-07 |
| 0.3 | 1.40 | 150 | 99.506 | ppGO+gh | 9.513e-08 |
| 0.5 | 1.05 | 65  | 54.812 | ppGO+gh | 7.120e-07 |
| 0.5 | 1.05 | 100 | 84.325 | ppGO+gh | 1.767e-07 |
| 0.5 | 1.05 | 150 | 126.488 | ppGO+gh | 3.770e-08 |
| 0.5 | 1.10 | 65  | 61.293 | ppGO+gh | 3.734e-07 |
| 0.5 | 1.10 | 100 | 94.298 | ppGO+gh | 1.543e-07 |
| 0.5 | 1.10 | 150 | 141.446 | ppGO+gh | 3.239e-08 |
| 0.5 | 1.15 | 65  | 68.083 | ppGO+gh | 4.109e-07 |
| 0.5 | 1.15 | 100 | 104.744 | ppGO+gh | 1.130e-07 |
| 0.5 | 1.15 | 150 | 157.116 | ppGO+gh | 2.734e-08 |
| 0.5 | 1.25 | 65  | 82.584 | ppGO+gh | 1.655e-07 |
| 0.5 | 1.25 | 100 | 127.052 | ppGO+gh | 5.952e-08 |
| 0.5 | 1.25 | 150 | 190.578 | ppGO+gh | 1.315e-08 |
| 0.5 | 1.40 | 65  | 106.626 | ppGO+gh | 6.312e-08 |
| 0.5 | 1.40 | 100 | 164.040 | ppGO+gh | 1.835e-08 |
| 0.5 | 1.40 | 150 | 246.061 | ppGO+gh | 5.540e-09 |
| 0.7 | 1.05 | 65  | 140.219 | ppGO+gh | 7.996e-09 |
| 0.7 | 1.05 | 100 | 215.722 | ppGO+gh | 2.192e-09 |
| 0.7 | 1.05 | 150 | 323.583 | ppGO+gh | 6.150e-10 |
| 0.7 | 1.10 | 65  | 155.414 | ppGO+gh | 4.368e-10 |
| 0.7 | 1.10 | 100 | 239.098 | ppGO+gh | 9.770e-11 |
| 0.7 | 1.10 | 150 | 358.647 | ppGO+gh | 2.578e-11 |
| 0.7 | 1.15 | 65  | 171.316 | ppGO+gh | 4.987e-09 |
| 0.7 | 1.15 | 100 | 263.563 | ppGO+gh | 1.284e-09 |
| 0.7 | 1.15 | 150 | 395.344 | ppGO+gh | 4.046e-10 |
| 0.7 | 1.25 | 65  | 205.242 | ppGO+gh | 7.682e-09 |
| 0.7 | 1.25 | 100 | 315.757 | ppGO+gh | 2.368e-09 |
| 0.7 | 1.25 | 150 | 473.636 | ppGO+gh | 6.701e-10 |
| 0.7 | 1.40 | 65  | 261.435 | ppGO+gh | 8.428e-09 |
| 0.7 | 1.40 | 100 | 402.208 | ppGO+gh | 2.332e-09 |
| 0.7 | 1.40 | 150 | 603.312 | ppGO+gh | 6.982e-10 |

## Verdicts

### Gate-partition verdict — YES, the gates partition the grid

- **39** gate-admitted (config, w) points; **6** gate-refused; **0** gate/serve skew
  (every admitted point served, no admitted point returned `None`).
- **Every admitted point is under the 1e-2 arm bar** — max served `rel_err` over all
  39 admitted points = **1.977e-06** (gamma=0.3, rho=1.40, w=65), ~4 orders of margin.
  0 admitted points overshoot.
- The partition is **one-directional and empirically confirmed on the admit side**:
  `{Im tau_c ≥ 0.4 AND sep ≥ 0.7}` ⇒ `rel_err ≤ 1e-2` (in fact ≤ 2e-6). The refuse
  side returns `None` (rung declines, falls through to the exact engine), so its
  correctness is by construction (handoff fact 4), not directly scored here.
- Caveat: on this grid the **decay gate alone** produces the partition; the separation
  gate is slack (≥ 2.02 everywhere). The separation gate guards a distinct failure
  mode (near-image ghost merger, small `min|x_a − x_c|`) that P2's `rho ≥ 1.05`,
  `theta = 0.6` sampling does not exercise. This is expected — P2 is a decay/accuracy
  probe, not a separation-boundary probe — and does not weaken the admit-side result.

### Acceptance band |y|/rc ≥ 1.15, w ∈ (60, 150]

- **27** served points, **MAX served rel_err = 1.977e-06 → PASS (< 1e-2)**.
  (Max attained at gamma=0.3, rho=1.40, w=65; the ≥1.15 band's worst case is the
  low-gamma / high-rho / low-w corner, still 4 orders under bar.)

### Caveat band |y|/rc = 1.05

- **3 / 9** (config, w) points REFUSED to the engine — exactly the gamma=0.3, rho=1.05
  column (all three w), driven by `Im tau_c = 0.3246 < 0.4`. The gamma=0.5 and 0.7
  rho=1.05 columns admit (large `Im tau_c`) and serve at rel_err ~1e-7 and ~1e-9.

### Overshoot / w-floor

- **None.** No `|y|/rc ≥ 1.15` config survives both gates yet overshoots 1e-2.
  Professor's prediction holds; **no frequency-dependent w-floor is needed**, and there
  is **no Inspector finding to raise** for a follow-on coder_fix. WP-2 is left untouched.

## Bottom line

The two shipping frequency-independent config gates cleanly gate the exterior ppGO+ghost
rung: every admitted point serves at ≤ 2e-6 relative error against the F069-safe
f_schwinger reconstruction oracle (bar 1e-2, ~4 orders of margin), the acceptance band
|y|/rc ≥ 1.15 over w ∈ (60,150] passes at max 1.977e-06, and the caveat band refuses
exactly where the decay physics says it should. No overshoot, no w-floor, no source edit.
