# Saddle engine_residual in the LOBE gauge (rho_lobe) — engine-free probe

Census: `.claude/handoff/demand_census_post_c3_regate_10k.json` (10 000 draws,
post-c3-band-split revival). Population: saddle draws (`gamma > 1`) with
`route == 'engine_residual'` — 1720 draws (17.20 % prior mass), which the
census splits on `caustic_rho` as 868 "interior" (<= 1) + 852 shell (1, 2].

Method (engine-free, production helpers only): per topology-stable saddle
gamma band (`stable_gamma_bands` over (1.01, 1.6] → 12 bands, 0 dropped
slivers), the lobe frame is `_saddle_lobe_admissions(band, config,
eta_max=min_eta_max)` with `min_eta_max = f_max * min(arc_r_min)` exactly as
`_train_band_charts`; the draw is D2-folded (`|y1|, |y2|`, the `_lobe_serves`
fold) into the canonical +y1 lobe and mapped with the production
`_to_lobe_fixed` / `_lobe_boundary_radius`. `rho_outer` per band is the
production exterior bound `1 + y_outer - coordinate_radius_min`
(`y_outer = _source_scale(m_lo) = 3.0`). Image counts:
`geometry.find_images`. Zero wave evaluations.

Class precedence (mirrors the chart set): `lobe_interior` (rho_lobe <= 1),
else `tube_shell` (nearest lobe-caustic distance < min_eta_max), else
`lobe_exterior_shell` (rho_lobe <= rho_outer, the WP2 exterior-chart domain),
else `farfield_beyond` (beyond every lobe chart's reach). Draws in the
gamma = 1 guard band (1, 1.01) have no lobe frame (`unbanded_guard`).

## Cross-tab: lobe-gauge class x census caustic_rho class

| lobe-gauge class      | caustic interior (<=1) | caustic tube (1,2] | total | prior % | images        |
|-----------------------|-----------------------:|-------------------:|------:|--------:|---------------|
| lobe_interior         |                    118 |                  0 |   118 |    1.18 | 4: 107, 2: 11 |
| tube_shell            |                    178 |                  6 |   184 |    1.84 | 2: 172, 4: 12 |
| lobe_exterior_shell   |                     95 |                 22 |   117 |    1.17 | 2: 117        |
| **farfield_beyond**   |                **443** |            **824** | **1267** | **12.67** | 2: 1267   |
| unbanded_guard (γ<1.01)|                    34 |                  0 |    34 |    0.34 | —             |
| total                 |                    868 |                852 |  1720 |   17.20 |               |

(No saddle residual draw has caustic_rho > 2 or undetermined — the origin
gauge saturates near 2 on the saddle, which is also why the Born rung never
fires there.)

## Far-field w-band statistics (the redesign's table need)

- `farfield_beyond` (n = 1267, 12.67 % prior): all in census band
  `w_hi <= 60`; w_hi p50/p90/p99/max = 4.61 / 16.5 / 30.7 / 38.0;
  w_lo p1/p50 = 0.026 / 0.090; no record carries a `w_split` (the c3 rung
  declined these end-to-end). gamma spans 1.014–1.600 (all 12 bands).
  rho_lobe p50/p90/max = 5.2 / 9.6 / 20.2 vs band rho_outer 1.25–2.40.
- `lobe_exterior_shell` (n = 117, 1.17 % prior): w_hi p50/max = 7.7 / 52.5,
  same `w_hi <= 60` band, no `w_split`.

Per-band rho_outer (production bound, caustic-fixed additive units reused by
`_lobe_exterior_tiles` as the rho_lobe cap): 1.25 at band [1.010, 1.059]
rising to 2.40 by [1.158, 1.207], then easing to 2.06 at [1.551, 1.600] —
while the prior box reaches |y| = 3, i.e. rho_lobe up to ~20 in near-cusp
directions. The gap between rho_outer (~2) and the prior edge is the
uncharted far zone.

## Verdict

The deltoid far-field redesign has real, dominant chart demand: in the lobe
gauge 1267 draws = 12.67 % of the total prior mass (73.7 % of the saddle
engine residual, 23.8 % of ALL engine_residual) are genuine 2-image
lobe-EXTERIOR far-field beyond the current lobe charts' rho_outer reach —
the census's caustic_rho split disguised this as 443 "interior" + 824 "shell"
draws, confirming the F073 gauge lesson that origin-based rho does not
discriminate on the saddle. Only 1.18 % is true lobe interior and 1.84 % is
tube-shell territory; the existing WP2 exterior-shell domain covers just
1.17 % more, because production reuses the origin-polar `rho_outer_region`
(~2.0–2.4) as a rho_lobe cap while the prior box extends to rho_lobe ~20.
The redesign should chart the 2-image exterior out to the prior edge; its w
needs are modest and bounded — every far-field draw sits in the
`w_hi <= 60` band with w_hi <= 38 (median 4.6) and w_lo >= 0.025 — so a
single ~3-decade log-w axis suffices. Not deferrable on demand grounds; the
only descope-safe fragments are the guard band (0.34 %) and, if the tube
charts land, the 1.84 % tube shell.

Probe: `/tmp/saddle_lobe_gauge_probe.py`; per-draw rows:
`/tmp/saddle_lobe_gauge_rows.json`. Engine-free (geometry + caustic sweeps
only). Measured at 9f331dd, 2026-08-17.
