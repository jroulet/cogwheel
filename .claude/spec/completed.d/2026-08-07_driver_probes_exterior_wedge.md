---
date: 2026-08-07
section: Backlog
---

### Driver probes: exterior recursion + wedge v3 re-measurement (post-strand)

The `remeasure_v3` build stranded in test_dev (3h, agent error); the regions
filter and slow-operation judge shipped as `e4b7b80` with the Inspector PASS.
The two probes the plan called for ran as driver steps on neso with the
`regions` filter and progress beats:

**Exterior recursion (`scripts/probe_exterior_recursion.py`, one band,
`gamma_band_halfwidth=0.04`):**

- 31 charts; **9/31 pass the 1e-3 bar** (vs the 2026-08-06 baseline of
  22/57 passing). Subdivision children are GOOD (eps 6e-5..1.2e-3); the
  failing charts are the ROOT tiles (eps up to 134).
- **Depth histogram: {0: 2, 1: 6, 2: 7, 3: 16}** — 16/31 hit the depth-3 cap.
- **13 depth-3 tiles still fail 1e-3**, eps 1.2e-3 .. 3.6 (hundreds to
  thousands x tolerance). Subdivision to the cap did NOT fix them.
- CONCLUSION: confirms `lensing_exterior_should_chart_in_polar_not_sd` —
  the `(s, d)` coordinate is wrong for the exterior bulk. Root tiles are
  intrinsically bad (foot tie_ratio degeneracy), recursion paper-overs them
  but the depth-3 residual failures are coordinate-level, not resolution.

**Wedge v3 (`scripts/probe_wedge_v3.py`) — FINAL, single-stratum:**\n\n- **CONFIRMED with the correct minimal test.** After adding `m_lens_range`\n  to `train()` (so a per-region probe is a real single-stratum call), the\n  wedge probe ran `train(regions=('wedge_interior',), m_lens_range=(10,15.8))`\n  and produced **10 charts, 9/9 with valid eps passing the 5e-2 bar**\n  (all 2.0e-3..1.6e-2, median 6.0e-3, 3.7 min).\n- The earlier "NaN median / 19 charts" readings were TWO probe bugs: (1) the\n  full-prior config (13 w-strata -> ~130 tiles, wrong scale for the v2\n  baseline), and (2) the probe read `chart.provenance` in-memory which lacks\n  `heldout_eps` after load (fixed to read NPZ provenance).\n- Chart count 10 vs v2's 18: fewer because single-stratum with minimal\n  subdivision; all pass the bar. The v2-vs-v3 question is ANSWERED: v3\n  reproduces a clean working interior, the coordinates are sound.\n- The wedge centre carrier-flip (`lensing_wedge_centre_carrier_flips_in_gamma`)\n  remains the only open interior gap.\n\n**Correction to the earlier record (supersedes the 19-chart claim):**
  The first run used `gamma_band_halfwidth=0.48` (a giant band, 12x
  production width) and returned all-NaN held-out eps — a PROBE-CONFIG
  artifact, not a coordinate failure. Re-run at the production
  `gamma_band_halfwidth=0.04`: the depth-0 wedge tiles carry GOOD eps
  (3.5e-3, 7.9e-3, 3.3e-2, 7.5e-3 — under/at the 5e-2 bar), and a marginal
  root tile subdivides to a passing child (7.5e-2 -> 7e-4). This matches
  the 2026-08-06 validated interior result; the interior wedge
  caustic-relative coordinate achievement stands.
- The residual NaN cluster is confined to ONE tile branch (`_3_0_c3*`,
  depth-3, all NaN = zero held-out points served). This is the DOCUMENTED
  astroid-centre carrier-flip / `theta_wedge` degeneracy
  (`lensing_wedge_centre_carrier_flips_in_gamma`), not a coordinate
  failure — the innermost wedge tile raises CarrierDiscontinuityError at
  small `r` (measured 2026-08-06). The v2-vs-v3 baseline question is
  ANSWERED for the valid tiles: v3 reproduces working interior charts.
- ACTION: the wedge centre carrier-flip (a filed fragment) is the open
  interior gap; the wide-band probe config mistake is a lesson (probe
  config must match the production tiling it claims to re-measure).

**Methodology learnings (AGENTS.md discipline):**
- Probes MUST emit progress beats (added: chart-count watcher thread).
- Probe configs must match the production tiling they claim to re-measure;
  a "wide band" for speed invalidates the comparison.
- The tree gate and training probes contend for the same box; run serially.
