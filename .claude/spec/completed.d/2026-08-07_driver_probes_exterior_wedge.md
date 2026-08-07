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

**Wedge v3 (`scripts/probe_wedge_v3.py`):**

- **INVALID measurement — misconfigured probe.** Used
  `gamma_band_halfwidth=0.48` (a single giant band, 12x production width)
  with `regions=('wedge_interior',)`. All 58 charts report `heldout_eps: nan`
  (zero held-out points served) and 43/58 hit depth-3. The NaN is a
  probe-config artifact (held-out sampling falls outside the giant band's
  tile w-ranges), NOT a v3 regression. The v2-vs-v3 question is UNANSWERED.
- ACTION: re-run the wedge probe at the PRODUCTION `gamma_band_halfwidth`
  (0.02-0.04) before quoting the 18-chart / 5.47e-4 v3 baseline.

**Methodology learnings (AGENTS.md discipline):**
- Probes MUST emit progress beats (added: chart-count watcher thread).
- Probe configs must match the production tiling they claim to re-measure;
  a "wide band" for speed invalidates the comparison.
- The tree gate and training probes contend for the same box; run serially.
