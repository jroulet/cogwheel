---
date: 2026-08-14
section: Lensing training
---

**`_find_cusps` wrap arithmetic fixed; cusp-arm coverage machinery
retired** `[→ spec]` — build `find_cusps_wrap_fix` (NEXT-SESSION ORDER
3/7; also closes `lensing_cusp_arm_coverage_constant_stale`). The span at
the periodic wrap is now the house mod-2pi idiom (saddle `periodic=False`
path byte-identical); `_EXPECTED_ARCS = {1: 4, -1: 6}` with a
`CausticTopologyError` arc-count check on the untruncated tiler output —
the astroid serves 4/4 arcs at every production gamma (measured windows
0.094/0.141/0.141/0.236 rad at gamma 0.2/0.5/0.7/0.9; theta=0 window
bit-identical to its theta=pi partner). `GOLDEN_INWARD_SIGN` astroid rows
re-frozen to 4-tuples with signs derived through the geometric-two-image-
side test. `_CUSP_ARM_COVERAGE` / `_SADDLE_CUSP_ARM_COVERAGE` + the
`_tube_serves` shrink deleted (registered in retired_concepts.json); four
dead scripts deleted; `census_dry_run.py` cusp_arm routes on the F074
w-floor 49. Build history: Inspector PASS; watchdog-killed mid-Professor
review at the 1200 s staleness threshold (first live firing of the 1/7
group kill — zero orphaned daemons); driver completed the review
(independent PASS, probes green: 49/49 caustic-cusps suite, 4/4 arcs
sweep gamma 0.05-0.95, saddle 6/6 unchanged), fixed the five review
findings, ran the tree gate, and committed.
