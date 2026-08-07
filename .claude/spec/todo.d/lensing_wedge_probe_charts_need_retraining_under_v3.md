---
section: Backlog
depends_on: [2026-08-07_subdivision-recursion-wedge-v3-r-caustic]
---

- **EVERY WEDGE CHART BUILT BEFORE 2026-08-07 IS UNLOADABLE, SO ANY
  MEASUREMENT RESTING ON ONE MUST BE RE-RUN** `[housekeeping]` — the
  `InteriorWedgeChart` schema bump to `wedge_caustic_relative_v3`
  (`5084e93`) drops BOTH v1 and v2 from the known set and makes `theta_to_u`
  required, so a stale artifact hard-refuses at load by design rather than
  being served on a mislabelled axis. That is the correct behaviour, and it
  invalidates the charts built during the 2026-08-06 coordinate probes.
  AFFECTED, specifically: the 18-chart / median 5.47e-4 / ~10.5 min interior
  result and the 13/16-children subdivision result, both measured with probe
  scripts under the scratchpad against v2 charts. The CONCLUSIONS are very
  likely unchanged — v2 and v3 store the identical `u = d**(2/3)` array under
  different field names, and serve is coordinate-agnostic through the stored
  map — but "very likely" is not measured, and these numbers are the
  justification for the recursion cap and for the interior's eps acceptance.
  ACTION: re-run the interior probe against v3 before quoting those figures
  again, or before they are used as a baseline for the production training
  run. Cheap (~10 min) relative to what rests on them.
  This was a specific instance of the general problem that the training path
  could not be invoked per region (now fixed:
  [[2026-08-07_lensing-training-path-per-region]]). That fix does not change
  the re-run need above: these measurements used the old probes and predate it.
