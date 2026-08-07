---
section: Backlog
---

- **THREE PIECES OF MACHINERY WERE BUILT, PASSED COMPONENT TESTS, AND WERE
  NEVER REACHED — nothing asserts INTEGRATION** `[housekeeping]` — observed
  2026-08-06, three instances in a single day:

  1. `InteriorWedgeChart` — serve-wired and unit-tested, but never produced by
     a training run that registered charts.
  2. `FARFIELD_KERNEL_SUM_MINUS_GHOST` — a declared envelope label that no
     producer ever stamps.
  3. `_subdivide_farfield_tile`'s interior branch — deleted by a brief while
     its callers remained.

  Each passed its own tests. The tests asserted the component in isolation,
  which is precisely what cannot see "nothing calls this".

  CHEAP STANDING GUARDS, both one-line greps, and two of the three would have
  been caught:
  - assert every declared envelope label is stamped by some producer;
  - assert every chart class is constructed by the training path.

  RELATED STANDING REQUIREMENT for the coordinate program
  ([[lensing_coordinate_program_spine]]): every step retires what it replaces
  — no `(s, d)` bridge left reachable after the exterior re-chart, no
  arc-length map left in the wedge path (done 2026-08-07). A grep for retired
  coordinate machinery should return nothing reachable. These guards are how
  that requirement becomes checkable instead of aspirational.
