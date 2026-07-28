---
date: 2026-07-28
section: Backlog
---

- **Port the subdivision test fixture to caustic-fixed coordinates** — CLOSED
  as OBSOLETE, not done. The TODO described 25 dead tests
  (`ExteriorEdgeAnnulusSubdivisionTestCase`,
  `InteriorEdgeAnnulusSubdivisionTestCase`,
  `CancellationChildSubdivisionTestCase`,
  `RegionKeyConsistencySubdivisionTestCase`,
  `SubdivisionSelfFalsificationTestCase`) routed through `_run_subdivision` in
  `cogwheel/tests/test_lensing_farfield_envelope.py`. That suite was deleted in
  `3c107d4` ("tests: delete the retired subdivision suite; root-cause the
  eps-gate fixture"), but the TODO fragment was never removed alongside it, so
  the backlog carried a port task for tests that no longer existed.

  Restore point if the coverage is ever wanted back:
  `git show 3c107d4~1 -- cogwheel/tests/test_lensing_farfield_envelope.py`.

  Housekeeping note for the next migration: a TODO describing a test file is a
  live document about a moving target. Deleting the tests without deleting the
  fragment left a stale instruction in the backlog for a week. Retire the
  fragment in the SAME commit that retires the code it describes.
