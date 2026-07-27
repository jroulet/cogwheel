---
section: Backlog
---
- **Port the subdivision test fixture to caustic-fixed coordinates**
  `[housekeeping]` — `cogwheel/tests/test_lensing_farfield_envelope.py` has 4
  failures and 21 errors: every test routed through `_run_subdivision`
  (line ~1974) is dead, so `ExteriorEdgeAnnulusSubdivisionTestCase`,
  `InteriorEdgeAnnulusSubdivisionTestCase`,
  `CancellationChildSubdivisionTestCase`,
  `RegionKeyConsistencySubdivisionTestCase` and
  `SubdivisionSelfFalsificationTestCase` assert nothing.

  NOT a stale-kwarg fix. Three interface migrations landed without the fixture
  following: (1) `exclusion_radius` -> `exclusion_rho`, still a float, same
  slot (`bc27d39`); (2) `interior_admit_radius: float` ->
  `interior_admission: _InteriorAdmission | None` plus new optional
  `exterior_admission` / `source_magnitude_max` — every value the fixture
  passes (1e9 default, 1.5 for the flip case) means "admit everything", which
  the object API expresses as `None`; (3) THE BLOCKER — the tile dict moved to
  caustic-fixed coordinates: `_subdivide_farfield_tile` unpacks
  `rho_c, theta_c = tile['center']` and `half_rho, half_theta = tile['half']`
  (`surrogate_training.py:3011-3012`) while the fixture supplies `(y1, y2)`
  centres and a SCALAR half (`_EXT_PARENT_CENTER = (1.2, 0.0)`,
  `_EXT_PARENT_HALF = 0.4`).

  So the port must rewrite the fixture's geometry — parent centres and halves
  in `(rho, theta_c)`, `_expected_children` child-centre arithmetic, and the
  "inner children disk-excluded / outer admit" construction that currently
  relies on Cartesian distance from the origin. Re-derive the exclusion
  semantics in the caustic-fixed gauge rather than transliterating: `rho` is
  multiplicative inside the caustic, additive outside, and the ppGO map uses a
  third (scalar-reach) convention.

  Introduced by `bc27d39` (8h-b4) on top of the 8h-b3 caustic-fixed migration.
  Silent because the suite was never run after that commit and nothing in the
  fast tier exercises `_subdivide_farfield_tile`.

  Acceptance: all 25 tests green, each demonstrably reachable-red (this file's
  self-falsification class must actually fire), and the fixture's coordinate
  gauge named in a comment so the next migration breaks loudly.
