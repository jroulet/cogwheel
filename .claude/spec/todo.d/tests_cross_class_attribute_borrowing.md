---
section: Backlog
---

- **TESTS THAT READ ANOTHER TestCase's CLASS ATTRIBUTE ARE GATE-ONLY FLAKES**
  `[housekeeping]` — a test reading `OtherTestCase.chart` is only correct when
  both classes run in the SAME process. The tree gate runs
  `pytest --dist loadscope` (`orchestrator.py`, tiers `-n 4` then `-n 2`),
  which distributes by CLASS, so whenever the scheduler places the two classes
  on different xdist workers the borrowed attribute is unset and the dependent
  test errors with `AttributeError: type object '...' has no attribute '...'`.

  MEASURED 2026-08-12. `test_lensing_ppgo_midw_and_minus_ghost.py`'s
  `MinusGhostServeRoundtripSelfFalsificationTestCase` borrowed
  `MinusGhostServeRoundtripTestCase.chart`. It was:
    green standalone (26/26),
    green under `--dist loadfile` (same file -> same worker),
    green under `--dist loadscope` on that file ALONE (2 workers, no scope
      competition, classes co-located),
    RED on two consecutive full tree gates.
  Reproduced deterministically by running the dependent class BY ITSELF —
  which is exactly what a split worker receives. Confirmed PRE-EXISTING by
  A/B at HEAD with no build changes applied; it entered with `c8cad0c`.

  Cost: it masqueraded as a consequence of a real `KeyError: 'm_lo'` in the
  same gate, so the first gate failure was recorded as a 4-test cascade when
  it was 3 tests plus one independent latent flake. Fixed by giving the
  dependent class its own `setUpClass` and using `type(self).chart`.

  ## The audit this needs

  `grep -rnE "[A-Z]\w*TestCase\.[a-z_]+" cogwheel/tests/test_lensing_*.py`
  returns matches in at least: `test_lensing_airy_fold.py`,
  `test_lensing_caustic_cusps.py`, `test_lensing_exterior_admission.py`,
  `test_lensing_exterior_windows.py`, `test_lensing_operator.py`. Most are
  benign (a class referencing its OWN name, or a shared base). The dangerous
  shape is specifically: class A's `setUpClass` sets an attribute, class B
  reads `A.<attr>`.

  For each, the decisive check is cheap and does not need the full suite:
  run that class ALONE (`pytest file.py::ClassB -q`). If it errors, the
  borrow is real and the test is a scheduling-dependent flake waiting for the
  gate to reshuffle.

  FIX SHAPE: build the fixture in the consuming class's own `setUpClass`, or
  hoist it to a shared base/module-level helper both classes call. Never read
  a sibling TestCase's class attribute.

  WORTH A STANDING GUARD: a one-line grep in the gate or a lint that fails on
  `<OtherClass>TestCase.<attr>` outside the defining class would catch the
  next one at authoring time rather than at a random future gate.
