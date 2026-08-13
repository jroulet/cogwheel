---
section: Backlog
---

- **A FULL EXTERIOR `train()` AT `m_lens_range = (10, 15)` M_sun REGISTERS
  ZERO CHARTS — UNSERVED PRIOR MASS AT LOW LENS MASS, AND ITS ONLY WATCHER
  WAS JUST DELETED** `[→ spec]` — found 2026-08-13 by the test-debt audit
  while clearing fixture rot; NOT investigated, because the answer is
  production-side and the audit was fenced to tests.

  `ValueError: A surrogate needs at least one chart.` Seven of the audit's
  25 errors in `test_lensing_surrogate_training.py` were this, raised from a
  full exterior training run at the low-mass stratum.

  ## This is a COVERAGE GAP, not a fallback

  FALLING THROUGH TO THE EXACT ENGINE IS A FAILED SERVE, NOT A SLOW ONE. The
  surrogate IS the speed layer; production is never supposed to reach direct
  evaluation (standing rule — see
  [[lensing_saddle_gap_is_a_routing_failure_not_coverage]], where the same
  mistake was made and corrected for the saddle gap). Rank this by UNSERVED
  PRIOR MASS, exactly like every other gap. Do NOT record it as "covered by
  the engine".

  ## The mechanism is structural, and it SPREADS with falling mass

  `farfield_w_floor = (RHO_END/2) / min|dtau|` is set by DIMENSIONLESS Fermat
  delays. Those are O(1) and independent of `m_lens` at fixed dimensionless
  `y`: the physical delay factorises as
  `Delta_t = (4 G M_L (1+z) / c^3) * tau`, so the mass lives ENTIRELY in
  `w = 8 pi G M_L (1+z) f / c^3` and `tau` depends only on `(y, gamma,
  kappa)`. (Caveat: mass-independent at fixed dimensionless `y`. Holding a
  PHYSICAL offset fixed while varying mass changes `y`, since the Einstein
  radius goes as sqrt(M_L).)

  So the band `w in [w_lo, w_hi]` slides DOWN linearly with mass past a FIXED
  floor, and the trainer's `[w_floor, w_trust]` window empties. Measured
  floors at `gamma = 0.42`: 1.33 at `rho ~ 1.05`, 0.73 at `rho = 1.5`, 0.46
  at `rho = 2.0`; a 60 M_sun band is `w in [0.119, 7.60]`, so a 10 M_sun band
  is roughly `[0.020, 1.27]`. The NEAR-exterior (where the floor is highest)
  empties first and the hole spreads OUTWARD as mass falls.

  Consequence: no tiling density fixes an empty window. This is the same
  `w_floor` mechanism as [[FINDINGS F070]], seen from the TRAINING side
  rather than the serving side.

  ## The question worth answering is the SIZE, not yes/no

  Because the hole spreads rather than switching on, the useful measurement
  is the fraction of the prior's `(m_lens, rho)` plane where
  `[w_floor, w_trust]` is empty or too narrow to train. That is pure
  geometry — `dimensionless_frequency` + `farfield_w_floor`, no engine calls,
  no training — so it is cheap and re-runnable.

  Report it as unserved prior mass, and state whether the emptied region
  overlaps where the exterior charts are the ONLY rung (if another rung
  covers it, the gap is smaller than the window arithmetic suggests).

  ## Why this needs an owner NOW

  The test that surfaced it has been DELETED (justifiably — it lived in
  `FoldCarrierTrainingIntegrationTestCase`, which ran a full production
  `train()` in `setUpClass`, cost 40 minutes per suite run, and had its own
  claims duplicated by `test_lensing_exterior_polar_fold.py` on sub-second
  fixtures). This failure mode currently has NO watcher.

  Do not restore that test to cover it. The right artifact is a census /
  coverage assertion over the `(m_lens, rho)` plane, not a 40-minute
  integration test.

  ## Acceptance

  Quote the unserved prior mass fraction and the `(m_lens, rho)` region where
  the window is empty. If some other rung covers part of it, say which and
  net it off.
