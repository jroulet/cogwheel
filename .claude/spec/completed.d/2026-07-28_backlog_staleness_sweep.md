---
date: 2026-07-28
section: Backlog
---

- **Backlog staleness sweep** — five fragments retired because their claims
  are the tree TODAY, not work owed. Audited 2026-07-28 against HEAD; each
  retirement cites the commit that overtook it.

  | fragment | now-false claim | overtaken by |
  |---|---|---|
  | `likelihood_schwinger-homogenization` | "Target architecture: Schwinger as THE single wave evaluator on both parities; the legacy path demoted to ORACLE duty" | Build 8d, 2026-07-21 (`completed.d/2026-07-21_build8d-homogenization.md`; SPEC section HOMOGENIZATION) |
  | `likelihood_serving-microlevers` | "PRE-BRIEF DRIVER STEP: profile the 2.0 ms partition residual" | Build 8f, 2026-07-21; profile scripts committed, residual since cut 460 -> 206 us (`d0dc6da`) |
  | `serving_band-split-ppgo-interior` | "Build: (1) per-node band-split dispatch — charts below w_cert, bare certified ppGO above" | `c6aa6e4` + `dc984c1`, 2026-07-23; `likelihood._ppgo_band_split` is in the tree |
  | `surrogate_farfield-envelope-v2` | "Fix: far-field charts subtract the FULL ppGO sum (switch forced on) with no `tau_c` carrier" | `8a00fd9`, 2026-07-22; that IS the definition, `farfield_eps_max` now 1e-3 |
  | `2026-07-16_lensing-program` | "three-build sequence implementing relative binning for microlensed waveforms" | ~15 builds; program is at 8h. Survived only as a stale umbrella title in *In progress* |

  RESIDUE PRESERVED, not discarded. Two of these carried live clauses that
  were re-homed rather than retired with the fragment:
  * `surrogate_component-representation-8hb` was ~80 % stale (its per-cell
    w-ceilings, ghost-pair subtraction and caustic-fixed interiors all
    shipped; `interior_report['served']` is now True for the saddle via
    `04f9f5c`) but its CROWN-BAND clause is unresolved and coverage-relevant,
    so the fragment is KEPT and narrowed rather than retired.
  * `likelihood_prior-bounds-instantiation` is mostly done or housekeeping,
    but its item-5 tiny-caustic clause names a genuinely unserved region (the
    small-gamma near-caustic collar, where `eta_max = 0.05` is absolute and
    the shrinking astroid makes `_min_curvature_radius` skip the tube chart
    while the far-field excludes the same collar). Also KEPT.

  WHY THIS MATTERS beyond tidiness: a stale TODO reads as work owed. Earlier
  the same day, `lensing_farfield_subdivision_fixture_port` was found to
  describe 25 tests deleted a week before in `3c107d4` — it had been sitting
  in the backlog instructing anyone who read it to port tests that no longer
  existed. Retire the fragment in the same commit that retires its subject.
