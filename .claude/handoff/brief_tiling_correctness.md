# Build 8h-d2: tiling correctness — ppGO annulus gauge, carrier continuity, cusp columns

## Mission

Four defects in the exterior tiling/training path, all of which change WHAT
gets served and HOW ACCURATELY, and all of which must land before the coverage
census measures anything. Each is silent: none crashes, none produces a NaN,
each leaves magnitudes and shapes looking correct.

## Measured facts (driver-measured 2026-07-27; do NOT re-derive)

1. STALE ppGO ANNULUS EDGE. `surrogate_training._train_band_charts` computes
   `ppgo_exclusion_rho = physical_exclusion_radius / reach_scalar` (~line 3266)
   BEFORE the positive-parity branch narrows the region, and never recomputes
   it from the narrowed `region_exclusion_rho` (~line 3323-3325, the per-column
   directional admission introduced in 8h-b4). The ppGO trim
   (`_stratum_ppgo_boundary` / `_stratum_ppgo_ceiling`, ~3283-3286) therefore
   reads `w_trust` / `w_ceiling` from an annulus cell FARTHER OUT than the
   region actually covers -- an easier cell, hence a lower `w_cert`. Charts are
   then capped or dropped at a frequency where ppGO is NOT certified for their
   inner columns. The comment at ~3279-3282 states the safety argument
   ("certifying THERE implies the easier outer regions are covered") that the
   `parity == 1` rewrite invalidates. `ppgo_exclusion_rho` has ZERO test
   references anywhere in `cogwheel/`.
2. `rho` MEANS THREE DIFFERENT THINGS and no converter is authoritative:
   `ppgo_map.py:195` multiplicative scalar-reach `rho = |y| / caustic_reach`;
   `surrogate.py:286-291` multiplicative DIRECTIONAL inside the caustic
   (`|y| / r_caustic`), ADDITIVE outside (`1 + |y| - r_caustic`), additive
   scalar-reach for the saddle exterior; `surrogate_training.py:2143-2145`
   lobe-local `rho_lobe in [0, 1]`. Two sites hand-roll the conversion into the
   ppGO gauge (`likelihood.py:1375` and `surrogate_training.py:3266`), and
   `CertifiedPpgoMap.w_cert(parity, gamma, rho)` takes a bare float and cannot
   tell the gauges apart. The test suite itself builds `exclusion_rho` in two
   mutually inconsistent gauges for the same callee.
3. FAR-FIELD LABEL IS FRAME-DEPENDENT, and only the INTERIOR branch is guarded.
   The interior (SACR-C) label is `tau_c`-demodulated and algebraically
   frame-invariant; the far-field label, with the carrier parked at
   `tau_c = 0`, is NOT -- each node stores its value in its own `t_min(x)`
   frame and the spline mixes them. `_assert_carrier_continuity`
   (`surrogate.py:561-620`) exists but is called only under `if interior:`
   (~1310-1315). Measured along the exterior arm at `gamma=0.30`,
   `theta_c=0.4`: `d t_min/d rho ~ -1.03`, and with `n_rho = 4` /
   `n_farfield_tiles_per_side = 5` the node gap in `rho` is ~5e-2, so
   `w * delta(gauge)` is ORDER RADIANS by `w ~ 60`. The gauge is not a small
   correction; it is the dominant spatial phase of the fitted object. The only
   backstop is the held-out `eps` gate, which samples the same tile and can
   under-sample the same oscillation.
4. CUSP-ALIGNED COLUMNS NOT WIRED INTO ONE CHART PATH. The positive-box
   reconstruction config `(gamma=0.40, y1=2.183, y2=0)` sits exactly on the
   `theta_c = 0` cusp ray, where `r_caustic` has a slope kink a cubic spline
   cannot represent while the ray falls in a CELL INTERIOR: eps 2.6e-1 against
   a 0.2 budget. 8h-b6 added cusp-ALIGNED exterior columns, which put the ray
   on a column EDGE and collapse eps to ~1.5e-4 (a 1700x improvement),
   certified structurally by
   `test_lensing_exterior_admission.OnCuspColumnEdgeTestCase`. That fix is not
   reaching the chart built by `test_lensing_surrogate.py`'s positive-box
   fixture, which is currently `@unittest.expectedFailure` with a marker
   saying NOT to widen `POS_RECON_TOL`.

## In scope

- Derive `ppgo_exclusion_rho` FROM the narrowed `region_exclusion_rho` for
  positive parity, so the ppGO trim reads the annulus the region actually
  covers. Add the report-path assertion that the two agree.
- ONE authoritative converter into the ppGO gauge -- e.g. an exported
  `ppgo_map.annulus_rho(gamma, y_magnitude, kappa=0.0)` -- and route both
  hand-rolled sites through it. Follow the `_FARFIELD_AXIS_SCHEMA` precedent
  (`surrogate.py:198`), which stamps the caustic-fixed gauge and rejects a
  stale artifact at load naming the three wrong gauges: consider the same
  stamp for the ppGO map artifact.
- Extend `_assert_carrier_continuity` to the EXTERIOR branch with `t_min` (or
  `w_max * delta t_min` per node gap, in radians) as the tracked quantity, or
  -- if the Professor judges the exterior label should instead be made
  frame-invariant -- do that and say so. Either way the far-field label's
  frame-dependence must stop being implicit.
- Wire cusp-aligned columns into the chart path that
  `test_lensing_surrogate.py::test_positive_box_reconstruction_within_budget`
  exercises, and REMOVE its `@unittest.expectedFailure`. Do not widen
  `POS_RECON_TOL`; the marker says so and it is correct.

## Out of scope (do NOT touch)

- The ghost gate (8h-d1, just landed), Born (`_born.py`, `b1`), the saddle
  lobe-frame serve wiring, the census, any campaign or engine production run.
- The uniform Airy/Pearcey arms.
- Structural test classes: `test_lensing_surrogate_training.py` and
  `test_lensing_farfield_envelope.py` carry PROVISIONAL headers saying not to
  contort production to keep their bookkeeping green. If a structural test
  there breaks, update or delete it -- do not widen production for it.

## Acceptance

- The ppGO trim's annulus edge provably equals the region's actual inner edge
  for positive parity, asserted, with a reachable-red proving the pre-fix
  value differed.
- Both former hand-rolled `rho`-into-ppGO conversions call the one converter;
  a wrong-gauge value is rejected or impossible rather than silently plausible.
- The exterior carrier gauge is either asserted continuous per node gap (in
  radians, with the measured `d t_min/d rho ~ -1.03` as the calibration) or
  made frame-invariant -- with a test either way.
- `test_positive_box_reconstruction_within_budget` passes with
  `POS_RECON_TOL` UNCHANGED and its `expectedFailure` removed; eps at the
  on-cusp config drops from 2.6e-1 toward the ~1.5e-4 that 8h-b6 measured.
- Existing suites green: `test_lensing_ppgo_bandsplit.py`,
  `test_lensing_exterior_admission.py`, `test_lensing_exterior_windows.py`,
  `test_lensing_channels.py`, `test_lensing_surrogate.py`.
- Full suite: driver-verified POST-BUILD.

## Constraints

- HARD test-tier ceiling: any single test < 60 s, any FILE < 5 min, fast tier.
  Engine-backed training fixtures are behind `COGWHEEL_TRAIN_TIER=1` and are
  the DRIVER's, not a build's -- do not add fast-tier tests that call `train`
  or `_build_farfield_chart`.
- Any new convention gets ONE authoritative expression plus an assertion that
  consumers agree. Today produced three separate bugs of exactly this class
  (delay frame at four sites, the distance convention, the ghost gate's
  caller-dependent `min(w)`); do not add a fourth.
- Values derived from `(source, matrix)` belong ON the partition, not
  re-derived inside hot-path functions -- that cost ~250 us per likelihood
  evaluation twice today.
- Accuracy dominates; units and conventions per AGENTS.md; numba compatibility
  preserved.
- Branch `claude-dev` only. Never commit on `main`.
