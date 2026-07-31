# Build Brief: 1b — Training-Path Consumers of Analytic Derivatives

## Mission

Retire the numerical derivative estimators in `cogwheel/lensing/surrogate_training.py`
that compute quantities now available in closed form from `geometry.caustic_derivatives`
(shipped in build 1a, commit `1a82046`). The targets are `_branch_speed_profile`,
`_find_cusps`, and `_caustic_inradius`.

## In scope

- Replace `_branch_speed_profile`'s `np.gradient` over sampled caustic points
  with direct calls to `caustic_speed` from `geometry.caustic_derivatives`.
- Replace `_find_cusps`'s speed-minimum detection (relative threshold
  `_CUSP_SPEED_REL_FRAC`, safety factors `_CUSP_WIDTH_SAFETY`,
  `_CUSP_MIN_HALFWIDTH`) with analytic root-finding of `|y'(theta)| = 0`.
- Replace `_caustic_inradius`'s cloud minimum with exact
  `nearest_caustic_point` or the derived bound described in the TODO.
- Delete any constants that lose their reason to exist once the above land.
- Tests verifying the new behaviour (tolerance-based against the geometry, not
  byte-identity with the retired estimators).

## Out of scope

- `_pearcey_cusp._cusp_vertex` (serving path — that's build 1c).
- Third-order `y'''` (also 1c).
- Cusp-window schema changes (deferred per TODO).
- Any training runs or artifact generation.
- The interior-admission distance optimization (measure-first, per TODO).

## Measured facts

- `caustic_speed` and `caustic_derivatives` are available from
  `cogwheel.lensing.chang_refsdal.geometry` (build 1a).
- `_CLOUD_MARGIN_FRAC` and `_PROBE_ETA` are already deleted.
- `_find_cusps` is called from ~6 sites in surrogate_training.py.
- `_branch_speed_profile` is called from ~4 sites.
- `_caustic_inradius` is called from 1 site.

## Acceptance

Per the TODO item:
- The `eta_max > 0.5 * r_min` decision flips on NO production band.
- `stable_gamma_bands((0.01, 0.30), +1)` drops zero slivers AND every stable
  band it returns yields `len(arcs) > 0`.
- Deleting `_CLOUD_MARGIN_FRAC` changes no admission decision (already true).
- Do NOT assert byte-identity with any incumbent estimator.
- Cusp angles agree with the incumbent detector to within the incumbent's own
  resolution, then pinned to the ANALYTIC value at 1e-10.

## Constraints

- Fast tests only (no COGWHEEL_BRUTE_ACCURACY, no COGWHEEL_STRICT_TIMING).
- Do not train any chart artifacts.
- Follow AGENTS.md and the spec/TODO workflow.
