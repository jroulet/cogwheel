# Build Brief: 1e-lobe — Lobe-Interior Chart Interpolation Coordinate

## Mission

Replace the lobe-interior chart's uniform `theta_local` grid with the
wedge-edge reparametrization `s = sqrt(theta_max - theta)` so the spline
interpolates in a coordinate where the caustic is regular. This removes the
F042 knife-edge near the deltoid cusps for macro-saddle lobe-interior charts,
completing the spatial-axis collocation for all three chart types.

## In scope

- `_build_lobe_chart` in `cogwheel/lensing/surrogate_training.py`: replace
  the uniform `theta_local` grid with a `s = sqrt(theta_max - theta)` grid
  (same reparametrization move as `u = sqrt(eta)` on the fold axis).
- Carry a theta↔s map in the LobeInteriorChart (same pattern as the tube
  chart's arc-length map and the far-field chart's arc_map).
- Update serialization (npz round-trip) for the new map.
- Update `_evaluate_chart` to map through the stored coordinate at serve time.
- Tests verifying the coordinate round-trip and that held-out eps is
  insensitive to small bound shifts (the F042 criterion).

## Out of scope

- 1e-eta, 1e-w, 1e-gamma (separate builds per the ordering).
- Any training runs or artifact generation.
- The positive-parity charts (tube already done, far-field already done).
- Step 2 driver measurements (depend on 1e but are driver-only).

## Measured facts

- `_build_lobe_chart` lives in `surrogate_training.py` (macro-saddle only).
- `LobeInteriorChart` is defined in `surrogate.py`.
- The wedge edge is `theta_max = (1/2) arcsin(lam / |gamma|)` (F044).
- `s = sqrt(theta_max - theta)` makes the edge a REGULAR point (F044:
  `y` and `dy/ds` both finite and nonzero in `s`).
- Build 1d deleted `_WEDGE_EPS` on the strength of this measurement.
- The tube chart's arc-length map pattern (N_map=2001, strict monotonicity,
  trapezoid quadrature) is the template to follow.

## Acceptance

- Lobe-interior chart held-out eps insensitive to a small theta-bound shift
  (the F042 knife-edge gone — the same acceptance as 1e-tube).
- Spline interpolates IN `s`, not merely sampled at `s` points.
- The stored theta↔s map is serialized, round-trips through npz, and is
  strictly monotone.
- Served lobe values unchanged to existing tolerance (no regression).
- `_WEDGE_EPS` remains deleted (build 1d already removed it).

## Constraints

- Fast tests only (no COGWHEEL_BRUTE_ACCURACY, no COGWHEEL_STRICT_TIMING).
- Do not train any chart artifacts.
- Follow AGENTS.md and the spec/TODO workflow.
