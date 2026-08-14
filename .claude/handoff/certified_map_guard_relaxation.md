# Build: per-cell relaxation of the certified map's saddle rho<1 guard (evidence-keyed)

## Mission

`CertifiedPpgoMap.w_cert` hard-refuses ALL saddle `rho < 1` (`ppgo_map.py`,
the F073-era defense-in-depth: `if parity == 'saddle' and rho < 1.0:
return UNKNOWN`). The artifact HOLDS three CERTIFIED saddle rho<0.5 cells,
and the driver's F080 re-validation shows they are NOT equal: one is clean,
one marginal, one contaminated. Replace the blanket parity x rho refusal
with PER-CELL relaxation keyed on re-validation evidence, so the clean
cell(s) serve as the second certification layer while everything
unvalidated keeps the F073 refusal.

## Measured facts (F080, driver pilot 2026-08-14, HEAD 3bca34a; raw data
in the pilot record — pairing gate 1.15e-7 green before every claim;
oracle = training-recipe ppGO vs exact Schwinger, 5 uniform in-box 2-image
configs/cell, 8 nodes on [w_cert, 58])

1. gamma [1.157, 1.339] x rho [0, 0.5], w_cert 19.164: CLEAN — 5/5
   configs, sup err 8.7e-5 < the 1e-4 bar. Eligible for relaxation.
2. gamma [1.339, 1.550] x rho [0, 0.5], w_cert 15.934: MARGINAL — 2/5
   configs at 1.0-1.4e-4 at the w_cert node only, under bar by w = 27.7.
   Relaxation permitted ONLY with a raised effective floor (e.g. serve at
   w >= 27.7 for this cell) or after a denser driver re-measurement —
   decide from the data, state the choice; never serve at the shipped
   15.934 on today's evidence.
3. gamma [1.100, 1.157] x rho [0, 0.5], w_cert 27.721: CONTAMINATED —
   sup err 4.49e-1 (3.5 orders over bar) at the lower-gamma-edge x
   transverse-angle x rho~0.3-0.5 corner; would not re-certify at its own
   center today (fan-worst 1.21e-4 > bar). STAYS REFUSED until the 7a
   retrain re-measures with edge-biased worst-over-cell sampling.
4. Root cause is a TRAINING-METHOD defect (F080): `_measure_cell` samples
   ONE center config per cell; a cell's edges can be orders worse. Also
   on record: the center fan's mirrored angles disagree 2.4x under exact
   D2 symmetry (fan asymmetry — Professor question, routed to 7a's
   retrain; NOT this build's to fix).
5. All other saddle rho_lo < 1 cells are BEYOND_WALL or INVALID — the
   guard's blast radius is exactly the three cells above.

## Scope

IN:
- The relaxation mechanism: per-cell, keyed on explicit re-validation
  evidence — a conservative shape is an allowlist of (parity, gamma-band,
  rho-band) -> effective serve floor, stored IN CODE next to the guard
  with the F080 provenance per cell (cell 1: w_cert as shipped; cell 2:
  floor raised to 27.7 if relaxed; cell 3: absent = refused). Do NOT
  invent a new artifact schema for this (the retrain re-ships the map
  properly; this is the bridge), and do NOT touch the artifact npz.
- Consumers: `w_cert`/`w_trust`/`w_ceiling` route consistently; census
  mirror moves in the same build (served == counted).
- Fast decision-level tests: the guard flips exactly on the allowlisted
  cells (two-sided: allowlisted cell serves at/above its floor, refuses
  below; non-allowlisted saddle rho<1 still UNKNOWN); values-not-paths.
- ONE evidence report: re-run the driver's re-validation for the
  relaxed cell(s) against the shipped gate (~1 min, priced) showing the
  bar holds at the effective floors.

OUT: retraining (7a owns it, with the F080 edge-biased binding); the fan
asymmetry (7a Professor question); any change to `_measure_cell`; the
F073 exterior ghost physics; artifact schema changes; positive-parity
cells (F075's 32-cell advisory is over-conservative direction — 7a).

## Acceptance

- Saddle rho<1 draws in cell 1 (and cell 2 iff relaxed with its raised
  floor) consult the map and serve where the evidence says; cell 3 and
  every unvalidated saddle rho<1 cell still refuse UNKNOWN.
- Two-sided flip tests per relaxed cell; census counts match serves.
- The allowlist carries per-cell provenance comments naming F080 and the
  measured numbers; full fast suite green.

## Constraints

Branch claude-dev; fragments (closes
`todo.d/lensing_certified_map_guard_relaxation.md`, `[→ spec]`);
values-not-paths; pairing gate before any oracle claim; measurement
belongs driver-side — the calibration evidence is handed in; if a WP
believes it needs a new scan, escalate, do not iterate.
