# Build Brief: Steps 2, 4, 8 — final gates before training

## Mission

Clear the three remaining steps (2, 4, 8) blocking step 9 (full training).
All are verification/measurement tasks, not feature work.

## Step 2: Validate tube fraction (f_max, f_floor)

Step 3 already shipped with `f_max = 0.40`, `f_floor = 0.16`. This step
validates those values by measurement rather than re-deriving them.

**Task:** Write a driver script `scripts/measure_tube_fraction.py` that:
1. For representative gammas (0.05, 0.1, 0.2, 0.4, 0.7 for positive parity;
   1.1, 1.3, 1.5, 2.0 for saddle), pick one arc per gamma.
2. At each arc, sweep `f = eta / R_c` from 0.05 to 0.6 in 12 steps.
3. At each f, build a minimal tube chart (n_theta=6, n_u=6, n_gamma=1,
   w_nodes_per_decade=6) and measure held-out envelope eps.
4. Report: for each gamma, the f where eps crosses `tube_eps_max = 5e-2`.
5. Confirm that `f_max = 0.40` sits below the crossing for ALL gammas tested.
   If not, report the binding gamma.

**Acceptance:** `f_max = 0.40` is validated (or a new value is proposed).

## Step 4: Validate far-zone crossover (rho*)

Step 5 (C8) already shipped with a measured `rho*` derived from the ppGO map's
`exterior_rho_min`. This step confirms that the tube+farfield coverage reaches
that boundary.

**Task:** Write `scripts/measure_farzone_crossover.py` that:
1. For the same representative gammas, measure the served fraction at
   `rho = rho_exterior_min` (the C8 boundary).
2. Confirm that for each gamma band, at least one chart type (tube or
   far-field) serves at that boundary.
3. If gaps exist, report which gamma/rho combinations fall through.

**Acceptance:** No gamma in the prior has a gap between tube ceiling and
far-field floor (or the gap is documented with a plan to close it).

## Step 8: Part 0 mechanical test

**Task:** Write `cogwheel/tests/test_lensing_part0_mechanical.py` that asserts:
1. No public constant in `cogwheel/lensing/` has a value that traces to
   `ANNULUS_INNER_RADIUS` (3.0), `PRIOR_BOX_HALF_WIDTH` (3.0/sqrt(2)),
   or any absolute source-plane length not derived from the geometry.
2. No live document (`SPEC.md`, `COVERAGE_DESIGN.md`, `DATA_CONTRACTS.yaml`)
   names a retired concept (`annulus`, `ANNULUS_INNER_RADIUS`, the old
   gamma fences `0.75`, `1.0502342`).
3. No constant in `surrogate_training.py` or `surrogate.py` has a docstring
   that mentions "discretization error", "sampling artifact", or "safety
   factor for..." (the bug-class signature from step 1's rule).

This is a GREP/AST test, not a numerical test. It should run in <1s.

**Acceptance:** Test passes. Any false positives are excluded with inline
`# Part0: exempt <reason>` comments.

## Constraints

- Fast tests only (no TRAIN_TIER, no slow sweeps).
- The measurement scripts go in `scripts/` (not tests — they're one-shot drivers).
- Follow AGENTS.md and the spec/TODO workflow.
- On completion, update the TODO step entries to DONE with commit SHA.
