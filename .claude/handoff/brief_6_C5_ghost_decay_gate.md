# Build Brief: Step 6 (C5) — Ghost Decay Gate

## Mission

The ghost branch needs a DECAY gate, not just a separation test. Per F027,
the current ghost admission uses `_GHOST_SEPARATION_MIN` (a minimum delay
separation) but lacks a check on whether the ghost amplitude has decayed
enough to be negligible. Add the decay gate.

## In scope

- Add a ghost decay gate that checks the ghost amplitude (not just separation).
- The gate should be based on the ghost's exponential decay rate: at large
  enough |y| (far from caustic), the ghost's complex-saddle contribution
  decays exponentially and becomes negligible relative to the real images.
- Determine the decay threshold empirically or from the physics (the ghost
  amplitude decays as exp(-w * Im(tau_ghost)) where Im(tau_ghost) is the
  imaginary part of the ghost's Fermat delay).
- Tests verifying the gate fires correctly.

## Out of scope

- `_GHOST_SEPARATION_MIN = 0.7` (step 7, separate).
- Training artifacts.
- The mechanical Part 0 test (step 8).

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
- This is independent of the coordinate work — owed on both branches.
