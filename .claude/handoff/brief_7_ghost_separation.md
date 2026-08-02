# Build Brief: Step 7 — _GHOST_SEPARATION_MIN = 0.7

## Mission

Per the TODO: "Ask the Part 0 question. F027 showed it never binds on
the saddle. Re-derive as relative, or delete."

The constant `_GHOST_SEPARATION_MIN = 0.7` is a minimum image-position
separation threshold for admitting the ghost contribution. F027 showed it
never binds on the saddle parity. Determine whether it should be:
- Re-derived as a relative/dimensionless quantity (caustic-relative), or
- Deleted entirely (if the new decay gate from step 6 subsumes it)

The "test-heavy step" warning: 22 references across test_lensing_ghost_gate.py
and test_lensing_exterior_windows.py.

## In scope

- Investigate whether _GHOST_SEPARATION_MIN ever binds in practice now that
  the decay gate (step 6) exists.
- If subsumed: DELETE it and update the 22 test references.
- If still needed: re-derive as a dimensionless ratio (Part 0 principle).
- Tests verifying the decision.

## Out of scope

- The decay gate itself (step 6, just shipped).
- Part 0 mechanical test (step 8).
- Training artifacts.

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
- The Part 0 question: "does this constant trace to physics or to the prior box?"
