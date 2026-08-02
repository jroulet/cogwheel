# Build Brief: 1e-eta — Eta Axis Collocation (the Airy Uniformizing Coordinate)

## Mission

Per the collocation fragment: "Derive xi = (3 w Delta_tau / 4)^{2/3} and state
u = sqrt(eta) as its w-independent shadow. Mostly explanatory: the incumbent
axis already works, so the deliverable is WHY it works and WHERE it does not,
plus the DRY test."

The tube chart's eta axis uses `u = sqrt(eta)` — which is already the correct
coordinate change (the fold's sqrt-branch is smooth in u). The deliverable is:
1. Document WHY u = sqrt(eta) works (it's the w-independent shadow of the
   Airy control xi = (3 w Δτ / 4)^{2/3})
2. A DRY test asserting the collocation coordinate equals the arm's own
   control to machine precision where they overlap
3. Document WHERE it does not work (near cusps, where the Pearcey control
   takes over)

## In scope

- Documentation/derivation showing u = sqrt(eta) is the w-independent
  Airy uniformizing coordinate
- A test asserting the tube chart's u-grid matches the Airy fold evaluator's
  xi coordinate (DRY: one representation, not two independent derivations)
- Any code that makes the connection explicit (e.g., a comment or a
  shared derivation function)

## Out of scope

- Changing the u-axis (it already works)
- 1e-w (separate build)
- Training

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
