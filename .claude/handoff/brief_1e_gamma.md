# Build Brief: 1e-gamma — Gamma Axis Collocation

## Mission

The gamma axis is currently uniform (`linspace(gamma_lo, gamma_hi, n_gamma)`).
The caustic's extent varies 28× across the prior and DIVERGES at the parity
wall (F036). The measure should be the same caustic-relative coordinate that
C8 established. After C8, the caustic-relative `rho = |y|/reach` is the
natural gamma axis coordinate.

Per the collocation fragment: "this one composes with step 5 (C8) rather
than duplicating it — run it AFTER C8, not before."

## In scope

- Replace the uniform gamma grid in tube/far-field/lobe chart construction
  with a grid that is uniform in a caustic-relative quantity (e.g., uniform
  in `1/reach(gamma)` or uniform in `log(reach(gamma))`, so nodes cluster
  near the parity wall where the caustic changes fastest).
- Determine the correct caustic-relative gamma measure from the physics.
- Update `_uniform_axis` calls for gamma or replace with the new grid.
- Tests verifying the new placement improves interpolation quality.

## Out of scope

- 1e-eta, 1e-w (separate builds).
- Training (step 9).
- The spatial axes (already done: tube=arc-length, farfield=s/d, lobe=sqrt-edge).

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
