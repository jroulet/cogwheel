# Build Brief: Wire Born Residual Chart into Serve Path

## Mission

The Born carrier code is landed (commit 31ee133). C8 is now done (the annulus
is retired, coordinates are caustic-relative). What remains: wire the residual
chart `F_exact - F_carrier` into the serve path so that when a trained
residual artifact exists, it is used; when absent, exact evaluation continues
(correct graceful degradation).

Per the TODO: "re-derive the registration/accuracy gate in the residual
currency and remove the fall-through at the fact-4 slot in
`likelihood.py::_surrogate_coefficients`"

## In scope

- Re-derive the registration/accuracy gate for the Born residual in the
  caustic-relative coordinate (rho, not absolute |y|)
- Wire `born_carrier_from_partition` + a residual chart lookup into
  `likelihood.py::_surrogate_coefficients` at the fact-4 slot
- The serve path should: check if a residual chart exists and serves this
  (gamma, rho) → if yes, return carrier + interpolated residual; if no,
  fall through to exact (current behavior, preserving correctness)
- Update the census to reflect the new serve path (the 'born' category
  should distinguish "carrier+residual served" from "carrier-only exact")
- Tests verifying the wiring: with no residual artifact → exact (unchanged);
  with a mock residual → carrier + residual returned

## Out of scope

- Actually TRAINING the residual chart artifact (that's step 9)
- The saddle branch Born carrier (separate TODO item)
- Changing the carrier formula itself (already correct)

## Measured facts

- `born_carrier_from_partition` is in `channels.py` (landed commit 31ee133)
- The fact-4 slot in `likelihood.py::_surrogate_coefficients` has a comment
  marking where the fall-through should be removed
- The carrier has 4-5% error at `rho ≈ 12-15 × reach` (step 4 measurement)
- The residual `F_exact - F_carrier` is small and smooth in the transition
  zone, making it suitable for low-node-count chart interpolation

## Acceptance

- With no residual artifact: behavior is IDENTICAL to HEAD (exact evaluation
  in the Born exterior)
- With a residual artifact present: the serve path returns
  `carrier + interpolated_residual` instead of calling the exact engine
- The registration gate checks eps in the RESIDUAL currency (not the raw F)
- Tests exercise both paths

## Constraints

- Fast tests only.
- Do not train any artifacts.
- Follow AGENTS.md and the spec/TODO workflow.
