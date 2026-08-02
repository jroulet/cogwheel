# Build Brief: Step 8 — Make Part 0 Mechanical

## Mission

Write a test that mechanically enforces the Part 0 principle:
1. No length-unit float in `cogwheel/lensing/` traces to the prior box.
2. No live document or public symbol names a retired concept.
3. No constant in the geometry or training path exists to absorb a
   discretization error.
4. No decision with a closed form is taken by stepping/sampling.

This prevents the bug class that arrived by accretion (one plausible constant
at a time) from returning.

## In scope

- A test (or test suite) in `cogwheel/tests/` that:
  - Scans all module-level numeric constants in `cogwheel/lensing/`
  - Asserts none of them are the prior-box half-width (4.2426, 3.0) or
    derived from it, unless explicitly cleared (allowlist)
  - Scans `__all__` exports and class/function names for retired concept names
    (annulus, prior_box, etc.) — uses the retired_concepts.json registry
  - Optionally: AST-scan for `np.gradient`, `np.diff` on closed-form quantities
    (the "no stepping" rule)
- Integration with the pre-commit hook's retired-concepts check (already exists
  at `.claude/hooks/check_retired_concepts.py`)

## Out of scope

- Training (step 9).
- Fixing any violations found (they should already be fixed by steps 1-7).
- The collocation sub-builds (1e-eta, 1e-w, 1e-gamma).

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
