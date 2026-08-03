# Build Brief: Schwinger quad-double extension (w > 60 saddle)

## Mission

The Chang-Refsdal Schwinger integral evaluator refuses above w ~ 60 on
the saddle branch (det A < 0) because double-precision oscillatory
quadrature loses all digits. This means charts CANNOT BE TRAINED above
that ceiling — region 11 in the coverage map is structurally blocked.

The fix is a quad-double (128-bit) precision extension of the Schwinger
integrator, previously owner-approved (2026-07-22).

## Context

- The Schwinger integrator lives in `cogwheel/lensing/chang_refsdal/_schwinger.py`
- It uses `scipy.integrate.quad` with double precision
- At high w the integrand oscillates at frequency w, requiring O(w) nodes
  for convergence — at w=60 this exceeds double-precision's ~15 digits
- mpmath or a compiled quad-double library can push to w ~ 155

## Implementation

1. Add an optional `quad_double=True` path in the Schwinger evaluator:
   - When `w > _SCHWINGER_DD_THRESHOLD` (~ 55-60), switch to mpmath
     `quadgl` or a FLINT-based evaluator
   - Return results as float64 (the RESULT is still O(1), only the
     intermediate quadrature needs extra precision)

2. Measure the new ceiling: sweep w from 60 to 200 at representative
   saddle configurations, find where even quad-double loses convergence.

3. Update `_SADDLE_W_CEILING` from 58 to the measured new ceiling.

4. Ensure the training pipeline respects the new ceiling for saddle charts.

## Acceptance

- Schwinger evaluator returns correct results at w = 100, 120, 150
  (verified against a reference at w = 50 where double precision works).
- New ceiling documented and wired into training config.
- Performance: quad-double path is 10-50x slower per call (acceptable for
  training; the serve path uses ppGO/charts, not direct Schwinger).

## Constraints

- This is HEAVY: may need mpmath or a compiled dependency.
- If mpmath is too slow (>1s per call), investigate FLINT or arb.
- Fast test suite stays fast (mock or skip the qd tests in CI).
- Follow AGENTS.md and the spec/TODO workflow.
