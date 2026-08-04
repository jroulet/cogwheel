# Build Brief: Born residual chart training

## Mission

The Born carrier wiring is complete (C11, commit 3d0e785):
- `BornResidualChart` frozen dataclass exists
- The fact-4 slot in `likelihood._surrogate_coefficients` is plumbed
- When a chart is attached, it reconstructs `F_carrier + R`
- When `None` (current), exterior draws fall through to exact engine

What remains: TRAIN the residual `R = F_exact - F_carrier` on a sparse
grid and attach it. The carrier is analytic (free); the residual is small
and smooth at large rho, so it needs far fewer nodes than a full chart.

## What the residual is

At large rho (far from caustic), the amplification is:
  F(w) ≈ F_carrier(w) + R(w)
where F_carrier = sqrt(mu_macro) * exp(i*w*phi_geo) (the Born carrier,
closed form) and R is the weak-deflection correction (small, smooth,
decays as 1/rho^2).

The residual R varies slowly in (gamma, rho, w) — no fold/cusp structure,
no oscillations beyond the carrier frequency. A coarse grid suffices.

## Implementation

Write `scripts/train_born_residual.py` that:

1. For representative exterior configs at large rho (rho = 2, 3, 4):
   - Sample gamma across the prior
   - Sample w across the band
   - At each (gamma, rho, w): compute F_exact via engine, compute
     F_carrier via born_lead_carrier, store R = F_exact - F_carrier

2. Build a `BornResidualChart` from the residual grid (sparse: maybe
   5×5×10 in gamma×rho×w — the residual is very smooth).

3. Save the chart artifact.

4. Wire it: the likelihood already has the slot, just needs the chart
   loaded at init.

## Acceptance

- Born residual chart serves exterior draws at large rho without
  falling through to exact engine.
- Relative error vs exact: < 1e-3 at rho > 2.
- No regression on other serve paths.

## Constraints

- This IS expensive (needs engine evaluations for the training grid).
  But the grid is small (~250 points × 90ms = ~22s).
- Follow AGENTS.md and the spec/TODO workflow.
