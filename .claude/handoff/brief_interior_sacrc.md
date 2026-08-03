# Build Brief: Interior SACR-C Charts (Positive-Parity Astroid Interior)

## Mission

The positive-parity interior (rho < 1, 4 real images, inside the astroid
caustic) has NO working chart type. A P2 pilot (Build 8h-b) recorded 0%
pass at the crown band (eps 3.4, vs bar 0.05). This is the single largest
coverage gap — 66% of census draws fall here.

The TODO identifies three levers:
1. Per-cell measured w-ceilings (already landed as part of ppGO map work)
2. Fold-gated ghost-pair subtraction in the far-field label with component
   grids — the analytic continuation of the 2-image exterior pair INTO the
   interior, subtracted to leave a smoother remainder
3. Caustic-fixed interior coordinates + crown-aware interior banding

The key insight from P1 probe: "the exterior annulus beat IS the
analytically-continued complex image pair (fold-only; 3-6× smoother
remainder; harmful at cusps where Im τ_c = 0)."

## The physics problem

Inside the caustic, F(w) = sum of 4 real-image kernels. The demodulated
envelope oscillates rapidly because the 4 images interfere. A naive spline
of this envelope needs enormous node counts (eps 3.4 at reasonable grids).

The component representation: decompose F into:
- An ANALYTIC part: the 2-image ppGO pair (continued from exterior)
- A REMAINDER: F_exact - F_analytic_pair

The remainder is 3-6× smoother (measured) because the dominant oscillation
is captured by the analytic pair. Spline the REMAINDER only.

## In scope

- Understand the current `_interior_label` and `interior_envelope_from_partition`
  (if they exist) or determine what needs to be built
- Implement the component representation: subtract the analytic pair, spline
  the smooth remainder
- Build interior charts that pass the 0.05 eps bar
- Start with mid-gamma (where eps was 6e-2 in the pilot — close to passing)
  before tackling the crown band

## Out of scope

- Crown-band gamma (where eps = 3.4 — needs separate investigation)
- Saddle interior (already has LobeInteriorChart)
- Training campaign (step 9)

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
- This is the hardest remaining physics problem. Use the Professor extensively.
