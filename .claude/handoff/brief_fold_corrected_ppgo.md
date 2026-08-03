# Build Brief: Fold-Corrected ppGO for Degenerate-Delay Interior Pairs

## Mission

Replace the raw ppGO contribution of the merging fold pair with
`fold_amplification` (Airy uniform asymptotic) in the above-split ppGO band.
This fixes the persistent 7% error at caustic-axis angles where the fold pair
has small ξ = (3wΔτ/4)^{2/3} and standard ppGO breaks down.

## The physics (Professor analysis)

The 7% error at ±π/2 angles is NOT an accidental delay degeneracy — it's a
fold pair on the astroid y-axis where the merging images have Δτ → 0. The
standard ppGO uses √|μ| (divergent near fold) while the correct asymptotic
is the Airy form (w^{1/6} scaling, finite at ξ=0). The existing
`fold_amplification` already computes this correctly.

## In scope

1. Add a `fold_ppgo_correction(w, source, gamma, beta, kappa)` function that:
   - Computes standard ppGO (all images)
   - Identifies the merging fold pair
   - Computes fold_amplification for that pair
   - Returns: standard ppGO minus pair's ppGO plus fold_amplification

2. Wire into `born_carrier_from_partition`'s above-split block: when the fold
   pair has ξ < threshold (e.g., ξ < 8), use the corrected version.

3. Relax `_uniform_error_estimate`'s `ξ > 0` refusal to allow ξ = 0 (the
   Airy form is exact on the fold, not invalid).

4. Verify: the ppGO error at axis angles drops from 7% to O(1%) or better.

## Out of scope

- Changing the chart architecture
- The crown band (high gamma) — separate issue
- Training

## Measured facts

- At gamma=0.5, rho=0.7, angle=π/2: ppGO error is 7% at w=54, flat in w
- fold_amplification exists in `_airy_fold.py` and handles ξ=0 (returns Ai(0))
- The current refusal gate requires ξ > 0 — this is conservative, not physical
- After fix: ppGO interior cells should certify in the map, enabling band-split

## Acceptance

- ppGO error at axis angles < 1% (down from 7%)
- Interior cells (rho < 1) certify in the ppGO map
- Census served fraction > 0 for the smoke surrogate

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
