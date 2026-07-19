# Professor short-term — 2026-07-18 (Build 6 verification gate review)

## Session task
Reviewed and corrected 7 gate specs for the Build 6 negative-parity verification plan.

## Observations

1. **geometric_amplification docstring stale**: claims `LensDomainError` if
   `1 - kappa <= abs(gamma)`, but the underlying `macro_matrix` now accepts BOTH
   parities (only refuses lam<=0 and |gamma|==lam). The geometric branch works
   correctly on saddle configs — the docstring just wasn't updated. NOT a code bug.

2. **Eigenframe rotation convention**: `_saddle_grid` uses `exp(-1j*beta)` to rotate
   the rescaled source y_scaled into the shear eigenframe. The positive-parity
   `_mass_sheet_map` path (inside `_grid_certified`) does the SAME rotation — verified
   consistent. The sign is correct: shear matrix Q(beta) has eigenvectors at angles
   beta and beta+pi/2, so rotating by -beta aligns to principal axes.

3. **Geometric takeover condition**: `w > W_CEILING_SCHWINGER AND w*delta_min >= RHO_END`
   (= 4.0). This means takeover happens when the SMALLEST pairwise delay separation
   times w exceeds 4. For a 2-image saddle config with typical delta_min ~ 0.5-2,
   takeover is at w ~ 2-8 (well below the ceiling 60). For the geometric branch to be
   actually INVOKED, w must ALSO exceed 60 — so the test must arrange w > 60
   specifically. The resolution condition alone doesn't trigger geometric; both must hold.

4. **Index sum physics**: For a Chang-Refsdal with indefinite macro matrix (saddle host),
   the Euler characteristic of the potential gives sum(-1)^n_a = sign(det A) - 1 = -2
   (not sign(det A) = -1). The 2-image case: both are saddles (n=1), so sum = (-1)^1 +
   (-1)^1 = -2. The 4-image case: one minimum (n=0) + three saddles (n=1), so sum =
   (+1) + (-1) + (-1) + (-1) = -2. Confirmed: -2 is the CORRECT signed Morse sum for
   BOTH topology sets individually.

5. **G6 at w=13 issue**: The geometric branch is NEVER invoked at w=13 in the delivered
   code — the condition requires w > 60 AND resolved. So G6 must compare
   `geometric_amplification(w=13,...)` (called DIRECTLY, bypassing `_saddle_grid`) vs
   `f_schwinger` at the same point, to verify the asymptotic formula's accuracy at
   that w*dtau. The _saddle_grid function itself would use Schwinger at w=13.
