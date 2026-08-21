---
date: 2026-08-21
bump: patch
---

### `low_w_diffractive_chart` residual representation corrected to the fold/cusp reference

The `low_w_diffractive_chart` entry's description updated from the stale
point-mass anchor to the code's actual representation: schema
`low_w_diffractive_v2` (was `v1`), residual `r_new = f_pure *
sqrt(1-gamma'^2) / F_ref(w)` with `F_ref = fold_cusp_reference` — the uniform
Airy fold q=p Wronskian form, or the uniform Pearcey cusp form where the fold
degenerates (`b3 -> 0`), declined (cell marked unbuildable in training /
exact-engine fall-through at serve) when both forms fail or the non-vanishing
guard `min|F_ref|/max|F_ref| >= _NON_VANISHING_MIN_RATIO` trips — the serve
re-modulation `F = mass_sheet_phase * F_ref(w) * sqrt_mu_full * r_pure`, and
the `w^(2/3)` frequency axis / `w23_grid` field (was `log_w_grid` / `log w`).
