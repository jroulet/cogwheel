---
date: 2026-08-21
bump: patch
---

SPEC.md's LOW-W DIFFRACTIVE RUNGS paragraph (Rung P) corrected from the stale
point-mass anchor to the code's actual residual representation: the chart
stores `r_new = f_pure * sqrt(1 - gamma'^2) / F_ref(w)` with `F_ref =
fold_cusp_reference` — the non-vanishing uniform Airy fold q=p Wronskian form,
or the uniform Pearcey cusp form where the fold degenerates (`b3 -> 0`, the
fold->cusp transition) — declined (exact-engine fall-through) when both forms
fail or the non-vanishing guard `min|F_ref| / max|F_ref| >=
_NON_VANISHING_MIN_RATIO` trips; the serve re-modulation is `F =
mass_sheet_phase * F_ref(w) * sqrt_mu_full * r_pure` and the interpolation
axis is `w**(2/3)` (not `log w`). The stale `prefactor_c(w) = C(w)` description
removed (deferred findings INS-1-004 / INS-2-003 / INS-3-001).
