---
bump: minor
---

The interior fold-ppGO rung now serves RAW ppGO
(`operator.geometric_amplification`) instead of `fold_ppgo_correction`.

Measured against the exact engine over `w in [30, 60]` — deliberately the
oracle-valid band, since `F_op` returns the uniform arm above 60 and is not an
independent oracle there (F069) — fold-corrected max relative error is
1.22e-1 against raw ppGO's 1.49e-4: an **818x** improvement on the rung's own
domain. Raw ppGO additionally improves with frequency (~`w^-2.75`) while the
fold residual is w-independent, so the margin widens toward the `w ~ 5e4`
where this gate first opens.

The correction was a net loss here because the gate's closed form is
`w*dtau >= 13344*c_A`: it SELECTS well-separated pairs far from the caustic,
which is exactly where the fold normal form is invalid. The fold correction
beats raw ppGO only for `rho >= 0.93` (`xi <~ 0.6`), while the gate demands
`xi >= 4`.

Unchanged: the gate itself, so `surrogate_census` accounting and the
`ppgo_fold` served-cause label are unaffected (the census evaluates only the
gate and never called `fold_ppgo_correction`). The label now means "ppGO on
the fold-pair gate" rather than "fold-corrected".

Still open: the gate remains mis-shaped — it opens away from the region where
a fold correction would help. Re-deriving it for small `xi` / small `eta`,
where the correction is valid AND lives inside the engine's checkable domain,
is tracked in `todo.d/lensing_fold_ppgo_rung_serves_wrong`.
