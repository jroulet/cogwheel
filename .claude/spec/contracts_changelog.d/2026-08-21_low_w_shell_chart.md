---
date: 2026-08-21
bump: minor
---

### `low_w_diffractive_chart` retired; `low_w_shell_chart` registered

The `low_w_diffractive_chart` artifact entry is REMOVED (module
`cogwheel/lensing/low_w_diffractive_chart.py`, producer
`scripts/train_low_w_diffractive_chart.py`, and the `_low_w_diffractive_chart_serve`
consumer are all deleted) and replaced by the `low_w_shell_chart` entry:
the trained near-fold-shell macro-lead demodulated-DIFFERENCE chart shipped
at `cogwheel/data/low_w_shell_chart.npz` (schema `low_w_shell_v1`), produced
by `scripts/train_low_w_shell_chart.py`, consumed by
`LensedRelativeBinningLikelihood._low_w_shell_chart_serve` and
`serve_route_census.classify_draw`. Fields: `gamma_prime_grid`, `rho_grid`,
`theta_grid`, `log_w_grid` (1-D ascending axes), `real_coeffs`/`imag_coeffs`
(4-D, shape n_gamma' x n_rho x n_theta x n_w), `provenance`, `content_hash`
(SHA1 over the 6 hashed fields), `schema`. NO `derate`, NO `declined_mask`.
The `born_residual_chart` entry is updated to the re-trained 7 gamma x 8 rho
x 13 log-w grid (rho_grid [1.4, 4.0], log_w_grid (0.4, 60)) and the lowered
gate `caustic_rho(...) > _BORN_RHO_FLOOR = 1.4`.
