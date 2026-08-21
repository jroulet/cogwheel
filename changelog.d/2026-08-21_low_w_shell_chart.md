---
date: 2026-08-21
---

### Low-w near-fold shell: macro-lead demodulated-difference chart replaces the quotient chart

The positive-parity low-w diffractive serve's near-fold shell
(`RHO_LO <= rho <= RHO_HI` = `[0.6, 1.4]`) is now served by the trained
`LowWShellChart` (`cogwheel/data/low_w_shell_chart.npz`, schema
`low_w_shell_v1`), which stores the SMOOTH macro-lead
demodulated-DIFFERENCE residual `R = f_pure(w) - born_lead_carrier(w)` — a
difference, never a quotient, so it has no poles (the retired
`LowWDiffractiveChart` quotient form produced 5800x poles). The chart is
consulted first in the Rung-P branch; the served band is split at
`w_shell = 1/delta_min` with the exact engine hosting the resolved
sub-band above it, both in the `FARFIELD_DIFFRACTIVE` gauge. The Born rung's
rho floor is lowered from 2.0 to `_BORN_RHO_FLOOR = 1.4` and the shipped
`born_residual_chart.npz` re-trained over 7 gamma x 8 rho x 13 log-w nodes
(rho_grid [1.4, 4.0], log-w (0.4, 60)), closing the shell/far-exterior
handoff gap; the serve-route census route label is renamed
`low_w_diffractive_chart` -> `low_w_shell_chart`.
