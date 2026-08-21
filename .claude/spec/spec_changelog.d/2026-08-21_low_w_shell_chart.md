---
date: 2026-08-21
bump: minor
---

SPEC.md's LOW-W DIFFRACTIVE RUNGS paragraph (Rung P) rewritten for the
low-w shell chart build: the retired quotient `LowWDiffractiveChart` (union
shell-OR-wall band, `fold_cusp_reference`, `_NON_VANISHING_MIN_RATIO`,
scalar de-rate + per-cell `declined_mask`, schema `low_w_diffractive_v1/v2`,
`w**(2/3)` axis) is replaced by `LowWShellChart` — the near-fold shell
(`RHO_LO <= rho <= RHO_HI` = `[0.6, 1.4]`) stored as the SMOOTH macro-lead
demodulated-DIFFERENCE residual `R = f_pure(w) - born_lead_carrier(w)` (a
difference, never a quotient — both sides carry the same carrier phase, so
the beating zeros cancel; schema `low_w_shell_v1`, `log w` axis over
`[log 0.02, log 1]`, NO de-rate / NO `declined_mask`), consulted FIRST in
the Rung-P branch with the served band split at `w_shell = 1/delta_min`
(chart serves the smooth sub-band `w * delta_min < 1`, exact engine hosts
the resolved remainder, both in the `FARFIELD_DIFFRACTIVE` gauge). The
retired `_WALL_GAMMA_PRIME` wall-band admission is gone — the `gamma'`
grid spans the full positive-parity range to `1 - DELTA_GAMMA_P`. FIRST-CLASS
BORN INTERCEPT and BORN EXTERIOR RUNG paragraphs: the `rho > 2` Born floor
is `rho > _BORN_RHO_FLOOR = 1.4` (a scalar-reach gauge independent of the
shell's directional `RHO_HI`; the two 1.4 surfaces are different physical
surfaces with a coverage gap), and the "BornResidualChart.load not yet
implemented" parenthetical is removed (the classmethod long shipped).
ENGINE-FREE SERVE-ROUTE DEMAND CENSUS paragraph: `SERVE_ROUTES` member
`low_w_diffractive_chart` renamed `low_w_shell_chart` (census label
mirrors the module migration). Deferred findings INS-1-007 / INS-2-003.
