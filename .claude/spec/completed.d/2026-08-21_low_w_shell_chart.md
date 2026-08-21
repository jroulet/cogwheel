---
date: 2026-08-21
section: Backlog
---

- **Low-w near-fold SHELL chart shipped (macro-lead demodulated-difference) —
  the quotient chart is DELETED and the Born rung's rho floor lowered to 1.4**
  `[→ spec]` — the end-state of the corrected rho-split ruling
  (`low_w_chart_rho_split_ruling.md`): the low-w band is served by the settled
  demodulated-DIFFERENCE representation, never a quotient. Two pieces:

  1. LOW-W NEAR-FOLD SHELL: `cogwheel/lensing/low_w_shell_chart.py` ships
     the frozen 4-D `LowWShellChart` dataclass (schema `low_w_shell_v1`,
     package artifact `cogwheel/data/low_w_shell_chart.npz`, content-hash
     load refusal, `covers()` box predicate) holding the SMOOTH macro-lead
     demodulated-DIFFERENCE residual `R(w; gamma', rho, theta) = f_pure(w) -
     born_lead_carrier(w)` over the near-fold shell grid (`gamma'` in
     `[0.05, 1 - DELTA_GAMMA_P]`, `rho` in `[RHO_LO, RHO_HI] = [0.6, 1.4]`,
     `theta` in `[0, pi/2]`, `log w` in `[log 0.02, log 1]`) — a DIFFERENCE,
     never a quotient: both sides carry the same carrier phase, so the
     beating zeros cancel identically and the residual has no poles (the
     retired quotient form produced 5800x poles). NO de-rate, NO
     `declined_mask`. `scripts/train_low_w_shell_chart.py` bakes it OFFLINE
     against `_schwinger.f_schwinger` (oracle only; `--scale smoke` in-build,
     `--scale full` driver bake). `likelihood.py` wires the chart-first
     Rung-P consult `_low_w_shell_chart_serve` with the band split at
     `w_shell = 1/delta_min` (chart serves the smooth sub-band
     `w * delta_min < 1`, exact engine hosts the resolved remainder, both in
     the `FARFIELD_DIFFRACTIVE` gauge), the `_AUTO_SHELL_CHART` sentinel
     auto-attach, and the 3-way `get_init_dict` round-trip. The census
     mirrors the shell gate (RHO_LO <= rho_dir <= RHO_HI AND
     w_hi*delta_min < 1 AND covers) as the `low_w_shell_chart` SERVE_ROUTE
     (renamed from `low_w_diffractive_chart`).

  2. BORN EXTENSION: the Born rung's rho floor is `rho > _BORN_RHO_FLOOR =
     1.4` (an independent SCALAR-reach gauge — ppgo_map.caustic_rho — with an
     honest gauge-aware comment: Born = scalar reach, shell = directional,
     disjoint sets with a coverage gap), and the shipped
     `born_residual_chart.npz` is re-trained over a 7 gamma x 8 rho x 13
     log-w grid (rho_grid `[1.4, 4.0]`, log-w `(0.4, 60)`), closing the
     shell/far-exterior handoff at RHO_HI.

  DELETED: `cogwheel/lensing/low_w_diffractive_chart.py`,
  `scripts/train_low_w_diffractive_chart.py`,
  `cogwheel/tests/test_lensing_low_w_diffractive_chart.py`, and the
  `_low_w_diffractive_chart_serve` wiring in `likelihood.py`. This supersedes
  the 2026-08-20 completion record's quotient chart (the earlier record stays
  as the historical artifact of that build).

  TESTS: `cogwheel/tests/test_lensing_low_w_shell_chart.py` (22 tests:
  no-poles finite-residual invariant with ratio cap 10, node-exact serve
  round-trip at 1e-10 + measured off-grid bar with doubled-carrier /
  unit-sqrt_mu / zero-residual falsification controls, rho=1.4 shell/Born
  handoff continuity, `_low_w_shell_chart_serve` end-to-end via the
  production binder, load-contract schema/hash hard refusals).

  ACCEPTANCE STATUS: in-build smoke grid certification only; the `--scale
  full` driver bake of `cogwheel/data/low_w_shell_chart.npz` and the
  re-trained `born_residual_chart.npz` (with the driver-prerequisite
  azimuthal sweep at rho=1.4) are DRIVER post-build steps per AGENTS.md.
