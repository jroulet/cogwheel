Session 2026-08-21 (inference review): low_w_shell_chart serve build — verdict PASS.

Verified by running fast tests (test_lensing_low_w_shell_chart.py 22/22 in 22.8s; test_lensing_born_certificate.py + test_lensing_born_analytic_reachability.py 75/75 in 23.4s) + independent numeric re-derivation.

Measured numbers (independent, not via chart.evaluate):
- No-poles: max|R|/max|F| = 0.216 (gp=.8 rho=1.1 th=.2), 0.273 (gp=.5), 0.884 (gp=.3 rho=.9 th=1.0) — all O(1), far under the 10 cap; |F_serve| stays in [0.5,5]*sqrt(mu_macro), no collapse/explosion. The difference form R = f_schwinger - born_lead_carrier is finite everywhere (no quotient pole at carrier-beating zeros).
- Node-exact serve vs f_schwinger oracle: worst rel err = 0.0 (bit-exact round-trip). Off-grid worst = 0.019 (~2e-2) < 0.1 measured bar.
- Boundary: RHO_HI == _BORN_RHO_FLOOR == 1.4 (no gap/overlap); shell serve node-exact at rho=1.4; Born carrier-only certificate REFUSES at rho=1.4 (safety*est >> bar) -> falls through to exact engine, so no finite-but-wrong serve (the honest no-step form).
- Null-residual: BornNullResidualReconstructionTestCase R=0 -> bare born_carrier_from_partition to 1e-13 (plus diffractive-bottom and ppGO-above tiers), independent oracles.

Falsification teeth: doubled-carrier / unit-sqrt_mu / zero-residual each break node-exactness >1e-3; 100x residual trips the ratio cap.

Concerns (non-blocking): (1) off-grid 1e-4 acceptance is the TRAINED chart's bar, operator-deferred full-bake; the 480-node synthetic fixture pins 0.1 (cubic theta overshoot ~2e-2). (2) shipped trained artifacts low_w_shell_chart.npz / born_residual_chart.npz absent -> auto-attach None + RuntimeWarning, low-w shell rung falls through to exact engine. (3) DATA_CONTRACTS.yaml:387 + spec fragments still reference deleted low_w_diffractive_chart (doc hygiene, librarian follow-up). _pearcey_cusp.py change is docstring/whitespace only. 221-collected adjacent suite (surrogate_census/diffractive/airy_fold) is slow-tier, left to operator.
