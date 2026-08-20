---
date: 2026-08-20
section: Backlog
---

- **Low-w near-fold / wall-band chart serve SHIPPED — the near-fold shell and
  the wall band are served analytically by the trained 4-D residual chart
  `LowWDiffractiveChart`** `[→ spec]` — completes the open todo item
  `todo.d/lensing_low_w_near_fold_serve` (removed by this completion). The
  positive-parity Rung P band bottom in the near-fold shell
  (`rho = |y'|/|y_c(theta)|` in `[RHO_LO, 1 + DELTA]`) and the wall band
  (`gamma' > _WALL_GAMMA_PRIME = 0.5`, the order-16 series' convergence-radius
  collapse) is now served O(1) by the trained residual chart instead of
  falling through to the exact Schwinger engine.

  WHAT SHIPPED (code): `cogwheel/lensing/low_w_diffractive_chart.py` — the
  frozen 4-D `LowWDiffractiveChart` dataclass (reduced/caustic-relative
  coordinate grid `(gamma', rho, theta, log w)`, D2 theta-fold, union
  `covers` predicate, scalar de-rate + per-cell `declined_mask`, schema +
  content-hash load refusal); `scripts/train_low_w_diffractive_chart.py` —
  the offline Schwinger-oracle trainer (Schwinger is an OFFLINE oracle only,
  never called at serve time) baking the npz with the measured wall-band rho
  spread and certified margins; the likelihood serve
  `_low_w_diffractive_chart_serve` consulted FIRST in the Rung-P branch
  (sentinel auto-attach, refuse-to-None on load anomaly, 3-way
  `get_init_dict` round-trip, per-cell decline fall-through to the exact
  engine — never an amplitude scale); and the census mirror — the 12th
  draw-level `SERVE_ROUTE` `low_w_diffractive_chart` with a chart-first
  consult on the DIRECTIONAL `_caustic_rho` gauge (INS-2-001 fixed a
  rho-rebind bug there: the scalar `caustic_rho` field gauge is preserved for
  the demand buckets). De-rate `derate = min(1.0, 1/max_overshoot)` is the
  sole conservativeness margin (INS-1-001 removed the 0.85 clamp); the 1e-4
  certification bar is enforced on the SERVED two-sided error; `declined_mask`
  is the 8th field in the content hash (INS-2-002) so a stale/tampered
  all-False mask is refused on load.

  WALL-BAND RULING REVERSED: the interim owner ruling had resolved the wall
  band separately by routing to Schwinger at serve (order-16 series
  convergence collapse); the fence build's gamma-domain fence (f33d85e) was
  then REVERTED (050d4cf) — the wall band needs a chart serve, not
  decline-to-engine — so the chart covers the UNION of the near-fold shell
  and the wall band, and the original fragment's "wall band is SEPARATE and
  resolved via Schwinger" sentence is superseded by this build.

  DATA CONTRACT (INS-1-003): `DATA_CONTRACTS.yaml` gains the
  `low_w_diffractive_chart` artifact entry (producer
  `scripts/train_low_w_diffractive_chart.py::main`; consumers
  `LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve` and
  `serve_route_census.classify_draw`; npz fields: 4 grid axes, real/imag
  coeff arrays, scalar `derate`, per-cell `declined_mask`, JSON `provenance`,
  `content_hash`, `schema = low_w_diffractive_v1`).

  TESTS: 36 test methods across 13 classes in
  `cogwheel/tests/test_lensing_low_w_diffractive_chart.py` (all green,
  including the self-falsification classes proving the suite can go red) —
  DC anchor re-modulation (sqrt(mu_macro) not 1; prefactor_c / mass-sheet
  phase applied once), node-exact serve-vs-engine accuracy (1e-10),
  one-sided de-rate conservativeness + overshoot-without-derate
  falsification, load-contract (schema/content-hash round-trip +
  hard-refusal), coverage-union predicate, theta D2 fold, and the census
  mirror (classify_draw routes the near-fold shell witness to
  `low_w_diffractive_chart`).

  ACCEPTANCE STATUS: in-build certification covers the node-exact serve
  accuracy and the one-sided de-rate conservativeness on the smoke grid, and
  the census no longer counts covered near-fold/wall draws as engine demand
  (they classify to the new route). The full-scale 10k-census recount and the
  `--scale full` driver bake are DRIVER POST-BUILD verification (in-build
  smoke only, per AGENTS.md). The fragment's ORDERING CONSTRAINT (land before
  demand-census-driven tiling-plan refresh / campaign sizing) is now
  satisfied: `engine_residual` demand can be re-measured on the honest
  post-chart map.
