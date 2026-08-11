---
section: Backlog
depends_on: [2026-08-10_saddle_exterior_full_treatment]
---

- **Zero-quadrature Pearcey hot path (expand table, remove live quadrature)**
  `[→ spec]` — identified 2026-08-10.

  User requirement: NO live quadrature anywhere in the evaluation hot path.
  The surrogate's purpose is speed; a single live-quadrature fallback
  anywhere defeats it. The only live-quadrature site is the Pearcey arm's
  fallback (`_consult_pearcey`): table-serve inside its box, live certified
  quadrature outside.

  The current Pearcey table covers x in [-27.6, 27.6], y in [-90.8, 90.8]
  (161x161). The ppGO crossover is r_ppgo_min = 71.1 (R const 3.0, bar_ppgo
  0.005). So the table's y-extent (90.8) already exceeds the ppGO crossover
  in the y-direction. The fix is to EXPAND the table (retrain via
  `scripts/train_pearcey_table.py`) so it covers the full serving region up
  to (and overlapping) the ppGO crossover — eliminating the live-quadrature
  fallback entirely. Simpler than fitting a table-ppGO residual surrogate.

  **Fix**: (1) map the full Pearcey serving region (the (x, y) controls
  reachable by any served source at any w) and verify the table box covers
  it, expanding as needed (train_pearcey_table.py domain); (2) remove the
  live-certified-quadrature fallback in `_consult_pearcey` so the table (or
  ppGO) serves everywhere — refuse (-> exact engine) rather than quadrature
  if a config is outside the table; (3) verify no hot-path draw falls to
  live quadrature (census / serving-path test).

  ACCEPTANCE: no live certified quadrature in the Pearcey arm; the table
  (expanded if needed) covers every Pearcey-served config; ppGO serves
  where the table doesn't; any config outside both refuses to the exact
  engine (rare) — never quadrature. The hot path is table + ppGO + spline
  only.
