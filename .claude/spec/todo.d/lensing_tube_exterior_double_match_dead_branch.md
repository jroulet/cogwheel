---
section: Backlog
---

- **Is the tube/exterior-polar double-match precedence branch dead
  production code under the beat-free contract?** `[housekeeping]` —
  measured 2026-08-17 (tree-gate diagnosis, beat-free recovery build):
  `_tube_f_ref` requires a 4-image interior source while
  `_exterior_polar_serves` requires rho >= its pos_ff `rho_grid[0]`
  (= 1.0099 in the shipped map), so no PHYSICAL source can be served by
  both charts and the tube-wins double-match precedence in the chart
  selector is unreachable in production. The selection-precedence pins
  (`ChartSelectionTestCase` overlap-band tests) were re-pointed to
  structural probes via `_tube_serves(..., require_fref=False)` to keep
  the precedence invariant tested. DECIDE: (a) prove the disjointness
  holds for every trained map (is `rho_grid[0] > 1` a contract or an
  accident of this training?), then delete the dead branch and retire
  the structural probes with it; or (b) if some map/parity can
  legitimately overlap, document the overlap domain and keep the
  precedence. Engineering-values call: one authoritative answer, no
  speculative code kept "just in case". Sequence: after the training
  campaign (the retrained maps answer whether rho_grid[0] > 1 is
  universal).
