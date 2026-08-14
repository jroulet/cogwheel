# Foreman-Lite Short-Term Observations

- INS-1-001 (2026-08-14, _pearcey_cusp.py): dangling comment reference to a
  git-rm'd script (scripts/calibrate_ppgo_rung.py) is fixed by rewording
  the comment to state the measurement result inline ("the retired
  calibration sweep observed...") rather than citing a live path — simple
  pattern for any future "deleted script cited in shipping comment" finding.
- INS-2-002 (2026-08-14, surrogate.py `_evaluate_chart` docstring): a stale
  "ignored for a tube chart" claim about `y1_eig`/`y2_eig` is fixed by
  documenting that the tube branch also folds `theta` via
  `_fold_caustic_theta(theta, y1_eig, y2_eig)` before frame/arc-length
  mapping — the same fold as `_tube_serves`. Pure docstring reword, no
  logic touched; verified with `ast.parse` only (no test impact expected
  for a docstring-only change per proportionate-verification memory).
