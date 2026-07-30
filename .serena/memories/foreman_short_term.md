# Foreman-Lite Short-Term Observations

- INS-2-001 (this session): fixed the stale ANGLES oracle in
  TruncationOnRefusalTestCase (test_lensing_ppgo_bandsplit.py) that still
  hardcoded the retired one-sided 5-angle sweep (0..pi/2). Replaced with
  `tuple(k*math.pi/8 for k in range(-4, 5))` matching the production
  `_measure_cell` symmetric 9-angle fan, and updated the docstring comment.
  Verified: _w_star(angle) uses raw (non-abs) angle, so the 4 added
  negative angles evaluate to w_star values (102.5-140) that exceed WALL
  (100), meaning they never restrict `w_nodes[w_nodes<=w_star][-1]` below
  the max node — confirms the finding's claim that the min-over-angles
  ceiling is still dominated by the +pi/2 angle (tightest, 40) and the
  test remains green (5 passed) after the fix, same as before. Did not
  touch `_w_star` itself — the finding scoped the fix to ANGLES + comment
  only, not to making the stub law mirror-symmetric like the real
  production geometry comment describes; that's a separate design choice
  outside this finding's scope.
