# Coder Short-Term Observations

- WP1 (delete _WEDGE_EPS / analytic _tube_normal / docstrings): deleted the
  `_WEDGE_EPS = 1e-3` constant + comment in surrogate_training.py and dropped
  the +/- offsets at all 6 wedge sweep sites so each spans
  [center-theta_max, center+theta_max] inclusive; kept the _saddle_arcs wall
  exclusion (edge_hw=_SADDLE_CUSP_MIN_HALFWIDTH) re-anchored at the true edge.
  Rewrote _tube_normal to derive tangent from
  geometry.caustic_derivatives(gamma, theta, branch=branch) (no finite diff);
  smoke-checked fd.tan == +1 (orientation preserved, no flip) and
  LensDomainError still propagates outside the wedge. Fixed 4 docstrings
  (geometry.caustic_derivatives x2: wedge edge is a REGULAR point, not a cusp,
  cite F044; _winding_number: IS applied to saddle lobes via _lobe_winding_loop
  at _SaddleLobeAdmission.admits; _lobe_winding_loop: loop closes exactly).
- OWED (Test Developer, per WP brief): cogwheel/tests/
  test_lensing_surrogate_training.py still imports/uses `_WEDGE_EPS` (L197,
  L1011-1012, L1488) — now a broken import after the constant deletion.
  Not fixed here (no test edits in WP1).
