# Coder Short-Term Observations

- WP1 tube-chart arc-length: added frozen field `theta_to_s` (2,N_map) to
  TubeChart; from_values/_assemble take optional s_grid+theta_to_s (identity
  default s=theta-theta_lo). Serve maps query theta -> s via np.interp before
  contracting knots[3] (now built in s). NPZ stores/reads `chart{i}_theta_to_s`.
  _build_tube_chart places theta nodes as images of uniform s grid at
  rep_gamma=median via new _tube_arc_length_map (scipy cumulative_trapezoid of
  geometry.caustic_speed, branch-aware, 2001 pts, finite+strict-increasing raise).
- Identity default is NOT bit-identical to HEAD raw-theta spline: matches to
  ~5e-15 (B-spline translation invariance; knot arithmetic differs in last bits).
  Fine for tolerance-based suites; a bit-exact-vs-stored-HEAD assertion would drift.
- Old (pre-schema) tube npz records hard-refuse on load (KeyError on
  chart{i}_theta_to_s) — correct: an old chart's knots are in raw theta, so an
  identity-map fallback would serve at the wrong offset. Do NOT add a fallback;
  old artifacts must be retrained.
- Diagnostic s_map_gamma_endpoint_dev recorded in the build_tube CLOSURE report
  dict (the actual build report), NOT in _build_tube_chart's return, to avoid
  changing its 3-tuple contract (test unpacks it at test_..._training.py:1043).
