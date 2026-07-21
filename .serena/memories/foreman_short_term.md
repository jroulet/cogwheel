- INS-1-003 (census_homogenization_corners.py node_classification_totals
  undercounting served_by_geometric for w<=60 positive-parity nodes that
  would be geometric in production): fixed via the doc-the-simplification
  option (not the "compute geometric independently of high" option) —
  added a `note` string key to the `node_classification_totals` dict in
  `run()` explaining the w<=60 schwinger/geometric split discrepancy vs
  operator.select_branch (which is w-independent for positive parity),
  and confirming the two headline fractions (gamma_prime_zero,
  unresolved_high_w_refusal_corner) are unaffected. Chose this over
  computing an independent geometric mask because the latter requires
  computing the expensive real-image delta_min for EVERY config (not
  just configs with a high node), which contradicts the function's
  documented efficiency contract ("computed at most ONCE per config,
  only when some node exceeds the ceiling") — that's a real behavior/
  perf change, out of trivial-fix scope for Foreman-Lite. Verified via
  ast.parse, import via sys.path insert (note: loading the script file
  directly via importlib.util.spec_from_file_location fails on its
  CensusConfig dataclass regardless of my edit — an unrelated dataclass/
  module-resolution artifact of loading outside normal package context;
  use sys.path.insert(0, 'scripts'); import census_homogenization_corners
  instead), and an actual `run()` smoke call confirming the note renders
  with W_CEILING interpolated and totals still populate correctly.