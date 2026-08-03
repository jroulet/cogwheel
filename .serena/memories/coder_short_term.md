# Coder Short-Term Observations

- wp1 (min_gamma_band 0.02→0.005): 3 value edits + 2 comment fixes across
  surrogate_training.py and scripts/measure_dropped_slivers.py. F041 test
  constant intentionally left at 0.02.

- wp1 (ppgo_interior_handoff): Added fold-ppGO interior handoff in
  _surrogate_coefficients (likelihood.py, after Born chart block, inside
  rho<=1.0 branch) and mirrored gate logic in characterize_sample
  (surrogate_census.py). Key discovery: geom.images is a LIST not ndarray
  (despite type annotation), so used list(geom.images) not shape indexing.
  matrix reconstructed via macro_matrix (not on geom). Census uses
  _XI_FOLD_THRESHOLD=4.0 locally (no circular import from likelihood).
  Category 'ppgo_fold' with served=True passes fallthrough_breakdown
  validation (served records skip category check).
