# Tidy Short-Term Observations

## cusp ppGO fast rung build (c0089e0-ish), 2026-08-09 — structural pass

- `cogwheel/tests/test_lensing_airy_fold.py`: removed unused `main` from the
  `unittest` import (never referenced; no `__main__` block, file ends at the
  last test class). Only edit of the pass.
- `cogwheel/lensing/chang_refsdal/_pearcey_cusp.py`: clean. Layering
  (stdlib/third-party/cogwheel) correct; all imports used; section-based
  private-helpers-beside-public-API organization is coherent, not a
  public/private inversion.
- Observation for the driver (NOT touched per "no blank-line reflow"): in
  `_pearcey_cusp.py` the constant `_PPGO_BAR_DIVISOR = 10` (line ~437) is
  followed by `def _real_stationary_points` with only ONE blank line — the
  only place in the module with a single blank between top-level defs. The
  mechanical script does not insert a missing blank line (it only collapses
  3+ runs), so this stays invisible to `tidy_mechanical.py --check`.