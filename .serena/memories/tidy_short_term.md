# Tidy observations (2026-08-10) — build 572eaa4 (2D (rho, u) fold-carrier)

Scoped style review of the build's three changed .py files.

## Changed
- cogwheel/tests/test_lensing_surrogate_training.py: removed genuinely unused
  import `from cogwheel.lensing.waveform import dimensionless_frequency`
  (was line 185). Verified by AST scan + reading: referenced only in the
  module docstring (line 65), a comment (213) and a test docstring (1765);
  the F002 oracle deliberately uses the independent `1.2372e-4` closed form
  "never the production `dimensionless_frequency`". No Load usage anywhere.

## Clean (no change) — surrogate.py
- Imports all used and correctly layered (stdlib → numpy/scipy → cogwheel),
  TYPE_CHECKING block commented and separated. `_SaddleLobeAdmission` is
  legitimately used in string annotations (from_lobe_engine / LobeInteriorChart)
  under `from __future__ import annotations` — do NOT remove.
- Organization: bottom-up (private coordinate helpers → chart dataclasses →
  guard stack → LensAmplificationSurrogate → private npz/validation helpers)
  is pre-existing and coherent with the module docstring; the build added
  `_probe_ghost_delay` + `_compute_rho_u_carrier` inside the existing private
  region. `select_chart` (public) sits among private `_*_serves`/`_evaluate_chart`
  — dependency-driven, not moved; reordering 4.4k lines would be unsafe.
- `make_interp_spline`/`BSpline`/`minimize_scalar`/`hashlib`/`files` all used
  (verified by reading, not just AST).

## Clean (no change) — test files
- test_lensing_exterior_polar_fold.py: helpers-first + private base class then
  public TestCases is the repo's standard test-suite layout; `matplotlib.use
  ('Agg')` before `pyplot` import is the canonical headless idiom. All imports
  used.
- test_lensing_surrogate_training.py: after the one removal above, no dangling
  imports; stdlib → numpy → cogwheel layering correct.
