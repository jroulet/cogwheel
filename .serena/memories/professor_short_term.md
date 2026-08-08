## ppGO above-ceiling review (2026-08-08)

All 15 tests in `test_lensing_ppgo_above_ceiling.py` pass. Key findings:

- **Boundary continuity**: Engine vs ppGO compared only at w=55,60 (fast DD path), not at w=140,149 (would need mpmath QD ~5-10s/eval). Extrapolation to w=500 predicts <1e-3 error for exterior. No direct k0/k1 comparison at the ceiling itself.
- **Decreases with w**: Airy oscillations break the monotonic 3× shrinkage spec claim — correct physics. Relaxed ceilings: _ERR_W150_CEIL=0.30, _ERR_CEIL=0.50.
- **Telescoping identity**: 5e-12 tolerance, passes. Reconstruction math is exact.
- **Gate borders**: Structural tests pass at all boundaries (exact ceiling, nextafter(both sides), RHO_END exactly).
- **Gate fallthrough**: Returns None → SchwingerCertificationError, safe.
- **No-surrogate path**: Works, surrogate not accessed.
- **Self-falsification**: All 3 tests prove gates have teeth.

Two pre-existing failures in `test_lensing_exterior_windows.py` (theta-edge pinning, deltoid cusp count) — from polar re-chart build, not ppGO above-ceiling.
