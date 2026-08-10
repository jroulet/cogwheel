---
date: 2026-08-10
section: lensing-surrogate
---
## ppGO rung gate calibration

Completed in build `ppgo_rung_gate_calibration` (commit d5da155), with manual
Coder completion after an agent runtime error.

- `_R_PPGO_ERROR_CONST` lowered from 50.0 to **3.0** (measured).
- `_W_PPGO_FLOOR` lowered from 50.0 to **8.0** (measured).
- Calibration via `scripts/calibrate_ppgo_rung.py`: swept over cert-passing
  cusp-window directions at w ∈ [3, 50]; ppGO error scales as R^{-3/2} with
  binding w-threshold 50.0 extrapolated to err < 0.5% yielding safety factor
  ~3. R-gate set at 71 (2.4x safety). w-floor dropped to 8 (1.6x safety, sub-
  percent agreement for w >= 5 in the serving region).
- 13 ppGO tests pass.

FINDING: ppGO does NOT reach the 0.5% bar in the immediate excised cusp-window
region (R too small there); it certifies only deeper in the exterior (R >= 71).
The excised cusp-window draws at their typical w/R still fall to the Pearcey
table / live quadrature. The R-gate is the binding constraint, not the w-floor.
The original goal (serving excised regions without quadrature) is not fully
achieved — ppGO is refusal-conservative in the excised window.

SPEC NOTE: `[→ spec]` tag — SPEC.md already describes the ppGO gate mechanism
correctly without provisional language; no spec-version bump needed.

Driver post-build action owed: if zero-quadrature serve of the excised cusp-
window is still a priority, a separate investigation is needed (e.g. a
different asymptotic or a pre-computed table path for that region).
