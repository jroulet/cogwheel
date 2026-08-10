# Build Brief: ppGO rung gate calibration — serve excised regions without quadrature

## Mission

Calibrate the ppGO fast-rung gates in `cusp_amplification` so the excised exterior regions (astroid cusp windows AND saddle deltoid-excised exterior) are served by the ppGO approximation (~10^3x faster) instead of falling through to the Pearcey table / live quadrature. The current gates (`_W_PPGO_FLOOR = 50`, `_R_PPGO_ERROR_CONST = 50.0`) are conservative and block ppGO in the excised regions' w-band. Keep the serve refusal-conservative.

## Background

- The `cusp_ppgo_high_w` build added a ppGO fast rung inside `cusp_amplification` (operator.py `_uniform_arm_value` path): when the Pearcey control radius `R >= r_ppgo_min` AND `w >= _W_PPGO_FLOOR`, serve via `fold_ppgo_correction` (the geometric image sum + Airy ghost for merging pairs), bypassing the certified Pearcey quadrature — ~10^3x faster.
- Current gates in `_pearcey_cusp.py`: `_R_PPGO_ERROR_CONST = 50.0`, `_W_PPGO_FLOOR = 50.0`.
- The excised regions (cusp windows within `_CUSP_ARM_COVERAGE = 0.07 rad` of a cusp vertex, and the ghost-transition zones) are the ones the exterior tiler excludes — they're served by the cusp arm / exact engine. Their w-bands are around [0.88, 19.3] (probe charts) — ALL below `_W_PPGO_FLOOR = 50`. So the ppGO rung never fires for the excised regions; they pay Pearcey table / live quadrature.
- The user's point: the whole purpose of excising is to avoid the surrogate AND the quadrature. The excised region should be served by ppGO without quadrature.

## Work

1. **Measure the ppGO error vs the exact Pearcey/engine** across w for representative excised-region configs:
   - Astroid cusp window: source within 0.07 rad of a cusp vertex, at a range of rho and w.
   - Saddle deltoid-excised exterior: near the deltoid cusps, both parities.
   - Sweep w from ~5 to ~100; compute `|ppGO - exact| / |exact|` where exact = `cusp_amplification` (Pearcey) or the exact engine `F_op`.
   - Determine the lowest w where ppGO is accurate below the certification bar (default 1e-3 envelope bar; use a safety margin, e.g. bar/10, per the ppGO build's `_PPGO_BAR_DIVISOR` philosophy).
2. **Recalibrate the gates**: lower `_W_PPGO_FLOOR` and/or `_R_PPGO_ERROR_CONST` so ppGO certifies across the excised exterior w-band. Verify the new gates on both parities. The gates must remain refusal-conservative — never serve a ppGO value that isn't accurate.
3. **Verify the excised regions now use ppGO**: census / serving-path check that excised cusp-window draws at mid-w are served by the ppGO rung (not live quadrature). Confirm the ppGO error vs the exact engine stays under the bar.
4. Keep the existing tests green (test_lensing_airy_fold.py, test_lensing_cusp_arm_coverage.py, test_lensing_schwinger.py).

## Measured facts (re-probe at HEAD before coding)
- `_W_PPGO_FLOOR = 50.0`, `_R_PPGO_ERROR_CONST = 50.0`, `_PPGO_BAR_DIVISOR = 10` in `_pearcey_cusp.py` (~427-437)
- The ppGO rung: `r_ppgo_min = (_R_PPGO_ERROR_CONST * _UNIFORM_ERROR_CONST / (envelope_bar / _PPGO_BAR_DIVISOR))**(2/3)`; fires when `radius >= r_ppgo_min AND w >= _W_PPGO_FLOOR` (~775-780)
- `fold_ppgo_correction` in `_airy_fold.py` (~475), `_uniform_arm_value` in `operator.py` (~402)
- Excised regions: cusp windows `_CUSP_ARM_COVERAGE = 0.07 rad` (surrogate.py); probe w-band [0.88, 19.3]
- Note: `_R_PPGO_ERROR_CONST = 50.0` is very large — the rung almost never fires even at high w. Measure whether the true error constant is much smaller (the brief from cusp_ppgo_high_w said it was "provisional, to be tightened by driver post-build measurement").
- Reference: the ppGO build's todo fragment `lensing_cusp_ppgo_at_high_w.md` and its completed.d.

## Constraints
- Fast tests. Follow AGENTS.md.
- The fix is a GATE CALIBRATION (measured), not a removal of the safety.
- Refusal-conservative: never serve a wrong ppGO value silently.
- Both parities (astroid cusp windows + saddle deltoid-excised exterior).
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The `_R_PPGO_ERROR_CONST = 50.0` was explicitly marked "provisional, to be tightened by driver post-build measurement" in the cusp_ppgo_high_w brief. This build does that measurement and calibration. The goal: the excised regions (both parities) are served by ppGO without quadrature across their w-band.
