# Build Brief: Zero-quadrature Pearcey hot path

## Mission

Eliminate ALL live certified quadrature from the evaluation hot path. The only such site is the Pearcey arm's fallback in `_consult_pearcey` (table-serve inside its box, live certified quadrature outside). The fix: expand the Pearcey table (retrain) to cover the full serving region up to (and overlapping) the ppGO crossover, so the hot path is table + ppGO + spline only — never live quadrature. Configs outside both the table and ppGO refuse to the exact engine (rare) rather than quadrature.

## Background (measured 2026-08-10)

- `_consult_pearcey` (channels.py/_pearcey_cusp.py ~152): "Table-first Pearcey lookup with live certified-quadrature fallback. table is None (the default) evaluates the certified quadrature... the live quadrature is used outside the box."
- Current table (`cogwheel/data/pearcey_table.npz`): x in [-27.6, 27.6], y in [-90.8, 90.8], 161x161 grid (demodulated Re/Im bicubic splines).
- ppGO crossover: r_ppgo_min = 71.1 (with calibrated _R_PPGO_ERROR_CONST=3.0, _PPGO_BAR_DIVISOR=10, envelope_bar 0.05). The table's y-extent (90.8) already exceeds this in the y-direction.
- The serving ladder: surrogate charts (spline) -> geometric -> uniform arms (fold Airy + cusp Pearcey) -> Schwinger exact -> named refusal. The Pearcey arm is the cusp portion; ppGO serves its high-R part.

## Work

1. **Map the full Pearcey serving region**: determine the reachable (x, y) Pearcey-control range across all served sources at all w (the y = delta_perp * w^0.75 / |C4|^0.25 control grows fast with w; x = delta_parallel * w^0.5 / sqrt(|C4|)). Confirm the current table box (x ±27.6, y ±90.8) covers it, or identify the gaps (corners, extremes) that need expansion.
2. **Expand the table if needed**: retrain `scripts/train_pearcey_table.py` over the expanded (x, y) domain so it covers the full region up to the ppGO crossover, with a safety overlap (ppGO already exact there). Verify the expanded table's interpolation accuracy (the demodulated Re/Im bicubic splines) over the new domain — especially at large R where the Pearcey asymptotes to the geometric sum (should be smooth).
3. **Remove the live-quadrature fallback**: in `_consult_pearcey`, replace the live-certified-quadrature fallback with a refusal (return None) for configs outside the table box. The serving ladder then falls through to ppGO (if R large enough) or the exact engine (rare). Verify the hot path never calls live quadrature.
4. **Verify zero quadrature**: a serving-path/census test that no hot-path draw is served by live certified quadrature. Confirm the table + ppGO cover everything the Pearcey arm used to serve, and the fall-through behavior (refusal -> ppGO -> exact engine) is correct and refusal-conservative.
5. Keep the existing Pearcey/ppGO tests green (test_lensing_airy_fold.py, test_lensing_cusp_arm_coverage.py, test_lensing_schwinger.py). Update any test that asserts live-quadrature fallback behavior.

## Measured facts (re-probe at HEAD before coding)
- Table box: x [-27.6, 27.6], y [-90.8, 90.8], 161x161
- ppGO crossover r_ppgo_min = 71.1 (R const 3.0, bar_ppgo 0.005)
- Live-quadrature site: `_consult_pearcey` fallback (the ONLY hot-path quadrature)
- Pearcey controls: x = delta_parallel * w^0.5 / sqrt(|C4|), y = delta_perp * w^0.75 / |C4|^0.25
- Table training: `scripts/train_pearcey_table.py`; artifact `cogwheel/data/pearcey_table.npz` (content-hash-verified load; live-quadrature fallback on ANY anomaly, per SPEC)
- Serving ladder: `_uniform_arm_value` (operator.py ~402) tries fold Airy then cusp Pearcey

## Constraints
- Fast tests. Follow AGENTS.md.
- NO live quadrature in the hot path — this is the mission. Refuse to exact engine rather than quadrature.
- Refusal-conservative: never serve a wrong value silently. The table-expansion must be content-hash-verified and accurate over its new domain.
- The table serves the cusp arm; ppGO serves its high-R part; both must cover the full former-quadrature region.
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The user's requirement is explicit: "no quadrature in the entire evaluation hotpath... Can't guarantee speedup unless it's everywhere." The preferred approach (user-suggested) is to EXPAND THE TABLE rather than build a residual surrogate — simpler, more robust. The current table's y-extent already exceeds the ppGO crossover, so the gap may be small or absent; verify first, expand only where needed.
