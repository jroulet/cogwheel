# ppGO rung build review (2026-08-09)

## Verdict: CONCERN — structural guards PASS, numerical calibration owed

All 13 ppGO tests pass. Gate logic is correct:
- `r_ppgo_min = (_R_PPGO_ERROR_CONST * _UNIFORM_ERROR_CONST / bar_ppgo)^(2/3)` with exponent 2/3 correctly inverts the R^(-3/2) cusp-proximity error scaling
- `_R_PPGO_ERROR_CONST = 50.0` (provisional), `_PPGO_BAR_DIVISOR = 10` → bar_ppgo = 0.005, r_ppgo_min ≈ 464.2
- `_W_PPGO_FLOOR = 50.0` independently gates kernel-truncation error O(1/w³)
- Finiteness guard (`np.isfinite(abs(result))`) catches all 4 NaN/Inf variants
- Do-nothing control: byte-identical result at intermediate R with/without ppGO rung
- Self-falsification: both gate-corruption directions verified

Concerns:
1. **Numerical accuracy NOT tested**: The specification's |ppGO - pearcey|/|pearcey| < 0.005 agreement is NOT asserted in tests. Docstring: "Numerical agreement with the Pearcey path is NOT asserted here because the current ppGO rung delegates to fold_ppgo_correction (a fold-corrected form) rather than a cusp-corrected form." Post-build driver measurement owed.
2. **`_R_PPGO_ERROR_CONST = 50.0` is provisional placeholder** — the rung serves in a narrower regime than it could. Needs post-build calibration sweep.
3. **Fold-corrected vs cusp-corrected**: `fold_ppgo_correction` uses Airy (fold catastrophe), not Pearcey (cusp catastrophe). At the cross-over (large R) Airy→geometric sum matching Pearcey→geometric sum, so the approximation is asymptotically sound. Conservative R-gate ensures we're deep enough.
4. **Pre-existing `test_moving_error_const_threshold_flips_a_fixed_node` timeout** — NOT ppGO-induced. The ppGO gate fails for this config's radius on both branches (r_ppgo_min ∝ _UNIFORM_ERROR_CONST, both scale together, r_ppgo_min ≈ 25× radius at the low-const setting). Pre-existing performance issue in the broader suite.

Post-build calibration needed: sweep R at fixed w, measure |ppGO - pearcey|/|pearcey| at worst-case direction, fit exponent, tighten _R_PPGO_ERROR_CONST.
