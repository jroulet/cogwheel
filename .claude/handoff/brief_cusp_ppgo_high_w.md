# Build Brief: High-w cusp serving via ppGO instead of Pearcey live quadrature

## Mission

At high `w`, the Pearcey function asymptotes to the geometric image sum — `fold_ppgo_correction` (which includes the Airy ghost for merging fold pairs) plus non-merging image kernels serves the cusp region accurately and ~10³× faster than live certified Pearcey quadrature. Add a high-w ppGO rung in the cusp arm that certifies and serves ppGO above a measured cross-over `w`.

## Work

1. **Measure the cross-over `w`**: at a cusp-probe point (e.g. `gamma=0.4`, `y` at a cusp ray), compute the ppGO error against the exact Pearcey quadrature for a sweep of `w`. Record the lowest `w` where `|ppGO - Pearcey| / |Pearcey| < envelope_bar` (default 1e-3).

2. **Add ppGO rung in `cusp_amplification`**: before the Pearcey table lookup, when `w >= CROSS_OVER_W_CUSP_PPGO`, build the ppGO approximation — `fold_ppgo_correction` for merging pairs + individual image kernels — and serve it. The computation mirrors the existing ppGO serving path in the likelihood's `_amplification_coefficients`.

3. **Certify accuracy**: wrap the ppGO serve in a runtime certification check against geometric cross-checks (e.g. verify that the Pearcey controls `(x, y)` are large enough that the asymptotic expansion's error bound is below the bar).

4. **Retire live quadrature for high w**: when the ppGO rung certifies, never fall through to live quadrature. The live path remains for `w < CROSS_OVER_W_CUSP_PPGO`.

## Measured facts (SHA 9597a4e)
- `cusp_amplification` at `_pearcey_cusp.py:638` — table first, live quad fallback
- `fold_ppgo_correction` at `_airy_fold.py` — existing, includes Airy ghost
- `_uniform_arm_value` at `operator.py:405` — fold Airy then cusp Pearcey
- Pearcey controls: `x = c_x * w^(1/2) * d_parallel`, `y = c_y * w^(3/4) * d_perp`
- Asymptotics: as `(x,y) → ∞`, Pearcey → sum of geometric-image contributions, error ~ `R^(-3/2)` where `R² = x² + y²`

## Constraints
- Fast tests. Follow AGENTS.md.
- The ppGO rung must certify — never serve a wrong value silently
- Cross-over `w` must be measured, not guessed
- Live quadrature remains for `w < CROSS_OVER_W_CUSP_PPGO`
