## 2026-08-04

### Born residual chart trained and shipped (far exterior rho > 2)

The Born exterior rung is now a **complete, zero-quadrature serve path**
for exterior-to-caustic draws (`rho > 2`) on both parities.  The
fact-4 slot in `likelihood._surrogate_coefficients` was already wired; it
now has a trained artifact to attach.

**Shipped artifact:** `cogwheel/data/born_residual_chart.npz` (≈ 8 KB).
The chart is a 3-D tensor-product cubic spline of the Born residual

    R(w; gamma, rho) = F_exact_demod(w) − F_carrier_demod(w)

over a 7 gamma × 5 rho × 10 w sparse grid, all in the min-relative delay
frame.  Training runtime ≈ 11 s; artifact is package-data and
content-hash-verified at load.

**Training driver:** `scripts/train_born_residual.py` (141 lines).
Produces the npz from the exact engine directly; no intermediate steps.

When `LensedRelativeBinningLikelihood` is constructed with
`born_residual_chart=BornResidualChart.load(...)`, exterior draws that
were previously handed to the exact engine are now served
`F_carrier + R` from the chart.  The default (`born_residual_chart=None`)
is unchanged — exterior draws continue to fall through to the exact engine.

Commit: `849e580`
