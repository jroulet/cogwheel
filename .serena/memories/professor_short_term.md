# Diffractive w_low_fit re-bake consult (2026-08-19)

Domain ruling on the Chang–Refsdal truncation-certificate fit fix
(`cogwheel/lensing/chang_refsdal/_diffractive.py::w_low_fit`).

- **Angular symmetry is exactly 4-fold (π/2-periodic), NOT merely cos(4kθ) as a
  convenience.** Physics: D_0 = ∂_u²-∂_v² is a spin-2 quadrupole; D_0^n applied to
  the radial point-mass kernel G_PM(|y|²) spans cos(2mθ), m≤n. |F| satisfies
  |F(θ+π/2; γ')| = |F(θ; -γ')| and |F| is even in γ', so |F| (hence w_low_true) is
  π/2-periodic → only cos(4kθ) harmonics survive. Leading angular correction to |F| is
  cos(4θ) (the cos(2θ) from t_1 enters |F|² only through |t_1|² ∝ cos²2θ and
  Re(t_0* t_1) is pure-imaginary). Harmonic content genuinely extends to k ~ 8
  (floor(M+1)/2 of the leading omitted term t_17), but amplitude decays with k.
- **Nyquist for k≤4 is 32 thetas, NOT 16.** At N=16 equispaced over [0,2π):
  cos(4θ) distinct; cos(8θ)=(-1)^j (Nyquist, marginal); cos(12θ_j)=cos(3πj/2)≡cos(4θ_j)
  (aliased to k=1); cos(16θ_j)=cos(2πj)=1 (aliased to CONSTANT). So k=3,4 carry zero
  independent information at 16 points. Rule: N ≥ 8K for k=1..K; K=4 → N≥32.
- **De-rate 0.85 floor suffices; do not double bake cost with off-grid midpoints.**
  Post-fix off-grid θ over-pred is small (smooth low-order minima dominate the
  over-serve risk; sharp peaks are in the conservative direction).
- **Oracle bias direction:** `_measure_w_low_true` returns `lo` (always-honest lower
  bound after 24 log-bisections); relative width ~3.4e-7. So w_low_true is a LOWER
  bound → ratio w_low_fit/w_low_true is biased UP (over-serve flagged more readily),
  and a literal zero-tolerance `<=` can false-positive only if the ratio sits within
  ~1e-7 of 1.0 — impossible under the 0.85 derate. Tiny eps 1e-5 relative on the ratio
  is justified insurance, not a widened tolerance.
