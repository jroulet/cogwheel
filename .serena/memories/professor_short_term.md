## D2 fold quadrant plan — Professor consultation (2026-08-07)

### Physics grounding
The Fermat potential for a diagonal macro matrix A=diag(1-γ,1+γ) at β=0, κ=0:
  τ(x; y) = ½ Σ λ_i x_i² - Σ y_i x_i + ½ Σ y_i² - ln|(x1,x2)|

Under any sign flip y_i → -y_i, the substitution x_i → -x_i in the full
diffraction integral maps the integrand pointwise onto itself, and the
integration measure/domain (ℝ²) are invariant. Therefore F(w; y) is D2
(Klein four-group) invariant for BOTH parities — the sign of λ_i does not
appear in the reflection argument. This is an exact continuous symmetry,
not an approximation.

The dual-deltoid saddle case (γ > 1): lobe 1 is the mirror image of lobe 0
across the y2-axis, so the lobe-local angular coordinate folds identically.

### Image-level confirmation
Under y1 → -y1, image x1 → -x1 (and similarly for y2). Each image's:
- delay τ: invariant (cross-term y1x1 → (-y1)(-x1) = y1x1)
- magnification: invariant (Hessian quadratic in x)
- Morse index: invariant (signature unchanged by coordinate reflection)

Hence F(w) = Σ √|μ_a| e^{iwτ_a + iπ n_a/2} is element-by-element invariant.

### Neighbourhood of γ=1 (parity boundary)
γ→1 has |λ1|→0 → infinite extent on BOTH sides (F026). The fold is valid
arbitrarily close to the parity boundary, but the parity boundary itself
is refused by name (det A = 0 → LensDomainError), so no edge case.

### Carrier-phase ambiguity caveat for envelope comparison
The SACR-C tau_c-demodulated envelope E(w) = F(w) e^{-iwτ_c} depends on
which nearest caustic point `nearest_caustic_point` selects. Under D2, an
equivalent nearest point at a different angle would change τ_c by an
additive constant Δ, making the raw envelope differ by e^{-iwΔ}. The
farfield kernel-sum label (FARFIELD_KERNEL_SUM) is carrier-independent and
should be bitwise D2-invariant. Tests comparing raw chart envelope values
must account for this phase ambiguity.

### Corridor test assessment
After folding to (|y1|, |y2|), sources from BOTH lobes map to lobe 0's
local frame. The corridor test correctly discriminates:
- Inside lobe (|y1| ≈ a): near_this ≈ 0 ≪ near_other ≈ 2a → serve
- Corridor (|y1| ≈ 0): near_this ≈ near_other → decline
The test is not degenerate after folding.
