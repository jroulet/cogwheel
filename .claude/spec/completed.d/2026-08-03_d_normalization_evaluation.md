---
date: 2026-08-03
section: Backlog
---

**Normalize the far-field `d` axis by curvature radius** — EVALUATED, REJECTED.

Build `eval_d_norm` (OpenCode, 2026-08-03) ran the Professor+Simplifier
evaluation from `brief_evaluate_d_normalization.md`. Conclusive finding:
do NOT normalize d by R_c. Five independent reasons:

1. Wrong physics: the Airy transition scale is ξ = (3wΔτ/4)^{2/3}, not d/R_c.
   Normalizing by R_c alone doesn't collapse the fold structure (still depends
   on w, b₃, λ_h).
2. Wrong chart: the far-field chart doesn't serve the Airy transition regime
   (d ~ R_c); the tube chart does. The far-field chart operates at d >> R_c.
3. Breaks tensor-product separability: d/R_c(γ,θ) couples spatial and parameter
   axes, which is architecturally forbidden for a separable spline grid.
4. R_c diverges near cusps: introduces numerical instability in regions the
   chart already excludes via cusp windows.
5. Non-problem: the current 4 d-nodes achieve eps < 1e-3. The ~2× R_c variation
   within a gamma band is smooth monotone drift handled by the γ-axis directly.

Recommendation: proceed to training with the current absolute-d architecture.
If a tighter eps bar (1e-4) later exposes d-axis interpolation error, add 1-2
d-nodes (50% training cost, no serve cost, no architecture change).

Zero work packages. No code change.
