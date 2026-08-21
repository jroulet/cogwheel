---
date: 2026-08-21
---

### Low-w diffractive chart: residual anchored on the non-vanishing fold/cusp reference (schema v2)

The trained low-w diffractive chart (`LowWDiffractiveChart`,
`cogwheel/data/low_w_diffractive_chart.npz`, schema `low_w_diffractive_v2`)
no longer anchors its stored residual on the exact point-mass prefactor
`C(w)`. The residual is `r_new = f_pure * sqrt(1 - gamma'^2) / F_ref(w)`,
where `F_ref` is the non-vanishing uniform fold/cusp reference
(`fold_cusp_reference`): the Airy fold q=p Wronskian form, or the uniform
Pearcey cusp form where the fold degenerates (`b3 -> 0`, the fold->cusp
transition), guarded by a non-vanishing magnitude ratio so a near-vanishing
reference declines to the exact-engine fall-through. The serve re-modulates
`F = mass_sheet_phase * F_ref(w) * sqrt_mu_full * r_pure`, rebuilding `F_ref`
engine-free from the reduced-frame geometry; the frequency axis is a uniform
`w**(2/3)` grid (`w23_grid`), replacing the former `log w` axis.
