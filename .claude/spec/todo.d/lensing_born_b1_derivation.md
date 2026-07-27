---
section: Backlog
---
- **Derive the Born correction coefficient `b1`, then wire the rung** — the
  Born module (`chang_refsdal/_born.py`) ships DORMANT: its two-term series
  `F_born = sqrt(mu_macro) * exp(1j*w*Phi_geo) * (1 + 1j*(w/2)*b1/Q2r + ...)`
  uses `b1 = 1.0` as a placeholder, giving up to ~13% disagreement with
  `operator.F_op` across the target annulus `3.0 < |y| <= 4.2426` in the region
  where `born_gate` PASSES (Inspector INS-c1-001; the build's own T1 target was
  rel err < 1e-3, missed by ~100x).

  What is already correct and tested (11 pass, 1 xfail, 4.3 s): the macro limit
  recovers `sqrt(mu_macro)`; the correction is O(w) with fitted slope 1, so the
  earlier `c1 = 1/(2w)` divergence is genuinely gone; both gate guards are
  reachable-red; `born_envelope` demodulates through `geom.t_min` and the
  far-field reconstruction round-trips to ~1e-16.

  Owed: a closed form for the O(1) numerator `b1` from the geometry of the
  mass-sheet-reduced coordinates, then re-derive T1 as a real accuracy gate
  against `operator.F_op` and only then remove the fall-through at the fact-4
  slot in `likelihood.py::_surrogate_coefficients` (the comment there marks it).

  Also owed from the same build: the `'born'` category in
  `surrogate_census.classify_fallthrough` was in the plan but is absent from the
  tree, so annulus draws are still attributed to `out-of-box`.

  Until this lands the annulus is NOT covered and the exact engine serves it --
  which is correct, certifiable (`w * |y| <= 60` never binds inside the prior),
  and not zero-quadrature.
