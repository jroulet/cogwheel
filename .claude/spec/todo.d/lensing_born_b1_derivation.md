---
section: Backlog
---
- **Wire the Born carrier + band-split residual charts (`b1` is DERIVED)**
  `[→ spec]` — the derivation blocker is discharged (FINDINGS F023, Professor
  2026-07-28). What remains is implementation, and the SHAPE changed: the
  analytic term is a CARRIER whose residual a chart interpolates, not a
  standalone approximation that must hit a tolerance. The rung's old T1 target
  of 1e-3 was never the right bar.

  **The coefficients** (single edit site, `_born_factors`; both collapse onto
  quantities already computed there, so no new geometry and no fifth
  convention site):

      b1 = -lam * (2.0*lam*r0_sq - x0_dot_y) / (det_a * r0_sq)
      a0 = -lam * (lam*r0_sq - x0_dot_y) / (det_a * r0_sq)

  `b1 = 1.0` was a placeholder with the WRONG SIGN (a pure point mass gives
  `-1`), and `a0` was missing from the series entirely — `born_amplification`
  and `born_envelope` need `+ a0/q2r` added to `correction`.

  **The ladder** (measured node counts in F023):
  1. `w < w_split` (`w * r0_sq <~ 8`, i.e. `w ~ 0.5`): carrier ALONE — no
     second image, no ppGO, no complex ghost. Chart the residual: 4-15 nodes
     on `log_w`, 4 per y-axis, prior-universal tiles.
  2. `w >= w_split`: `geometric_amplification` with BOTH real images at full
     C1/C2 + `farfield_ghost_term` where admitted. Chart the residual: 4-8
     nodes. Tolerate `GhostDomainError` — the complex ghost is not universally
     available in the annulus.
  Do NOT mix the bands: ppGO below `w = 0.05` inflates the residual by five
  orders of magnitude via its `1/w**2` kernel.

  **No low-`w` analytic rung.** The chart absorbs `ln(w/2)` on its existing
  `log_w` axis at zero node cost (F023). The Chang-Refsdal low-`w` closed form
  was derived and is recorded there for provenance, but building it would add
  a fifth ladder component for no measured gain.

  **Guard A must be re-derived**, two ways: its estimate rescales by `b1**2`
  (3.3x at `gamma' = 0.45`, ~4e4x at the guard-B edge, since
  `|b1| <= 1/(1-gamma')`), AND it should be re-keyed to the actual band-split
  criterion `w * r0_sq` rather than the `O(w**2/q2r**2)` term, which is far
  smaller than the two terms it ought to be catching.

  **Correct the module docstring.** Its WHY says the low-`w` far zone "varies
  on the Einstein scale, so trained tiles there are prior-sized". Measurement
  says the reverse: demodulating `exp(1j*w*phi_geo)` removes that variation
  entirely (4 y-nodes at `w <= 0.2` vs 9-17 at `w >~ 1`). Also correct the
  "low-frequency far zone" framing — this is a MID-`w` resolved-image
  expansion, valid for `1/q2r**2 <~ w << q2r`.

  Only after the above: re-derive the accuracy gate in the residual currency
  and remove the fall-through at the fact-4 slot in
  `likelihood.py::_surrogate_coefficients` (the comment there marks it).

  Also still owed from the original build: the `'born'` category in
  `surrogate_census.classify_fallthrough` is absent from the tree, so annulus
  draws are still attributed to `out-of-box`.

  Saddle branch: see [[lensing_saddle_born]].
