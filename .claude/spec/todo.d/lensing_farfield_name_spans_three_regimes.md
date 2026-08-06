---
section: Backlog
---

- **`FarFieldChart` IS NOT A FAR-FIELD OBJECT — the name spans three regimes**
  `[housekeeping]` — audited 2026-08-06. DEFERRED DELIBERATELY: the confusion
  is endemic and the rename touches a large surface. Recorded so it is not
  rediscovered.

  The exterior splits physically into an INTERMEDIATE field (ghost weakly
  damped, `Im tau_g` small, ghost still contributing) and a TRUE far field
  (ghost exponentially dead, one image dominant). The code calls all of it
  `farfield_*`: `FarFieldChart`, `farfield_envelope_from_partition`,
  `FARFIELD_KERNEL_SUM`, `farfield_eps_max`. The only place the physical
  boundary appears at all is `_GHOST_SEPARATION_MIN`, which decides whether
  the ghost term is subtracted from the LABEL; nothing downstream treats the
  two regimes differently in accuracy bar, node budget, or tiling.

  Worse, the class ALSO currently hosts interior tiles (see
  [[lensing_interior_wedge_chart_unwired]]), so one name covers intermediate
  field, far field, and interior.

  This is the failure mode step 5 of [[lensing_caustic_relative_coordinates]]
  names: "a public symbol named for a retired concept is how the concept
  survives its own deletion." Here the concept is not retired, it is
  OVERGENERALISED — the name records the class's first use, not its role.

  Candidate: `CausticFixedChart` (it is the caustic-fixed `(s, d)` chart,
  valid at bounded distance from the caustic curve on either side).

  A SECOND overloaded symbol in the same family, worth doing in the same pass:
  `rho` carries three meanings in `_to_caustic_fixed` — positive-parity
  interior `|y| / r_caustic(theta_c)` (multiplicative), positive-parity
  exterior `1 + |y| - r_caustic(theta_c)` (additive, directional), and
  macro-saddle `1 + |y| - _caustic_reach` (additive, SCALAR reach). It reads
  as a near-synonym of the wedge chart's `r`, which is always the
  multiplicative form. Note the piecewise map is C0 but NOT C1 at `rho = 1`
  (`drho/d|y|` is `1/r_caustic` inside and `1` outside); harmless only because
  no chart straddles `rho = 1`.

  DO NOT start this before the interior-wedge wiring lands: that change
  removes the interior from `FarFieldChart` and shrinks the rename's scope.
