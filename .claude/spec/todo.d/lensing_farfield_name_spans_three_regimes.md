---
section: Backlog
---

- **`farfield_*` HELPER NAMES SPAN TWO PHYSICAL REGIMES** `[housekeeping]` —
  audited 2026-08-06; `FarFieldChart` class deleted 2026-08-07 (`0a31fcf`,
  post-strand cleanup); rename of the class is moot. Remaining scope:
  `farfield_envelope_from_partition`, `FARFIELD_KERNEL_SUM`, `farfield_eps_max`,
  `_farfield_tiles`. DEFERRED DELIBERATELY:
  the confusion is endemic and the rename touches a large surface. Recorded so
  it is not rediscovered.

  The exterior splits physically into an INTERMEDIATE field (ghost weakly
  damped, `Im tau_g` small, ghost still contributing) and a TRUE far field
  (ghost exponentially dead, one image dominant). The code calls all of it
  `farfield_*` — none of the remaining helper names distinguishes the two
  regimes. The only place the physical boundary appears at all is
  `_GHOST_SEPARATION_MIN`, which decides whether the ghost term is subtracted
  from the LABEL; nothing downstream treats the two regimes differently in
  accuracy bar, node budget, or tiling.

  This is the failure mode step 5 of [[lensing_caustic_relative_coordinates]]
  names: "a public symbol named for a retired concept is how the concept
  survives its own deletion." Here the concept is not retired, it is
  OVERGENERALISED — the name records the class's first use, not its role.

  A related overloaded symbol in the same family, worth doing in the same pass:
  `rho` carries three meanings in `_to_caustic_fixed` — positive-parity
  interior `|y| / r_caustic(theta_c)` (multiplicative), positive-parity
  exterior `1 + |y| - r_caustic(theta_c)` (additive, directional), and
  macro-saddle `1 + |y| - _caustic_reach` (additive, SCALAR reach). It reads
  as a near-synonym of the wedge chart's `r`, which is always the
  multiplicative form. Note the piecewise map is C0 but NOT C1 at `rho = 1`
  (`drho/d|y|` is `1/r_caustic` inside and `1` outside); harmless only because
  no chart straddles `rho = 1`.
