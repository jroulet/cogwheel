---
date: 2026-07-28
---

### Added: Born carrier + band split for the far annulus; 'born' census category

`_born_factors` now returns the DERIVED coefficients `b1` and `a0` (F023):
`b1 = -lam*(2*lam*r0_sq - x0.y)/(det_a*r0_sq)` replaces the wrong-signed
placeholder `b1 = 1.0` (a point mass gives `-1`, not `1`), and the real,
w-independent `a0` term — previously missing from the series — is
`a0 = -lam*(lam*r0_sq - x0.y)/(det_a*r0_sq)`.

`born_lead_carrier` is the new serve object: the lead-only carrier
`sqrt(mu_macro)*exp(1j*w*phi_geo)`, with NO `a0`/`b1` correction. F025 found
that the resolved-image `(a0, b1)` correction serves nowhere — it violates the
exact `F(w->0) = sqrt(mu_macro)` limit (F009) and inflates the demodulated
residual's azimuthal node count 2.5x-11x. `a0`/`b1` stay in the module as the
correct macro-limit diagnostic (`born_amplification`, `born_envelope`), not the
serve path.

`channels.born_carrier_from_partition` assembles the band-split carrier at
`w * Delta_tau = RHO_END`, with `Delta_tau` read directly from the partition's
real-image delays (never recomputed, never `phi_geo`, never `w*r0_sq` — that
currency was a positive-parity coincidence, F024): below the split, the
lead-only carrier; above, the two-real-image geometric-optics sum plus
`farfield_ghost_term` where admitted (tolerating `GhostDomainError`, since the
ghost is additive, never a precondition).

`born_gate`'s guard A is re-keyed to the same `w * Delta_tau >= RHO_END` band
split, and gains an exact exterior fence: `gamma < 3/4`, where the astroid
caustic's `max|y| = 2*gamma/sqrt(1-gamma)` reaches the annulus inner edge
exactly at `gamma = 3/4` (F025/F026).

`surrogate_census._FALLTHROUGH_CATEGORIES` gains `'born'` (now six-way MECE):
far-annulus positive-parity draws were previously mis-attributed to
`out-of-box`.

The likelihood serve slot (`_surrogate_coefficients`) STAYS unwired: the
served object is `F_carrier` minus a driver-trained residual chart, and that
chart does not exist yet (a TRAIN_TIER artifact). Until it is trained the
annulus continues to fall through to the exact engine, which is certifiable
throughout (`w * |y| <= 60`). See FINDINGS F023-F026.
