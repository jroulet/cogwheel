---
date: 2026-07-28
---

### Macro-saddle Born carrier, band-split, and census arm

Extended the Born (weak-deflection) carrier of `chang_refsdal/_born.py` to
the macro-saddle parity (`det A = (1 - kappa)**2 - gamma**2 < 0`), which had
previously been served only for positive parity. `born_lead_carrier` now
applies the exact Morse phase `-1j` at the saddle image origin (F024/F009-S)
so `|F_carrier|` stays `w`-independent on both parities. `born_gate` adds a
two-sided parity-wall margin (guard B, `abs(gamma_p - 1) > DELTA_GAMMA_P`)
and a saddle exterior fence built from the new F026 closed-form helper
`saddle_caustic_max_y`, confining saddle serving to the exterior band
`1.0502342 < gamma < 3` (`kappa = 0`) where the astroid caustic stays inside
`|y| < 3`. `channels.born_carrier_from_partition` gains a macro-saddle
above-split branch: the pure two-real-image geometric-optics sum, with the
complex ghost explicitly refused (`geometry.ghost_kernel`'s sqrt branch is
not derived for `det A < 0`); the positive-parity above-split path
(ppGO + `farfield_ghost_term` where admitted) is unchanged.
`surrogate_census.classify_fallthrough` gains the mirroring saddle arm of
the `'born'` fall-through category, keyed on the same shared closed-form
fences. `born_amplification`/`born_envelope` remain positive-parity-only
resolved-image diagnostics — the `a0`/`b1` correction is not derived on the
saddle. The serve slot stays unwired on both parities: the driver-trained
residual chart (`F_exact - F_carrier`) is a TRAIN_TIER artifact that does
not yet exist, so annulus draws still fall through to the exact engine.

29 new tests added to `test_lensing_born.py`: 11 covering the carrier
against a matrix-solve oracle, the F009 magnitude pin, and the exact fence
band; 12 covering the saddle band split (`w * Delta_tau` currency, ppGO-only
vs ghost-admitted node counts, low-band residual splineability); and 6
covering the saddle arm of `surrogate_census.classify_fallthrough`.
