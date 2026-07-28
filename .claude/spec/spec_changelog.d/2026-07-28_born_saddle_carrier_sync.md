---
date: 2026-07-28
bump: minor
---

Update the Born rung Conventions bullet (stale since commit `31ee133`,
Inspector INS-10-001): the carrier now serves BOTH macro parities, not
positive parity only. `born_lead_carrier` applies the exact Morse phase
`-1j` on the macro saddle (`det A < 0`); `born_gate` gains a two-sided
parity-wall margin (guard B) and a saddle exterior fence via the F026
closed form `saddle_caustic_max_y` (serving band `1.0502342 < gamma < 3`);
`channels.born_carrier_from_partition` gains a macro-saddle above-split
branch serving the pure two-real-image geometric-optics sum with the
complex ghost explicitly refused. `born_amplification`/`born_envelope`
remain positive-parity-only diagnostics. Serve slot is still unwired
(TRAIN_TIER residual chart not yet built) — unchanged on both parities.
