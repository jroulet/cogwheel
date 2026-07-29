---
section: Backlog
---

- **Born rung for the MACRO-SADDLE (`det A < 0`) — carrier, gate, and census
  landed 2026-07-28 (commit `31ee133`, F024/F026); only the wiring step
  remains.**

  Landed:

  1. `b1`/`a0` closed form for positive parity — done earlier (F023,
     see [[lensing_born_b1_derivation]]).
  2. Saddle expansion origin + its own convergence guard: `born_lead_carrier`
     applies the exact Morse phase `-1j` at the saddle image origin
     (F024/F009-S, NOT `cmath.exp(-1j*pi/2)` — that injects a ~6e-17 real
     part that rotates into `|F|`); `born_gate` guard B is now a two-sided
     parity-wall margin (`abs(gamma_p - 1) > DELTA_GAMMA_P`) so it catches
     degeneration from either side of `det A = 0`, and the parity wall
     `gamma_p = 1` stays the measure-zero named refusal it already was.
  3. A saddle exterior fence against the exact astroid caustic extent (F026
     closed form `saddle_caustic_max_y`, `max(off_axis, on_axis)` over the
     two candidate cusps — correcting F024's measured extent table), confining
     saddle serving to `1.0502342 < gamma < 3` (`kappa = 0`). Verified in
     `test_lensing_born.py` against the matrix-solve oracle and the F009
     magnitude pin (29 new tests total, including 6 gating the mirroring
     `'born'` saddle arm in `surrogate_census.classify_fallthrough`).
     `channels.born_carrier_from_partition` gained the macro-saddle
     above-split branch (pure two-real-image ppGO, complex ghost explicitly
     refused for `det A < 0`).

  Owed, still open:

  0. **BLOCKED ON C8 — do not train, do not wire yet.** The saddle exterior
     fence `1.0502342 < gamma < 3` is derived from `ANNULUS_INNER_RADIUS`, a
     prior-box length being retired (F036); it dissolves with the annulus
     rather than being ported. The F026 closed form `saddle_caustic_max_y` is
     real physics and SURVIVES — it is the fence built on top of it that goes.
     See [[lensing_caustic_relative_coordinates]].

  4. Wire the saddle branch (together with the positive-parity branch —
     same blocker) through the fact-4 slot in
     `likelihood.py::_surrogate_coefficients`, once the driver-trained
     residual chart exists. See [[lensing_born_b1_derivation]] for the
     shared TRAIN_TIER blocker.

  Recorded 2026-07-28 after the owner noticed it was missing from the plan
  list; carrier/gate/census landed the same day.
