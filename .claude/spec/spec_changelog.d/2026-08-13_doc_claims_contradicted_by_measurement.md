---
bump: minor
---

Corrected four SPEC.md claims that today's measurements contradict, so the
spec stops asserting things an agent would act on and be wrong.

- **Low-w flat extrapolation** was stated as universally safe ("the envelope
  is smooth and nearly constant below the first Airy fringe"). That is
  label-dependent: true for the tube / SACR-C envelope, FALSE for
  `FARFIELD_KERNEL_SUM`, which diverges into the diffractive bottom. A
  correctly tiled kernel-sum chart queried below its `farfield_w_floor`
  passes every gate and serves 468x `max|F|` wrong (F070). Recorded as a
  blocker on the full-box training campaign, not a today-regression — no
  surrogate artifact ships.

- **The fold-ppGO interior handoff** was described purely as a validity gate
  (`xi >= 4` AND estimate `<= CERTIFICATION_BAR`). The rung is measurably
  wrong by ~21% where it serves, on 791 live census draws, because the
  estimate is exactly `(4/3) c_A / (w * dtau)` — it decays as `1/w` while the
  true error is w-independent, so its optimism grows without bound in exactly
  the direction the gate opens (F069). The gate also selects AWAY from where
  the fold form is valid.

- **The cusp ppGO fast rung's predicate** now carries the structural exterior
  discriminator `len(images) < 4`. `nearest.distance >= _ETA_MAX_FOLD` alone
  did not discriminate interior from exterior, and once `_R_PPGO_ERROR_CONST`
  fell 3.0 -> 0.10 the rung began serving interior 4-image sources up to 155%
  wrong.

- **`test_lensing_fold_ppgo_handoff.py`'s description** claimed it certifies
  "fold-ppGO accuracy vs exact engine at high xi". It does not and cannot —
  the independent oracle ceiling is `W_CEILING_SCHWINGER = 60`, not 150,
  because `F_op` returns the uniform arm above 60. The suite now certifies
  protective refusal instead.

Also cross-referenced F061: its cost table is for a DIRECT `f_schwinger`
call and must not be transferred to `F_op`, which often never reaches the
mpmath path in `(60, 150]` because the arm intercepts first.
