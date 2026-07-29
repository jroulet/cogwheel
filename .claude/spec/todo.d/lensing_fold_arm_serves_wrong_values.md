---
section: Backlog
---

- **Tighten the fold arm's caustic fence; the `b4` route is CLOSED
  [→ spec].** F028 defect 2. Defect 1 (admission routing) closed by the
  authoritative-gate work; the arm is now fenced to `eta < _ETA_MAX_FOLD`
  (0.3) and no longer serves where F032 measured it 63%-64% wrong.

  DO NOT derive `b4`. F033 measured why: the far-field error is not the
  `q = 0` symmetric-fold assumption but the O(eta) truncation of the CUBIC
  NORMAL FORM itself. Production's `p` (from the finite cubic curvatures)
  and the CFU `p` (from the merging pair's amplitudes) agree to 0.7% at
  `eta = 0.015` and diverge monotonically as `ratio - 1 ~ 0.5*eta`
  (gamma = 0.70) to `~1.1*eta` (gamma = 0.90). `p` is wrong away from the
  caustic by the same mechanism as `q`, so refining `q` alone cannot
  recover that region.

  WHAT REMAINS: the fence threshold is unmeasured for the arm.
  `_ETA_MAX_FOLD = 0.3` was set as the complement of `ETA_MIN_GEOMETRIC`,
  which was measured for the GEOMETRIC branch. F033 shows the arm's
  amplitude is already off by 14% (gamma = 0.70) to 29% (gamma = 0.90) at
  `eta = 0.3`, and 3%-7% at `eta = 0.1`.

  Needed before changing it, mirroring what F031 did for the geometric side:
  a served-`|F|` error sweep versus an oracle binned by `eta` (F033 measured
  an amplitude RATIO, not a served error), and the coverage cost — the
  current fence already gives up ~10% of draws (measured over 2500), and
  tightening to 0.1 will give up more. Recovering that region needs a
  higher-order uniform form, not a patched cubic one.
