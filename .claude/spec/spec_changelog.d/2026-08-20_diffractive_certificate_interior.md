---
date: 2026-08-20
bump: patch
---

SPEC.md's LOW-W DIFFRACTIVE RUNGS paragraph (Rung P) corrected to the
SHIPPED admission mechanism: Rung P is admitted by the O(1) fitted
truncation-certificate surface `w_low_fit` (log-log degree-2 polynomial +
even-harmonic `cos(2 k theta)` basis + directional-caustic feature, with
the TOTAL-amplitude `(1/(M+1)) log(lam sqrt_mu)` normalization pinned by
construction), fitted to the ENGINE-HONEST ceiling and baked by
`scripts/fit_diffractive_certificate.py`, replacing the retired
per-proposal `diffractive_w_low` scan (the old text still described the
formula-scan gate). The paragraph now states the de-rate as the SOLE
conservativeness margin (`min(., _DIFFRACTIVE_FIT_CEILING)` is a hard
oracle-domain cap, `W_CEILING_SCHWINGER`, no-op wherever the fit is
calibrated), the near-fold shell fence (`_DIFFRACTIVE_FIT_FENCE_*`, the
shell declines as `None` falling through to fold arm / exact engine),
and the deep-interior coverage: the calibration grid now reaches `r ~
0.1` (build INS-3-001, 2026-08-20), so the interior is served by the
calibrated de-rated fit at its engine-honest ceiling (~4-41), not at
the clip.
