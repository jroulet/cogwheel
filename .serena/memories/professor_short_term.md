diffractive_certificate_fit REVIEW (Professor, 2026-08-19) — VERDICT FAIL.

The `w_low_fit` surface OVER-SERVES off-grid in angle by up to ~5x. Root cause:
`_DIFFRACTIVE_FIT_N_HARM = _DEFAULT_MAX_ORDER = 16` harmonics `cos(4k theta)`
fitted on only 8 eigenframe thetas (multiples of pi/4) + 12 random
(beta,kappa,r,theta) rows. At the 8 grid angles `cos(4k theta)` aliases to 2
patterns (even-k -> +1, odd-k -> (-1)^m), so 16 angular DOFs are wildly
underdetermined and lstsq produces catastrophic off-grid oscillation.

MEASURED (independent, reusing scripts/fit_diffractive_certificate.py's
`_measure_w_low_true`/`_unreduced_source`, exact `f_schwinger` oracle):
- gamma=0.2, r=0.9 (beta=kappa=0), theta sweep [0, pi/2]: w_low_fit oscillates
  0.004 .. 60.0 (the DD cap) while w_low_true stays 13.7-21.4; 8/33 off-grid
  probe angles OVER-SERVE, worst ratio ~4.2x (theta=pi/8: fit 60 vs true 14.4).
- gamma=0.3 r=0.9: worst ratio ~5.0x (theta=pi/8, fit 36.8 vs true 7.4).
- End-to-end at (gamma=0.2,r=0.9,theta=0.6): w_low_fit=32.93 but series
  breaches CERTIFICATION_BAR=1e-4 at w~20 (rel 3.7e-4) and hits 9e-2 at w=32;
  honest ceiling ~17.3. Silent interior 1e-4 breach — the bug class under repair.
- Tightness ALSO broken off-grid: ratios as low as 0.004-0.05 (Y_REF=(0.8,0.4)
  at gamma=0.1 gives fit 2.16 vs true 40.9).

Why the committed suite is green: FullGridCertificateOracleTestCase probes only
the SAME 8 on-grid thetas + 12 random rows (which never land in aliasing
troughs); TruncationCertifiedBandTestCase uses only CLEAN_GAMMAS at Y_REF. The
INS-1-001 migration DELETED the exact tests that would have caught this:
NONMONOTONE_DRAW, CeilingTightnessTestCase, CeilingMonotonicityTestCase.
The derate 0.745 (=1/max_on_grid_overpred) cannot bound the ~5x off-grid
over-prediction. The `_DIFFRACTIVE_FIT_N_HARM = _DEFAULT_MAX_ORDER` coupling is
the latent trap, now realized. Fix: cut harmonics to ~k<=4 (resolvable at 8
thetas) OR >=32-theta re-bake + an off-grid over-serve re-validation; re-derate
against the off-grid worst.