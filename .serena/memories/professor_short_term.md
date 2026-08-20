# Diffractive w_low_fit corner-fix review (inference review mode, 2026-08-20)

Reviewed WP-1 (even-harmonic + caustic representation of w_low_fit). Fast-tier
tests: test_lensing_part0_mechanical.py 41/41 PASS, test_lensing_diffractive.py
33/33 PASS + 3 gated skips. Direct engine measurements (n_w=16):
- caustic coefficient = -0.7267 (NEGATIVE, correct — ceiling dips toward fold)
- corner raw over-prediction = 1.9863x (served/honest 3.4565/3.4565)
- dropping caustic feature -> 2.4587x (crosses 2.0 bar; self-falsification PASS)
- D2 symmetry: period-pi + reflection ~1e-12, pi/2 changes value (PASS)
- de-rate 0.503444 = 1/1.9863 = sole margin (derate-teeth PASS)

CONCERN: acceptance re-scoped from de-rate>=0.70 (ratio<1.43) to ~0.5
(ratio<2.0) via INS-1-001 corner-resonance limitation (marginal order-16
resonances near fold, ~0.1-wide, n_w=16 coarse scan samples inconsistently).
Physically sound and documented, but a target reduction the operator must
sign off. Zero-over-serve oracle + corner pin gated behind
COGWHEEL_DIFFRACTIVE_FULL_BAKE=1; final full-bake validation operator-deferred.

Minor: de-rate emitted via round(derate,6) ROUNDS UP (0.503443949->0.503444),
inflating served ceiling ~1e-7 above honest at corner (served/honest=
1.0000001). Negligible vs 1e-4 bar, but floor/truncate would be conservative
by construction.
