# Professor short-term

## 2026-08-13 — fold-ppGO interior certificate BUILD review (Phase 2) — PASS

Reviewed `test_lensing_ppgo_certificate.py` (16 tests, all pass, 9.6s) against
the 5 specs. Independent numeric probe (shipping funcs, cogwheel-newlal python):
- Leg-1: interior rho=0.5 -> 4 real images, exterior rho=1.5 -> 2, census flip
  bisects to rho=1 within ~1e-13 (bar 1e-3). Predicate == closed-form caustic.
- c1 relerr 1.6e-15, c2 relerr 2.2e-14 vs saddle_coefficients (bar 1e-12; matches
  Fact 6 memory 2.4e-15/5.8e-14). c3 |Re|/|Im| = 7e-15 => purely imaginary. OK.
- ppgo_error_estimate w^-3 ratio err 1.3e-16 (exact cube). None on w<=0 and
  non-finite mu. OK.
- Interior conservativeness (rho=0.5, 5 configs x w{20,40,60}): max true_err/cert
  = 0.885 < 1.0 — never optimistic (consistent with earlier 0.980 consult figure;
  config-set dependent, both safe).
- Near-caustic (rho 0.90-0.95, 9 pts): certs 0.5-2772, ALL self-refuse (cert*2 >
  CERTIFICATION_BAR=1e-4); AND even so true_err/cert 0.03-0.38 all <1. The
  forbidden ADMIT-AND-OPTIMISTIC state never occurs. c3 diverges -> cert blows up
  -> refuse, exactly the invariant that makes dropping leg 2 sound.
Self-falsification class has teeth (wrong reach, real-c3, w^-2 exponent, shrunk
cert all caught). Verdict PASS. Heavy full-sampling validation operator-deferred
(not run; not in scope for fast gate).
Note: no image-render tool this session — plots verified via numeric probe that
reproduces each plot's encoded quantity.

## (prior) 2026-08-13 — design consult on ppgo_interior_certificate.md
Rulings issued (drop leg 2 on interior; weight sqrt|mu_a|, exponent w^-3; eval at
w_min; refuse None on non-finite; safety factor 2.0; 4-real-image predicate is the
sound cheap guard; absolute-vs-bar conservative since interior max|F|>=sqrt(mu_macro)>=1).
The c3 term is the eps^6 stationary-phase term; purely imaginary by Gaussian-moment
parity — imaginary-purity is a derivation invariant worth the cheap test.
