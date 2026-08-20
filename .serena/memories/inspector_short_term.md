# Inspector Short-Term Observations

## 2026-08-20 (diffractive_certificate_fit_corner RE-REVIEW #2 — INS-3-001 re-check, code PASS)

Scope: fresh full review of the uncommitted working tree for build "fix w_low_fit corner + move full bake to driver". Re-checked the ONE mandatory open finding (INS-3-001) plus a fresh pass over all changed code/tests, with fresh test runs (not trust).

### MANDATORY RE-CHECK (INS-3-001, SPEC.md doc staleness)
- NOT RESOLVED. `git status` shows `.claude/spec/SPEC.md` is NOT in the changed-file set; grep confirms
  line ~54 still carries the LOW-W DIFFRACTIVE RUNGS paragraph describing the RETIRED closed-form
  truncation certificate `w_low = (gamma'/2)*[sqrt(mu_macro)*R_{M+1}/bar]^(1/(M+1))` + root-find fallback.
  `w_low_fit` (grep count 0 in SPEC.md) and the even-harmonic cos(2k theta) + parametric-caustic basis are
  absent. The plan explicitly listed SPEC.md as an expected change; it never landed. Recurring doc-staleness
  lineage. Direction: Librarian doc-sync, NOT a Coder defect. Do NOT re-open as a code bug.

### FRESH VERIFICATION (all re-run today, green)
- test_lensing_diffractive.py fast tier: 33 passed / 3 skipped (the 3 gated: zero-over-serve, diagnostic
  plot, corner pin) in 87 s — file well under the 5-min ceiling; teeth test (~39 s) under the 60 s
  single-test budget.
- Corner pin test under COGWHEEL_DIFFRACTIVE_FULL_BAKE=1: 1 passed (4.9 s) — < 2.0 bar holds (1.986x).
- CausticPointMirrorFidelityTestCase: 1 passed (part of 29 green in test_lensing_geometry.py).
- test_lensing_part0_mechanical.py D2 symmetry suite: 41 passed (period-pi + reflection +
  pi/2-changes-value all green, the pi/2 test keeps the symmetry pins from being vacuous).
- Constant consistency (imported, counted): N_HARM=7 == len(HARMONIC_COEFFS)=7; n_poly=10 ==
  len(POLY_COEFFS)=10 == len(_fit_poly_exponents(2)); caustic coeff -0.7267 (negative, per premise);
  derate 0.503444.
- `_fit_features` return = poly + harmonics + (caustic,), n_poly+n_harm+1 = 18; `w_low_fit` slices
  features[n_poly:n_poly+n_harm] and uses features[n_poly+n_harm] for the caustic term; script design
  matrix n_poly+n_harm+1=18 — all consistent. `_evaluate_fit` does np.dot(coeffs, features) with the full
  18-vector, so the +1 caustic column is handled (n_poly/n_harm params are dead args, pre-existing).

### NEW DERIVATION (physics, confirms feature normalization)
- `caustic_point(gamma_prime, theta, beta=0, kappa=0)` computes EXACTLY the reduced caustic radius:
  full caustic `_caustic_source(gamma,beta,kappa)` = lam*[diag(1-g',1+g')x - x*|x|^-2 /lam-normalised],
  so /sqrt(lam) cancels the lam prefactor and yields the same closed form as the kappa=0 pure-shear
  caustic in the eigenframe. Hence `log(|y'|/|y_c|) = log(r/|y_c_reduced|)` matches `s=r^2` exactly.
  The caustic feature is physically the reduced-offset/reduced-caustic log ratio, not a loose proxy.

### No new code findings. Carry:
- INS-3-001 -> Librarian (SPEC.md LOW-W DIFFRACTIVE RUNGS paragraph + spec_changelog + completion record).
- Provisional smoke coefficients (de-rate 0.5034, corner 1.986x) are committed BY DESIGN; full bake is a
  DRIVER step (`python scripts/fit_diffractive_certificate.py --scale full`).
- `_fit_features` has unused `lam`/`sqrt_mu` params (pre-existing, not this diff).
- _DIFFRACTIVE_FIT_N_HARM=7 at 32 thetas is alias-free (Nyquist >4k, 32>28); a 64-theta bump reopens k=8+.
