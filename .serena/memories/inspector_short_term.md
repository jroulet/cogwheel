# Inspector Short-Term Observations

## 2026-08-19 (diffractive_certificate_fit — pass 4, RESOLVED / PASS)

Scope: final re-check after the Coder re-baked the w_low_fit coefficients.
Verified EMPIRICALLY: ran `--scale full` (438 s) AND the full-grid engine
oracle suite, not by trusting comments.

### RESOLVED (all three mandatory re-checks)
- INS-2-001 / INS-3-001 (stale baked constants over-serve): the shipped
  constants are now BYTE-IDENTICAL to a fresh `--scale full` run at HEAD
  (7eeedee). Derate = 0.745168 (was 0.85), 236/236 conservative (worst ratio
  1.0000), median 0.7506, p90 0.8852. Poly coeffs (10) + harmonic coeffs (16)
  match verbatim. The 0.85-derate over-serve (up to 4.72x) is gone.
- INS-3-002 (oracle suite too narrow): FullGridCertificateOracleTestCase now
  sweeps `_grid_points('full', 42)` (252 rows, imported from the calibration
  script — single source of truth) and asserts ZERO over-serve at w_low and
  0.9*w_low. test_removing_derate_trips_overserve (self-falsification) proves
  derate=1.0 over-serves somewhere. TestWLlowFitSMonotonicity re-derived from
  the live surface. 34/34 diffractive, 84 mechanical+born, 30 born-reach,
  42 census all pass.
- INS-2-002 (dead w_lo): confirmed resolved in pass 3.

### New finding (Librarian scope)
- SPEC.md ~line 54 "LOW-W DIFFRACTIVE RUNGS" still describes the retired
  closed-form + root-find + N/2N self-certificate (`w_low = (gamma'/2)*[...]`
  "root-find fallback"); code now uses the fitted `w_low_fit`. SPEC.md was in
  the plan's expected-change list but is NOT in the changed files. Doc-sync
  to Librarian (INS-4-001).

### Carry (unchanged trivial notes)
- `_DIFFRACTIVE_FIT_N_HARM = _DEFAULT_MAX_ORDER` couples harmonic count to
  truncation order (latent zip() truncation trap on an order bump).
- `w_low_fit` now VALIDATES y shape (np.asarray + shape check) — the old
  "un-validated float(y[0])" note is CLOSED.
- `_measure_w_low_true` bare `except Exception` (calibration script, ok).

## 2026-08-19 (diffractive_certificate_fit — re-review, pass 3, STILL BLOCKING)

Scope: re-check INS-2-001 (stale baked w_low_fit constants over-serve) and
INS-2-002 (dead w_lo param). Verified EMPIRICALLY (ran the engine oracle and
the full-scale fit script), not by trusting the Coder's comment.

### INS-2-002 — RESOLVED (verify before re-flagging)
- `_diffractive_bottom_ceiling` now has signature `(self, lens, *, w_hi=None)`;
  `w_lo` dropped from the wrapper and all 3 call sites (likelihood.py:2885,
  :3083; serve_route_census.py:857). Docstring rewritten to describe the
  `_band_split_mask`-owned null-split mechanism (no `w_low <= w_lo` claim).
  Retired scan symbols (`diffractive_w_low`, `_rootfind_w_low/high`,
  `_honest_tail_ratio`, `_DIFFRACTIVE_CERT_SAFETY`, `_CERT_REFERENCE_W`) have
  ZERO code references in production (only one docstring prose mention at
  _diffractive.py:368).

### INS-2-001 — NOT RESOLVED (BLOCKING; re-confirmed by fresh probe)
The shipped constants (`_DIFFRACTIVE_FIT_DERATE = 0.85` + old coeffs) STILL
over-serve the engine-honest ceiling. My probe (47 measurable grid points,
kappa=0, beta=0): **25 over-serve**, up to **4.72x** (gamma=0.5, r=1.3:
fit 3.132 vs true 0.664). The finding's cited point reproduces exactly:
gamma=0.25, y=(0.3,0): fit=52.83 vs true=21.6 (2.45x), relerr 5.6e-3@w=30,
0.40@w=40, 17.8@w=52.83. Over-serve regions: small r~0.3 AND large r~1.1-1.3,
gamma>=0.15 (worse toward gamma=0.5). The Coder did NOT re-bake — they only
REWROTE the provenance comment to document the staleness MORE precisely
("baked constants are NOT bit-reproducible at HEAD ... de-rate 0.7452 ...
Re-bake ... and paste"). Running `scripts/fit_diffractive_certificate.py
--scale full` at HEAD (7eeedee, 445s, 252 pts -> 236 measurable) yields:
de-rate **0.745168**, drastically different poly+harmonic coefficients,
236/236 conservative, worst ratio 1.0000. THE COMMENT'S NUMBERS ARE THEMSELVES
STALE: it claims "224/224 measurable, median 0.7572, p90 0.8869" but the live
run gives "236/236, median 0.7506, p90 0.8852". Fix = paste the emission block
verbatim (derate 0.745168 + new coeffs + provenance SHA 7eeedee).

### Engine-oracle suite still too narrow (the reason the bug ships green)
test_lensing_diffractive.py sweeps CLEAN_GAMMAS=(0.1,0.2,0.3) x Y_REF=(0.8,0.4)
(s=0.8, r~0.894) — which sits in the CONSERVATIVE region of the fit (probe
ratios 0.94-0.99 there). The over-serve regions (small r, large r, gamma
0.4-0.5) are uncovered, so the suite is GREEN (31 passed) while the shipped
surface over-serves 4.7x. test_lensing_part0_mechanical.py (40 passed) tests
monotonicity/symmetry/ceiling-cap/derate-MECHANISM only — the
`TestWLlowFitSMonotonicity` docstring explicitly admits the fit's positive
`log(s)**2` coefficient turns the surface UP at s>~0.5 (a fit artifact) but
sweeps only s<=0.4 and does NOT pin the large-s over-serve. The derate-teeth
prove the mechanism, not that the SHIPPED derate+coefficients are conservative.

### Pattern re-confirmed (baked-constant flavor, now 3rd occurrence)
FITTED-CERTIFICATE STALENESS SHIPS GREEN: a fitted certificate's oracle suite
passes while the shipped surface over-serves up to ~5x, whenever the suite's
sweep grid (CLEAN_GAMMAS x one Y_REF) is a strict subset of the calibration
grid. A fitted surface's "never over-serve" claim is only as honest as (a)
coefficients re-baked after every series change AND (b) the oracle sweep
covers the FULL calibration domain (gamma 0.05-0.5, r 0.3-1.3, kappa>0,
multiple thetas), not a handful of points.

### Trivial / carry
- `_DIFFRACTIVE_FIT_N_HARM = _DEFAULT_MAX_ORDER` couples harmonic count to
  truncation order (latent zip() truncation trap if order bumps).
- `w_low_fit` docstring claims "ValueError if y not shape (2,)" but the body
  does un-validated `float(y[0])/float(y[1])` (IndexError/TypeError) —
  PRE-EXISTING (old diffractive_w_low had the same), not introduced here.
- `scripts/fit_diffractive_certificate.py::_measure_w_low_true` bare
  `except Exception: return None` (calibration script — acceptable).
