---
date: 2026-08-18
section: Backlog
---

- **Low-w diffractive rungs SHIPPED — the band bottom serves on both
  parities; engine residual fell 53.30% → 24.10%** `[→ spec]` — build
  `low_w_diffractive_rung` (recovery launch via SDK_RESUME_PLAN after
  the WP2 double-exhaustion; driver-split 5-WP plan). PHYSICS
  (Professor R1/R2, correcting the brief's own premise): the low-w
  anchor is F(w→0) = sqrt(mu_macro)·exp(−iπn/2) — NOT 1, which holds
  only at gamma=kappa=0 (the shipped FARFIELD_DIFFRACTIVE docs carried
  the same bug, fixed); the parity wall is a branch point, so TWO
  rungs: Rung P (positive parity) = reduced-shear closed form
  C(w)·G_PM + shear-operator series, truncation certificate
  w_low = (gamma'/2)[sqrt(mu)·R_{M+1}/bar]^(1/(M+1)) with the
  lam·sqrt_mu normalization (the Inspector-caught over-certification:
  the omitted term lives in total-amplitude space |total| = lam·sqrt_mu;
  the sqrt_mu-normalized gate over-admitted at kappa>0, measured breach
  1.475e-4 > 1e-4 at kappa=0.3 — fixed with a root-find fallback
  pushing the served floor up instead of refusing); the w·ln(w) phase
  rides exactly in C(w), never certificate-bounded. Rung S (macro
  saddle): the series DIVERGES at every order (Fermat-moment
  divergence), so the band bottom is HOSTED by the exact 1D f_schwinger
  with the paired N/2N quadrature certificate — counted
  `diffractive_engine_hosted` (ENGINE side of the 7b ledger by
  construction; the campaign's saddle-bottom charts store the residual
  against the exact −1j·sqrt(mu_macro) anchor and retire it, see
  [[lensing_training_campaign]]). MEASURED (in-build 10k census, seed
  0, engine-free, /tmp/wp3_census_out.json archived in the record):
  diffractive_analytic 14.27%, diffractive_engine_hosted 14.93%,
  engine_residual 53.30% → 24.10%, all 1409 saddle_c3 splits covered
  (W_reach ≥ w_split; w_split p50 14.5 / p99 40.3 / max 51.6).
  Program-to-date: engine residual 72.25% → 24.10% in one day of
  analytic-rung work, zero charts trained. Gate history: functional
  8-way ALL GREEN; five fixture-staleness clusters hand-finished
  (probe-stub diffractive leg, near-cusp canary, ratio-layer forks
  incl. the timing-tier class, part0 allowlist with justification);
  [[lensing_force_direct_bypasses_analytic_intercepts]] filed from the
  triage.
