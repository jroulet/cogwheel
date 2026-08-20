# Build: low-w near-fold / wall-band chart serve (Rung P low-w residual chart)

## Mission

The near-fold shell (rho in [RHO_LO~0.6, 1+DELTA~1.4]) and the wall-approach
band (gamma' > ~0.5) of the low-w diffractive rung have NO analytic serve:
the order-16 shear-operator series has a convergence-radius collapse there
(measured: 40% error at M=16, 10% even at M=64 at gamma'=0.98, w->0), the
uniform Airy fold arm refuses at low w (its xi control is a large-w
asymptotic), and `w_low_fit` declines or serves the clipped ceiling 60. These
draws currently fall to the exact engine. This build trains a CHART serve for
that band — Schwinger as OFFLINE ORACLE ONLY, never a serve-time call, never
a decline-to-engine.

The tracked todo (`lensing_low_w_near_fold_serve`) is the binding spec; this
brief is its executable form.

## Binding representation + coordinates (owner, "as always")

Follow the codebase's settled representation doctrine, NOT a raw serve:

- REPRESENTATION = residual against the KNOWN ANALYTIC ANCHOR
  F(w->0) = sqrt(mu_macro)*exp(-i*pi*n/2) (n=0 positive parity), exactly the
  pattern of tube `r = E/F_ref` and saddle-bottom residual vs the exact
  -1j*sqrt(mu_macro).  The chart stores `r(w) = F_engine(w) / [sqrt(mu_macro)
  * exp(phase)]` — the smooth residual.  The known scaling
  `sqrt(mu_macro) = 1/sqrt(1 - gamma'^2)` is factored OUT analytically (it is
  the wall divergence); the fitted residual is smooth and low-dimensional.
  MEASURED: the series' deficit C(gamma') = |F(0)|/sqrt(mu_macro) is a smooth
  1D function (1.0001 at gamma'=0.5 -> 0.60 at 0.98), confirming the residual
  representation is the right one.
- COORDINATES = reduced/caustic-relative, never raw lens-plane:
  rho = |y'|/|y_c(theta)| (geometry.caustic_point — the SAME discriminator as
  the fence), 1 - gamma' (wall-collapse scale), eigenframe theta (even-
  harmonic cos(2k theta) basis), and w (or log w).  Same doctrine as the
  low-w series' relative-error currency (lam*sqrt_mu normalization).
- The chart's served value re-modulates: F_serve(w) = r_fit(w) * [sqrt(mu)
  * exp(phase)] — the anchor times the fitted residual (mirror of the tube
  beat-free r*F_ref serve).

## Scope

IN:
- A new chart artifact + training script, modeled on `BornResidualChart`
  (npz, content-hash-verified, grids in the reduced coordinates above).
  Class name e.g. `LowWDiffractiveChart` in `cogwheel/lensing/
  low_w_diffractive_chart.py` (or alongside the born chart); trained grids
  over (gamma', rho, theta, log_w) on the near-fold + wall band.
- The training oracle is `_schwinger.f_schwinger` (the exact engine),
  evaluated at the reduced parameters (mass-sheet map as `_engine_reference`
  does) — offline only.  The residual target is `F_engine / [sqrt(mu)*phase]`.
- Serve wiring: the diffractive low-w serve (`_low_w_diffractive_serve` /
  `_diffractive_bottom_ceiling` path) consults the chart for the band the fit
  declines (rho in the near-fold shell, or gamma' beyond the fit's calibrated
  range).  Serve = re-modulated residual, conservative (the chart is trained
  on the oracle; de-rate or lower-envelope as the tube/born charts do).
  NEVER a Schwinger call at serve.
- The census mirror (`serve_route_census.py`) classifies chart-served draws
  as the new route (NOT engine demand) — the honest post-serve demand map.
- Conservative validation: held-out off-grid (theta midpoints) x near-fold x
  wall fixtures, |F_serve - F_engine|/|F_engine| <= CERTIFICATION_BAR at the
  served w, never over-serve, de-rate the sole margin.

OUT (do not touch):
- The order-16 series, `w_low_fit`, the near-fold fence, `diffractive_
  amplification` internals.
- Rung S / macro-saddle engine-host.
- Any surrogate-chart or campaign work.

## Acceptance

- Chart-served draws agree with the exact engine to <= 1e-4 on held-out
  near-fold + wall fixtures (values, not paths).  The serve-route census no
  longer counts the chart-served band as engine demand.
- The representation is the residual-vs-sqrt(mu_macro)-anchor in the reduced
  coordinates — the raw amplification is never stored/fitted directly.
- Schwinger is offline-oracle-only: `grep` shows no f_schwinger call in the
  serve path (only the training script).
- The chart is a data product with a data contract (DATA_CONTRACTS.yaml
  update), content-hash-verified load, and a regeneration script.
- In-build tests FAST (engine-free serve tests + a small oracle-pinned
  fixture set); the full oracle calibration is a DRIVER step (bulk sweep).

## Constraints

- Branch `claude-dev`. Slow tiers stay gated.
- Spec/TODO workflow: `[→ spec]` + DATA_CONTRACTS update + completion record.
- The tracked todo `lensing_low_w_near_fold_serve` is binding; do not deviate
  from the representation/coordinates spec above.
- Full calibration = DRIVER step (the SDK 1200s ceiling kills in-build bulk
  sweeps; a smoke-scale calibration proves the pipeline in-build).
- After this build: the tiling-plan refresh and campaign sizing can proceed
  on the honest post-serve demand map.
