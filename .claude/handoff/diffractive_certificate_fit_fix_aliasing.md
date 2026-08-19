# Build: fix w_low_fit angular aliasing — never-over-serve off-grid

## Mission

The `diffractive_certificate_fit` build GATE-FAILED at Professor inference
review with a genuine bug: `w_low_fit` OVER-SERVES off-grid in eigenframe
angle by up to ~5x, violating the build's load-bearing never-over-serve
guarantee. This build fixes the root cause, re-bakes, re-derates, and adds
off-grid validation so the suite CANNOT be green with a silent interior
breach.

## Root cause (Professor, verified)

`_DIFFRACTIVE_FIT_N_HARM = _DEFAULT_MAX_ORDER = 16` harmonics `cos(4 k theta)`
(k=1..16) were fitted on only **8 eigenframe theta samples**
(`scripts/fit_diffractive_certificate.py`, thetas = 8 points on [0, 2pi)).
At those 8 angles every harmonic aliases to two patterns (even-k -> +1,
odd-k -> (-1)^m), so 16 angular DOFs are underdetermined and the least-squares
fit oscillates catastrophically between grid nodes. The de-rate
(D ~ 0.745 = 1/max-on-grid-overpred) cannot bound the ~5x off-grid
over-prediction.

MEASURED (Professor, at HEAD): gamma=0.2, r=0.9, beta=kappa=0, theta sweep
[0, pi/2]: w_low_fit oscillates 0.004..60 (the DD cap) while the engine-honest
ceiling stays 13.7-21.4; 8/33 off-grid probe angles over-serve, worst ~4.2x
(theta=pi/8: fit 60 vs true 14.4), ~5.0x at gamma=0.3. End-to-end at
(theta=0.6): w_low_fit=32.93 but the series breaches 1e-4 at w~20 and reaches
9e-2 at w=32 — a silent interior 1e-4 breach. Tightness also fails off-grid
(ratios down to 0.004-0.05).

## Fix direction (Professor, binding)

Either (a) reduce harmonics to k <= 4 (resolvable at 8 thetas) OR (b) re-bake
on >= 32 thetas. THEN re-derate and re-validate against an OFF-GRID over-serve
sweep (angles NOT on the calibration grid). The never-over-serve guarantee
must hold off-grid, not just on the calibration nodes.

Recommendation: do BOTH (a) and (b) if cheap — harmonics k<=4 AND a denser
theta grid (>=32) — so the angular basis is safely oversampled. If choosing
one, (a) is the structurally-safer fix (fewer DOFs than samples), but (b) is
what makes the fit's angular shape trustworthy. The completion record must
quote the off-grid validation numbers.

## Scope

IN:
- Fix the harmonic count: `_DIFFRACTIVE_FIT_N_HARM` must be <= 4 (or whatever
  is resolvable given the calibration theta count), decoupled from
  `_DEFAULT_MAX_ORDER` — the angular basis size is a FIT property, not a
  series property.
- Fix the calibration grid in `scripts/fit_diffractive_certificate.py`:
  >= 32 thetas (or match the harmonic count per the Nyquist argument: the
  k-th harmonic needs > 2k thetas to be resolvable; k<=4 needs >= 9, use
  >= 16-32 for margin). Re-run `--scale full` at HEAD, paste the VERIFIED
  re-baked coefficients + de-rate verbatim.
- Off-grid validation: add an OFF-GRID over-serve sweep to the oracle suite —
  probe angles strictly BETWEEN calibration grid nodes (e.g. the midpoints
  of the calibration theta spacing, plus random thetas) across gamma x radius.
  The suite must go RED if w_low_fit over-serves ANYWHERE on that off-grid
  set. This is the regression pin that would have caught INS-4-001.
- Re-derate against the off-grid worst case (not just the grid worst case),
  so the margin absorbs interpolation error between nodes. Conservative wins
  over tight: spend the margin on the hard never-over-serve guarantee.
- Keep the exact-zero over-serve tolerance and the tightness (>=0.5) pins;
  extend them to the off-grid set.

OUT (do not touch):
- The served series `diffractive_amplification` (exact order-16).
- Rung S / the macro-saddle engine-host.
- The refusal-gate semantics (wall via `_reduced_shear`, degenerate None).
- Any surrogate-chart or campaign work.

## Acceptance

- NEVER-OVER-SERVE OFF-GRID: on a held-out set of off-grid angles x gamma x
  radius, `w_low_fit <= w_low_true` at EXACT float64 (zero tolerance), with
  the engine-truth measured via `_measure_w_low_true` (sup-over-w vs
  `f_schwinger`). Quote the worst over-prediction ratio (must be <= 1.0).
- TIGHTNESS OFF-GRID: `w_low_fit >= 0.5 * w_low_true` on >= 90% of the
  off-grid set. Quote the median/p90 ratio.
- END-TO-END: at `w = w_low_fit`, |series - engine|/|engine| <= 1e-4 on the
  off-grid fixtures (no silent interior breach).
- The re-baked coefficients are pasted VERBATIM from `--scale full` at HEAD
  with the provenance SHA in the completion record.
- `_DIFFRACTIVE_FIT_N_HARM` is decoupled from `_DEFAULT_MAX_ORDER` and the
  calibration theta count is documented in the fit script docstring.
- Existing 39 mechanical + 34 diffractive tests stay green (minus any
  now-correctly-re-scoped ones), PLUS the new off-grid over-serve pin.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated. In-build tests FAST.
- The full re-bake (`--scale full`) is ~1-2 min — run it IN-BUILD (the
  previous build's deferral is what shipped the bad coefficients).
- Spec/TODO workflow: `[→ spec]`.
- Census mirror (`serve_route_census.py`) stays mirror-fidelity: binds the
  SAME `w_low_fit` predicate object, never a re-type.
