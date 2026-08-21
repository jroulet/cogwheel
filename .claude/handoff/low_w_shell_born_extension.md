# Build: low-w serve end state — macro-lead chart for the shell + Born extension to rho ~ 1.4

## Mission

Implements the corrected rho-split ruling (`low_w_chart_rho_split_ruling.md`):
the bespoke wall-band chart is KILLED; the low-w band is served by the settled
demodulated-DIFFERENCE representation (never a quotient). Two pieces:

1. LOW-W NEAR-FOLD SHELL (rho in [0.6, ~1.4], w*Delta_tau < 1): a MACRO-LEAD
   demodulated-difference chart. At low w, F -> sqrt(mu_macro) regardless of
   rho (F009), so the residual r = f_pure - sqrt(mu_macro)*exp(i w phi_geo)
   (demodulated difference) is smooth and O(1). MEASURED: |r| ~ 0.61-1.6,
   arg within ~0.2 rad across w in [0.02, 1.0] at the shell witnesses incl.
   the b3->0 cell (gp=0.8 rho=1.1). NO engine at serve; Schwinger is the
   offline oracle ONLY.
2. INNER FAR-EXTERIOR (rho in [~1.4, 2]): EXTEND the shipped Born chart
   downward -- re-train born_residual_chart.npz with rho_grid down to ~1.4
   and log_w_grid down to the low-w band. The Born rung's representation
   (demodulated difference) is the settled, certified machinery; extending
   its grid is a node-budget change, not a new representation.

## The representation (binding — the lesson of three failed bakes)

NEVER divide by an oscillatory carrier. The settled representation is the
DEMODULATED DIFFERENCE, exactly as BornResidualChart does:
    R(w) = F_exact_demod(w) - F_carrier_demod(w)
with the shared carrier phase subtracted from both sides (demodulated by
e^{-i w phi_geo} or the frame phase). A difference has no poles -- the
carrier's beating zeros cancel identically because F_exact carries the same
beat. The served value is carrier + residual (interpolated R, re-modulated).

The bespoke quotient chart (r = f_pure*sqrt(1-gamma'^2)/F_ref) is DEAD: it
produced 5800x poles (three separate representation attempts). Do NOT
resurrect any quotient form.

## Scope

IN:
- The LOW-W SHELL CHART: modeled on BornResidualChart's structure (frozen
  dataclass, npz, content-hash, schema, covers()), but with the macro-lead
  carrier and a low-w grid:
  - carrier = born_lead_carrier = sqrt(mu_macro)*exp(i w phi_geo) (import
    from _born; do NOT re-implement)
  - residual R = f_pure_demod - carrier_demod (difference, demodulated by
    the same phase)
  - grid over (gamma', rho in [0.6, ~1.4], theta, log w in [log 0.02, log 1])
    (the low-w shell: w*Delta_tau < 1, the smooth regime)
  - serve: F = carrier + interpolated R, in the FARFIELD_DIFFRACTIVE gauge,
    re-modulated with the mass-sheet phase as the diffractive rung does
  - Schwinger = offline oracle ONLY (training); no f_schwinger at serve
  - a rho/w gate: cells where w*Delta_tau >= 1 (fold/cusp developing) decline
    to the fold arm / engine -- the chart owns only the smooth low-w shell
- BORN EXTENSION: re-train born_residual_chart.npz with rho_grid down to
  ~1.4 (from 2.0) and log_w_grid down to the low-w band. Lower the Born rung's
  rho gate to ~1.4. Re-measure the residual's rho-dependence (the gate is a
  node-budget boundary, not a physics law -- Born is the weak-deflection
  expansion valid throughout rho > 1, degrading gradually toward rho -> 1).
- Serving order: low-w shell chart FIRST (rho in [0.6, ~1.4], low w), then
  Born (rho >= 1.4), then the exact engine. The rho/w partition is the DRY
  single source shared by trainer, serve, and census mirror.
- Tests: per-region served-accuracy pins (shell vs inner far-exterior vs far
  exterior) |F_serve - F_engine|/|F_engine| <= 1e-4 at served w; the
  demodulated-difference representation has NO poles (assert finite residual
  on the grid); the rho/w gate continuity (no step at the shell/far-exterior
  boundary); Schwinger-stays-oracle (no f_schwinger on the serve path).

OUT (do not touch):
- The order-16 series, `w_low_fit`, the fence, Rung S.
- The fold arm / tube chart domain (larger w*Delta_tau) -- those are the
  EXISTING serving arms, not this build's scope.
- The existing Born rung's settled machinery (extend its grid, don't
  re-implement).
- Any quotient residual form.

## Acceptance

- The low-w shell is served by the macro-lead demodulated-difference chart at
  |F_serve - F_engine|/|F_engine| <= 1e-4 (measured against f_schwinger
  oracle) with NO residual poles (the quotient's failure mode is gone).
- The Born extension covers rho in [~1.4, 2] at the same bar; the far
  exterior (rho > 2) stays on the shipped Born chart.
- The rho/w partition is DRY and the census mirror counts chart-served draws
  as analytic (not engine demand) -- the honest post-serve demand map.
- Smoke-scale calibration in-build (< ~10 min, PROVISIONAL); full bake +
  shipped npz(s) + held-out validation = DRIVER step.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated.
- Spec/TODO workflow: `[→ spec]` + DATA_CONTRACTS update + completion record.
- `lensing_low_w_near_fold_serve` is superseded by this build; close it.
- The full bake + held-out validation is a DRIVER step.
- Schwinger is the offline oracle ONLY, never a serve-time call.
