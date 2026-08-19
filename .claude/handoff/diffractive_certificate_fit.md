# Build: fit the diffractive certificate — instantaneous serve at the good (order-16) baseline

## Mission

Replace the per-proposal `diffractive_w_low` certificate SCAN with a smooth
FITTED function, so the hot-path serve is a lookup, not a search. The served
series value (order 16) is already cheap (~2ms/node); the expensive thing is
the certificate `w_low` (0.44s at order 8, 1.96s at order 16 for a whole-band
draw) which `_low_w_diffractive_serve` recomputes per proposal via a two-tier
scan. The fit makes `w_low` an O(1) evaluation.

BASELINE IS ORDER 16 (owner ruling, 2026-08-19): order 8 is NOT a principled
baseline — it over-certified (fixed), had the gamma=0.1 re-crossing, and
falsely refused gammas 0.4/0.5. The order scan showed the series is
CONVERGENT; order 16 extends reach (engine-verified, worst rel-err 1.000e-4
at ceiling). `_DEFAULT_MAX_ORDER = 16` is the fit's oracle.

## Measured structure (engine-free, order 16)

`w_low` is SMOOTH and near-power-law in the natural coordinates:
- `log(w_low)` vs `log(gamma)` at y=(0.8,0.4): monotone, nearly linear.
  67.08@0.05 -> 7.57@0.30 (slope ~ -1.2 in log-log over that range).
- `log(w_low)` vs `log(radius)` at gamma=0.2: monotone, smooth.
  29.05@r=0.3 -> 11.15@r=1.3.
- vs source ANGLE: smooth but non-monotone (oscillates ~4.2-5.2 at gamma=0.2,
  r=0.9) — the fit must capture this without over-fitting noise.
- Functional form hint: `w_low ~ const * (gamma'^a * s^b)` with a, b near
  power-law, times a smooth angle-dependent factor. VERIFY the exact
  exponents and the angle dependence; do not assume.

## Scope

IN:
- A fitted certificate: `w_low_fit(gamma, beta, kappa, y1, y2) -> float`,
  trained at order 16. Representation is the build's design decision
  (log-log polynomial / spline / small table with interpolation). The fit
  MUST be CONSERVATIVE: it must never return a `w_low` ABOVE the true
  honest order-16 ceiling (over-serving = serving a band with interior
  breach), and should be within a small factor BELOW (under-serving is safe
  but wasteful). A margin or lower-envelope construction is expected.
- Wiring: `_low_w_diffractive_serve` (and the two census mirrors that call
  `diffractive_bottom_ceiling`) use the fit in place of the scan. The fit
  must preserve the band semantics (w_lo/w_hi caps, whole-band return w_hi,
  floor-fail None) — those are cheap logic, not the expensive part.
- The served series `diffractive_amplification` stays EXACT (order 16) —
  the fit replaces ONLY the certificate boundary search, never the served
  value.
- Conservative validation: over a residual-demand sample spanning the
  gamma x y plane, `w_low_fit <= w_low_true` everywhere (or <= true with a
  documented, tolerance-level exception), and `w_low_fit >= c * w_low_true`
  for a well-chosen c (e.g., 0.5-0.9) on most of the sample — tight AND
  safe, never arbitrary.

OUT (do not touch):
- The served series / `diffractive_amplification` internals.
- The certificate's REFUSAL semantics (None at wall / deep-optimistic): the
  fit must reproduce the refusal region boundary, not erase it. The
  deep-optimistic gammas {0.4, 0.5} at order 16 now ADMIT (honest) — the
  fit must match the order-16 admit/refuse boundary, NOT the old order-8
  one.
- Any training of surrogate charts / campaign runs.

## Acceptance

- `diffractive_w_low` (exact, order 16) and `w_low_fit` agree: fit <= true
  on a held-out residual sample (no over-serve), fit >= 0.5*true on >=90%
  of it (not pathologically tight-averse). Report the achieved margin.
- Serve-time cost: `_low_w_diffractive_serve` certificate call drops from
  ~1.96s (scan) to ~microseconds (fit lookup). Measure and report both.
- Refusal boundary preserved: the fit reproduces the order-16 admit/refuse
  regions (gammas 0.4/0.5 admit; gamma>~0.35 wall refusals as the exact
  does).
- Band semantics intact: whole-band-certified returns w_hi, floor-fail
  returns None, byte-identical serve routing on the null cases.
- Values-not-paths: assert the SERVED ceiling is <= the exact honest
  ceiling on fixture points (against `_honest_tail_ratio` at order 16), not
  which branch produced it.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated. In-build tests FAST.
- Spec/TODO workflow applies: behavior change — `[→ spec]`.
- The fit data generation uses `diffractive_w_low` at order 16 over a
  modest grid (hundreds of points, minutes — NOT the 10k census). Do not
  hand-roll a parallel census.
- Keep `_DEFAULT_MAX_ORDER = 16`; the fit oracle is exactly that.
