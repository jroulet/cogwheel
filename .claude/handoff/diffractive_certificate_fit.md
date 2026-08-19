# Build: fit the diffractive certificate — replace the per-proposal scan

## Mission

Replace the per-proposal `diffractive_w_low` certificate SCAN with a smooth
FITTED function, so the hot-path serve is a lookup, not a search. The served
series value (order 16) is cheap (~2ms/node); the expensive thing is the
certificate `w_low` — 0.44s at order 8, 1.96s at order 16 for a whole-band
draw — which `_low_w_diffractive_serve` recomputes per proposal via a two-tier
running-max scan. The fit makes `w_low` an O(1) evaluation in the hot path.

## Baseline / oracle: THE EXACT ENGINE (owner ruling, 2026-08-19)

A fitted certificate frees the oracle choice — the expensive truth is paid
ONCE at fit-training time, never per proposal. Use the best available truth:
the EXACT Schwinger engine `_schwinger.f_schwinger`. Define the certificate as

    w_low(y, gamma, beta, kappa) = largest w such that
        |F_order16(w) - f_schwinger(w)| / |f_schwinger(w)| <= CERTIFICATION_BAR

measured DIRECTLY against the exact engine — no N/2N tail-ratio estimator
bias. The estimator exists only because the certificate had to be engine-free
per-proposal; a fitted lookup is a pure function at serve time, so the
engine-free census mirror stays valid. This supersedes the order-8-vs-16
question entirely: the fit encodes "how far is the served series honest
against the truth," not against a truncation proxy.

Served series stays order 16 (`_DEFAULT_MAX_ORDER = 16`) — the served value
is exact order-16; only the certificate boundary is fitted.

## Measured structure (engine-free, order 16 — smooth, fit-friendly)

`w_low` is SMOOTH and near-power-law in natural coordinates:
- `log(w_low)` vs `log(gamma)` at y=(0.8,0.4): monotone, near-linear.
  67.08@gamma=0.05 -> 7.57@gamma=0.30 (log-log slope ~ -1.2 over range).
- `log(w_low)` vs `log(s)` at gamma=0.2: monotone, smooth. 29.05@r=0.3 ->
  11.15@r=1.3 (log-log slope ~ -0.34 in s).
- vs source ANGLE: smooth but non-monotone (oscillates ~4.2-5.2 at
  gamma=0.2, r=0.9). The fit must capture this without over-fitting noise.

Hint: `w_low ~ const * (gamma')^a * s^b * f(angle)` with a, b near power-law
exponents and f smooth. VERIFY the exact exponents and angle dependence
empirically; do not assume a form.

## Scope

IN:
- A fitted certificate `w_low_fit(gamma, beta, kappa, y1, y2) -> float`,
  calibrated to the ENGINE-defined honest ceiling at order 16. Representation
  is the build's design decision (log-log polynomial / spline / small table
  with interpolation). MUST be CONSERVATIVE: never return a `w_low` ABOVE
  the true engine-honest ceiling (that would serve a band with an interior
  breach — the exact bug class this work fixes), and should be within a
  small factor BELOW (under-serving is safe but wasteful). A safety margin
  or lower-envelope construction is expected.
- Wiring: `_low_w_diffractive_serve` (and the two nested-split call sites of
  `_diffractive_bottom_ceiling`, plus the census mirror) use the fit in
  place of the scan. The fit must preserve the band semantics (w_lo/w_hi
  caps, whole-band -> w_hi, floor-fail -> None) — those are cheap logic,
  not the expensive part.
- The served series `diffractive_amplification` stays EXACT (order 16) —
  the fit replaces ONLY the certificate boundary search, never the served
  value.
- Validation: over a residual-demand sample spanning the gamma x y plane,
  `w_low_fit <= w_low_true` everywhere (conservative) AND `w_low_fit >= c *
  w_low_true` for a well-chosen c (e.g. 0.5-0.9) on most of the sample.
  Report the achieved margin distribution. `w_low_true` is measured against
  the exact engine (hundreds of points, ~1 min at 90ms/node).

OUT (do not touch):
- The served series / `diffractive_amplification` internals.
- The certificate REFUSAL semantics (None at the wall / deep-optimistic):
  the fit must reproduce the order-16 admit/refuse boundary, NOT the old
  order-8 one. Gammas 0.4/0.5 ADMIT at order 16 (honest, engine-verified);
  the gamma > ~0.35 positive-parity wall still refuses. Match that.
- Any surrogate-chart training or campaign work.

## Acceptance

- `w_low_fit <= w_low_true` (no over-serve) on a held-out residual sample;
  `w_low_fit >= 0.5 * w_low_true` on >= 90% of it. Report achieved margins.
- Serve-time cost: the certificate call in `_low_w_diffractive_serve` drops
  from ~1.96s (scan) to ~microseconds (fit lookup). Measure and report both.
- Refusal boundary preserved: gammas 0.4/0.5 admit with a positive ceiling;
  the wall refusals match the exact certificate.
- Band semantics intact: whole-band-certified returns w_hi, floor-fail
  returns None, null-split byte-identical.
- VALUES NOT PATHS: assert the served ceiling is <= the exact engine-honest
  ceiling on fixtures (against `f_schwinger` at order 16), not which branch
  produced it.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated. In-build tests FAST.
- Spec/TODO workflow applies: behavior change — `[→ spec]`.
- Fit data generation: a MODEST grid (hundreds of points) of
  engine-vs-order16 ceiling measurements, minutes — NOT the 10k census.
  Serial, progress-tracked is fine; do NOT hand-roll a parallel census.
- Keep `_DEFAULT_MAX_ORDER = 16`; the served series is exact at that order.
- The census mirror (`serve_route_census.py`) must keep mirror-fidelity:
  bind the SAME production fit predicate, never re-type it.
