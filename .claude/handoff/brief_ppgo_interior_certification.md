# Build Brief: Certify ppGO for Interior Cells at High Frequency

## Mission

The ppGO certification map returns UNKNOWN for interior cells (rho < 1)
because the exact engine hits the Schwinger wall (w ~ 60) before the ppGO
error drops below the 1e-4 bar. But ppGO IS physically exact at high w
(geometric optics limit) — the certification just can't VERIFY it because
the exact reference breaks down.

This means band-split serving can't work for interior points, forcing them
onto charts that must cover the full w-band — a major coverage gap.

## The problem precisely

- At interior points (4 real images, close together), ppGO error is ~1e0 at
  w=1, dropping with w but still above 1e-4 at w=60 (the Schwinger wall)
- Above w=60, the exact engine refuses (double-double precision exhausted)
- ppGO is physically correct at high w (geometric optics limit) but we
  can't numerically verify it

## Possible approaches (for the Architect/Professor to evaluate)

1. **Fix the metric**: maybe `|exact - ppgo| / max|exact|` is suboptimal.
   Per-image or per-channel metrics might show ppGO is already accurate
   enough with a better normalization.

2. **Use GLow**: the repo has `glow` (a high-frequency exact evaluator for
   positive parity) that works past the Schwinger wall. Use it as the exact
   reference for interior certification.

3. **Extrapolate the error scaling**: below the wall, the ppGO error scales
   as a known power law in 1/w. Measure that scaling in the certified range
   and extrapolate to certify higher frequencies.

4. **Trust ppGO above the wall on physics grounds**: if images are
   well-separated in delay (w * Delta_tau >> 2*pi), ppGO is exact. Add a
   frequency-independent geometric check (delay separation > threshold)
   instead of a measured bar.

## In scope

- Determine why interior cells fail certification
- Implement a fix that allows the ppGO map to certify interior cells
- Verify that band-split then works for interior draws in the census

## Out of scope

- The surrogate training itself (running in background)
- Changing the surrogate chart architecture

## Constraints

- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
