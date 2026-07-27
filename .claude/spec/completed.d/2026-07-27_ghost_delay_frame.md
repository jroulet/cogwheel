---
date: 2026-07-27
section: lensing
---

# Ghost complex-saddle delay-frame repair (build 8h-b7)

`channels.farfield_ghost_term` now carries the decaying ghost in the
partition's min-subtracted frame, `C_c(w) * exp(1j*w*(tau_c - t_min))`, via a
new module-private `_frame_t_min(source, matrix)`. `geometry.ghost_kernel` and
`geometry._ghost_delay` are unchanged and byte-identical; the ghost gate still
reads the frame-invariant raw `Im tau_c`.

Tests: `cogwheel/tests/test_lensing_chang_refsdal_ghost_frame.py` (12 tests,
5.9 s) pins the frame at machine precision and carries three self-falsification
cases; `GhostFrameCollapseTestCase` and `RealImagePathBitIdentityTestCase` in
`cogwheel/tests/test_lensing_exterior_windows.py` pin the residual collapse
(measured 174x / 31x / 607x) and the untouched real-image primitives.

## Known follow-ups (measured 2026-07-27, NOT addressed here)

- The `w_min` ghost gate admits only `w >= (RHO_END/2)/Im tau_c`, which is
  above the band where the ghost is most valuable. Driver measurements with an
  ungated, frame-corrected ghost show 5x-155x accuracy gains BELOW the gate,
  down to `w = 0.5`. Removing the discrete gate in favour of always including
  the decaying member (the exponential suppresses it naturally) is the next
  build. Note the term has its own high-`w` floor: at `gamma = 0.90, w = 40.9`
  it degrades a 1.9e-15 residual to 5.2e-7 -- harmless, but real.
- Near the astroid cusps the ghost cannot help at all: three saddles coalesce
  (1 real + the conjugate pair; `min|real - ghost|` falls 1.33 -> 0.24 as
  `theta -> 0`) and `Im tau_c -> 0.001`, so the per-saddle expansion is invalid
  regardless of framing. Whether a principled multi-saddle sum beats deferring
  to a uniform Pearcey arm is OPEN and untested.
- The frame fix is parity-agnostic and was confirmed in the macro-saddle
  regime (59x at `gamma = 1.5, theta = 20 deg`; 107x at `gamma = 1.2`), but a
  wedge-aware probe family is needed before any `gamma > 1` coverage claim.
  The deltoid lobes lie in wedges `|sin 2 theta| <= 1/gamma` about the
  negative-eigenvalue axis; radial placement from a scalar reach lands
  sources between the lobes, where `GhostDomainError` is correct behaviour.
- `t_min` is now a shared convention across several sites. Per-site
  recomputation is a latent frame-skew bug of exactly this class; the durable
  safeguard is the single shared helper plus an assertion that consumers agree
  on `t_min` for a fixed `(source, matrix)`.
