# Build brief — Born carrier + band-split residual charts (positive parity)

## Mission

Wire the Born rung so the far annulus `3.0 < |y| <= 4.2426` is served WITHOUT
quadrature at positive parity. The physics is derived and measured (FINDINGS
F023); this build is implementation.

Read F023 before starting. The rung's SHAPE is not what the module currently
says it is, and getting that wrong is the main way this build fails.

## The one thing to internalise

The analytic term is a CARRIER. A trained chart interpolates the RESIDUAL
`F_exact - F_carrier`. The carrier does NOT have to hit an accuracy target on
its own — the criterion is HOW CHEAPLY THE RESIDUAL SPLINES. This is the same
decomposition used throughout the package (SACR-C: analytic image kernels plus
one interpolated envelope; far-field: kernel sum plus interpolated remainder;
band-split: ppGO above `w_trust`, chart below).

The module's existing T1 target of `rel err < 1e-3` for the standalone series
is NOT the bar and must not be reinstated. In production the exact engine is
never called, so there is no fall-through to compare against.

## Measured facts (inline; do not re-derive)

- Coefficients, both collapsing onto quantities `_born_factors` ALREADY
  computes (`r0_sq`, `x0_dot_y`) — no new geometry, no fifth convention site:

      b1 = -lam * (2.0*lam*r0_sq - x0_dot_y) / (det_a * r0_sq)
      a0 = -lam * (lam*r0_sq - x0_dot_y) / (det_a * r0_sq)

  `b1 = 1.0` was a placeholder with the WRONG SIGN (a pure point mass gives
  `-1`). `a0` is a real, `w`-independent term at the SAME order `1/q2r` that
  the current series omits entirely. Series becomes
  `1 + a0/q2r + 1j*(w/2)*b1/q2r`.
- Band split at `w_split`, keyed on `w * r0_sq <~ 8` (i.e. `w ~ 0.5`):
  * `w <  w_split`: carrier ALONE. No second image, no ppGO, no complex ghost.
    Residual 2.4e-2 - 8.7e-2 of `max|F|`; 4-15 nodes on `log_w`, 4 per y-axis.
  * `w >= w_split`: `geometric_amplification` with BOTH real images at FULL
    C1/C2 kernels, plus `farfield_ghost_term` where admitted. Residual
    1.6e-3 - 2.5e-2; 4-8 nodes on `log_w`, 4 per y-axis.
- DO NOT mix the bands. Adding ppGO below `w = 0.05` inflates the residual by
  FIVE ORDERS OF MAGNITUDE through its `1/w**2` kernel.
- The chart absorbs `ln(w/2)` on its existing `log_w` axis at ZERO node cost,
  so there is NO low-`w` analytic rung. Node counts are identical with and
  without the log term at every tolerance from 4e-3 to 1e-5.
- Two distinct objects, count each ONCE: the faint near-lens SECOND REAL IMAGE
  (`find_images`, Morse index 1, worth 4.4e-2 - 8.4e-2) is NOT the COMPLEX
  saddle ghost (`farfield_ghost_term`, conjugate quartic pair, gated on
  geometric separation >= 0.7). Both are real; conflating them double-counts
  one and drops the other.
- `ghost_kernel` raises `GhostDomainError` at
  `(|y|=3.6, theta=0.5, gamma=0.25, kappa=0.3, beta=0.5)` while `find_images`
  returns 2 real images. The complex ghost is NOT universally available in the
  annulus; the serve path must tolerate its absence (it already does —
  `GhostDomainError` is a `LensDomainError`).

## In scope

- `_born_factors`: replace the placeholder with the two derived coefficients.
  Single edit site.
- `born_amplification` / `born_envelope`: add `+ a0/q2r` to `correction`.
- The band-split serve: carrier below `w_split`, ppGO + ghost above, with
  `w_split` a NAMED, parameterised constant keyed on `w * r0_sq` — not a
  hardcoded 0.5. The saddle branch may need a different value (its derivation
  is in flight), so the criterion must be settable without restructuring.
- Residual charts for both bands, trained through the existing chart
  machinery. Reuse the existing eps gate and held-out machinery; do NOT invent
  a parallel one.
- `born_gate` guard A: re-derive. Its estimate rescales by `b1**2` (3.3x at
  `gamma' = 0.45`, ~4e4x at the guard-B edge since `|b1| <= 1/(1-gamma')`),
  AND it should be re-keyed to the band-split criterion `w * r0_sq` rather
  than the `O(w**2/q2r**2)` term, which is far smaller than the two terms it
  ought to be catching.
- The `'born'` category in `surrogate_census.classify_fallthrough`, which the
  original build planned but never landed — annulus draws are currently
  mis-attributed to `out-of-box`.
- Correct the `_born.py` module docstring: its WHY premise is BACKWARDS
  (measurement in F023), and "low-frequency far zone" mislabels a MID-`w`
  resolved-image expansion valid for `1/q2r**2 <~ w << q2r`.

## Out of scope

- The macro-saddle branch (`gamma > 1`). Its derivation is in flight; it is
  the NEXT build. Do not speculate it into this one — but do not hardcode
  anything that would make adding it a restructure either.
- The low-`w` analytic rung. Measured unnecessary; building it adds a fifth
  ladder component for no gain.
- Both carrier-continuity guards (F022). Different observables, not implicated.
- Cusp exclusion balls. Separate open hole, separate build.
- Any census RUN. The census is the last step of the programme.

## Acceptance (build-level)

1. `b1` and `a0` match the closed forms above to machine precision, checked
   against an INDEPENDENT reconstruction of each (e.g. the matrix form
   `-lam * x0^T A^-1 x0 / |x0|**2`), not against a copy of the same expression.
2. A pure point mass (`gamma = 0`, `kappa = 0`) gives `b1 = -1` exactly.
3. Residual node counts are within a factor ~2 of the measured table above, in
   both bands, on a small synthetic annulus config. This is the real gate: it
   is what "the residual splines cheaply" means operationally.
4. The band-split criterion is a named constant, and crossing it is continuous
   in the served `F` to the chart's own eps bar — no seam discontinuity.
5. A source where `ghost_kernel` raises still serves (carrier or ppGO without
   the ghost term), and is not a refusal.
6. `classify_fallthrough` attributes annulus draws to `'born'`, not
   `out-of-box`, with a reachable-red proving the old attribution would fail.
7. Positive-parity paths that do NOT touch the annulus are byte-identical.
   Prove it: state the driver-run recipe (config sweep + which outputs and npz
   byte streams to diff against the pre-build tree). Do NOT write a committed
   test that imports a module from a git revision — that apparatus was retired
   in 8901b0b (F022); the premise expires the moment the build commits.
8. Full fast suite green, driver-verified post-build.

## Constraints

- Branch `claude-dev` only. Never commit on main/master.
- Slow tests NEVER run in-build; `COGWHEEL_BRUTE_ACCURACY` /
  `COGWHEEL_STRICT_TIMING` / `COGWHEEL_TRAIN_TIER` stay unset in agent envs.
  In-build tests must be FAST — small synthetic configs, few-eval oracles.
- Units and conventions per AGENTS.md; numba-compatible hot paths; `_born.py`
  is a pure float64 scalar path — keep it so, and do NOT add `fastmath` (the
  phase must stay reproducible).
- Verify existing tests for backward compatibility BY READING, including
  skipped/gated ones. NOTE: this build changes numerical VALUES in `_born.py`
  (the sign fix), so the existing Born tests WILL move. Any test that pinned
  the placeholder's output is asserting a known-wrong value — say so and fix
  it, do not preserve it.
- The pre-commit drift hook blocks on gated tests referencing changed APIs and
  a build CANNOT satisfy it (it needs tier runs the driver owns). If you hit
  it, report the flagged list in your change report and stop — do not
  `--no-verify`. The driver will run the tiers and acknowledge.
