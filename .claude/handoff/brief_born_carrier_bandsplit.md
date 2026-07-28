# Build brief — lead-only carrier + band-split residual charts (positive parity)

## Mission

Serve the far annulus `3.0 < |y| <= 4.2426` at positive parity WITHOUT
quadrature, for `gamma < 3/4`. The physics is derived and measured; this build
is implementation.

Read FINDINGS F023, F024 and F025 before starting, in that order. F024 and
F025 each CORRECT F023, and building from F023 alone will produce the wrong
thing — that has already happened twice at the planning stage.

## The one thing to internalise

The analytic term is a CARRIER. A trained chart interpolates the RESIDUAL
`F_exact - F_carrier`. The carrier does NOT have to hit an accuracy target on
its own — the criterion is HOW CHEAPLY THE RESIDUAL SPLINES. Same
decomposition as SACR-C (analytic image kernels + one interpolated envelope)
and the far-field label (kernel sum + interpolated remainder).

In production the exact engine is NEVER called, so there is no fall-through to
compare against. The module's old T1 target (`rel err < 1e-3` for the
standalone series) is not the bar and must not be reinstated.

## Measured facts (inline; do not re-derive, do not extrapolate)

Every number below carries its sweep. Do not apply any of them outside it.

- **The carrier is LEAD-ONLY**:

      F_carrier = sqrt(mu_macro) * exp(1j*w*phi_geo)

  Do NOT put `a0` or `b1` in the serving path. Measured over
  `gamma in {0.45 ... 0.75}`, `|y| in {3.05, 4.2426}`,
  `theta in {0.3, 0.9, 1.35}`: `N(F) == N(lead-only residual)` EXACTLY at all
  18 sampled combinations, while the `(a0,b1)` carrier costs 11 nodes at
  gamma=0.45, 20 at 0.60, 44 at 0.75 — a 2.5x to 11x penalty.
  REASON: `a0` violates F009. `F(w->0) = sqrt(mu_macro)` exactly, but
  `1 + a0/q2r` adds a constant offset below the split, where `a0` (a
  resolved-image correction valid only for `w*Delta_tau >> 1`) does not apply.
  `b1` alone is harmless but buys nothing.
- **`a0` and `b1` still belong in the module** as correct physics and the
  macro-limit diagnostic. Fix the placeholder (`b1 = 1.0` has the WRONG SIGN;
  a pure point mass gives `-1`) and add the missing `a0`, both from F023:

      b1 = -lam * (2.0*lam*r0_sq - x0_dot_y) / (det_a * r0_sq)
      a0 = -lam * (lam*r0_sq - x0_dot_y) / (det_a * r0_sq)

  They are then used by `born_amplification` for the resolved-image regime and
  by tests. They are NOT used by the serve path.
- **Band split at `w * Delta_tau ~ 4`**, with `Delta_tau` the Fermat-delay
  difference of the two real images (from the partition via `geometry.delay`,
  no extra cost). Holds across the whole swept gamma range. It coincides with
  SACR-C's own switch scale `RHO_END = 4`.
  Do NOT key it on `w * r0_sq`: those agree only where `Delta_tau ~ r0_sq/2`,
  a positive-parity coincidence. On the saddle `r0_sq/(2*Delta_tau)` spans
  0.16 to 35.6 (F024). Use the invariant now so the saddle build adds a
  branch, not a re-key.
  * `w * Delta_tau <  4`: lead-only carrier. No second image, no ppGO, no ghost.
  * `w * Delta_tau >= 4`: `geometric_amplification` over BOTH real images at
    FULL C1/C2 kernels, plus `farfield_ghost_term` where admitted.
- **DO NOT mix the bands.** ppGO below `w = 0.05` inflates the residual by
  five orders of magnitude through its `1/w**2` kernel.
- **Residual node ceilings**, `gamma <= 3/4`, eps 4e-3 absolute of `max|F|`:
  5 on `[1e-3, 0.05]`, **31** on `[0.05, 0.5]`, **27** on `[0.5, 8]` for
  `log_w`; 4 and 14 per y-axis in the two sub-split bands. (F023 quoted 4-15
  and 4-8; those were floors measured at gamma <= 0.45 and radially only.)
- **The exterior fence is exact**:

      max |y| on the astroid = 2*gamma / sqrt(1 - gamma)          (kappa = 0)
                             = sqrt(lam) * 2*gp / sqrt(1 - gp)    (gp = gamma/lam)

  The annulus inner edge `|y| = 3.0` is breached at `gamma = 3/4` EXACTLY.
  SERVE ONLY `gamma < 3/4`; refuse by name above it. Above 3/4 the annulus
  straddles (to 0.8423291) then lies inside the caustic — a different geometry
  with fold crossings in the tile, and an interior problem, not this rung's.
- The chart absorbs `ln(w/2)` on its existing `log_w` axis at zero node cost
  (F023), so there is no low-`w` analytic rung.
- Two distinct objects, count each ONCE: the faint near-lens SECOND REAL IMAGE
  (`find_images`, Morse index 1) is NOT the COMPLEX saddle ghost
  (`farfield_ghost_term`, gated on geometric separation >= 0.7).
- `ghost_kernel` raises `GhostDomainError` at
  `(|y|=3.6, theta=0.5, gamma=0.25, kappa=0.3, beta=0.5)` while `find_images`
  returns 2 real images. The complex ghost is NOT universally available; the
  serve path must tolerate its absence (it already does).

## In scope

- `_born_factors`: fix `b1`'s sign, add `a0`. Single edit site.
- `born_amplification` / `born_envelope`: `+ a0/q2r` in `correction` (the
  resolved-image form), and a LEAD-ONLY entry point for the serve path.
- The band-split serve: lead-only below, ppGO + ghost above, keyed on a NAMED
  constant in `w * Delta_tau`.
- The `gamma < 3/4` fence as a named refusal above it.
- Residual charts for both bands through the EXISTING chart machinery and eps
  gate. Do not invent a parallel one.
- `born_gate` guard A: re-derive. It currently estimates an
  `O(w**2/q2r**2)` term that is far smaller than what actually limits the
  rung; re-key it to the band-split criterion.
- The `'born'` category in `surrogate_census.classify_fallthrough`, planned in
  the original build and never landed — annulus draws are currently
  mis-attributed to `out-of-box`.
- Correct the `_born.py` docstring: its WHY premise is BACKWARDS (F023), and
  "low-frequency far zone" mislabels a MID-`w` resolved-image expansion.

## Out of scope

- The macro-saddle branch (`gamma > 1`). Derived (F024) but a separate build.
  Its carrier is also lead-only, so keep the split and carrier parameterised
  and adding it should be a branch, not a rewrite.
- `3/4 <= gamma`. Refuse by name.
- The low-`w` analytic rung (measured unnecessary).
- Both carrier-continuity guards (F022). Not implicated.
- Cusp exclusion balls; dropped gamma slivers. Separate holes, separate builds.
- Any census RUN.

## Acceptance (build-level)

1. `b1` and `a0` match the closed forms, checked against an INDEPENDENT
   reconstruction (e.g. the matrix form `-lam * x0^T A^-1 x0 / |x0|**2`), not
   a copy of the same expression. A pure point mass gives `b1 = -1` exactly.
2. The SERVE path uses lead-only below the split. A reachable-red shows that
   including `a0` there inflates the azimuthal node count (the measured
   signature: N goes 4 -> 11 at gamma=0.45).
3. `F(w->0)` from the served path equals `sqrt(mu_macro)` to machine precision
   (F009). This is what `a0` broke; pin it.
4. The split criterion is a named constant in `w * Delta_tau`, and a
   reachable-red shows keying it on `w * r0_sq` mis-splits where
   `Delta_tau != r0_sq/2`.
5. Residual node counts are within a factor ~2 of the ceilings above, measured
   AZIMUTHALLY as well as radially. A radial-only sweep is what hid the `a0`
   pathology for two rounds; do not repeat it.
6. `gamma >= 3/4` in the annulus refuses by name; `gamma < 3/4` serves.
7. A source where `ghost_kernel` raises still serves, and is not a refusal.
8. `classify_fallthrough` attributes annulus draws to `'born'`, with a
   reachable-red proving the old attribution would fail.
9. Positive-parity paths that do NOT touch the annulus are byte-identical.
   State the driver-run recipe (config sweep, which outputs and npz byte
   streams to diff against the pre-build tree). Do NOT write a committed test
   that imports a module from a git revision — retired in 8901b0b (F022); its
   premise expires the moment the build commits.
10. Full fast suite green, driver-verified post-build.

## Constraints

- Branch `claude-dev` only. Never commit on main/master.
- Slow tests NEVER run in-build; `COGWHEEL_BRUTE_ACCURACY` /
  `COGWHEEL_STRICT_TIMING` / `COGWHEEL_TRAIN_TIER` stay unset in agent envs.
  In-build tests must be FAST — small synthetic configs, few-eval oracles.
- Units and conventions per AGENTS.md; `_born.py` is a pure float64 scalar
  path — keep it so, and do NOT add `fastmath` (the phase must stay
  reproducible).
- Verify existing tests for backward compatibility BY READING, including
  gated ones. This build CHANGES NUMERICAL VALUES in `_born.py` (the sign
  fix), so existing Born tests WILL move. A test pinning the placeholder's
  output is asserting a known-wrong value — fix it, do not preserve it.
- The pre-commit drift hook blocks on gated tests referencing changed APIs and
  a build CANNOT satisfy it (it needs tier runs the driver owns). If you hit
  it, report the flagged list in your change report and STOP — do not
  `--no-verify`. The driver runs the tiers and acknowledges.
