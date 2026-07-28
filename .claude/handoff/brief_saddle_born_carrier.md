# Build brief — saddle Born carrier + band split (macro saddle, gamma > 1)

## Mission

Serve the far annulus `3.0 < |y| <= 4.2426` on the MACRO-SADDLE branch without
quadrature, for `1.0502342 < gamma < 3`. The physics is derived and measured;
this build is implementation.

It is the twin of the positive-parity build that shipped in `31ee133`. Read
that code first — `_born.py`'s `born_lead_carrier`, `channels.
born_carrier_from_partition`, and the `'born'` census category — and follow its
shape. Most of this build is adding a branch, not inventing a mechanism.

Read FINDINGS F024 (saddle physics) and F026 (the exact fence) before starting.
F023 and F025 are positive-parity history; F026 CORRECTS F024's measured extent
table, so do not use that table.

## The one thing to internalise

Same as positive parity: the analytic term is a CARRIER, and a trained chart
interpolates the RESIDUAL. The carrier does not have to hit an accuracy target
alone — the criterion is HOW CHEAPLY THE RESIDUAL SPLINES. The likelihood
serve slot stays UNWIRED here too: the residual chart is a driver TRAIN_TIER
artifact this build does not produce.

## Measured facts (inline; every number carries its sweep)

- **The carrier is LEAD-ONLY**, with the Morse-phase origin:

      F_carrier = sqrt(|mu_macro|) * exp(-1j*pi/2) * exp(1j*w*phi_geo)

  Measured `gamma in [1.05, 1.6]`, `|y| in {3.05, 4.2426}`, several angles,
  band `[1e-3, 0.05]`: lead-only gives residual 1.0e-2 - 7.4e-2 at N=4 on
  `log_w` AND N=4 per y-axis, uniformly. The full `(a0,b1)` carrier gives
  1.7e-2 - 1.42 and needs N = 23-65 in the y-plane — `|a0|,|b1| ~
  1/(gamma'-1)` blow up near the wall and inject theta-structure that is not
  in `F`. Same verdict as positive parity, arrived at by a different
  mechanism.
- **The Morse phase cancels in the derivation but NOT in the value.** The
  Fresnel prefactor is `(2*pi*i/w)*|det A|^(-1/2)*exp(-i*pi*n/2)` for any
  non-degenerate real symmetric `A`, and `<u_i u_j> = (i/w)(A^-1)_ij` follows
  from differentiating it — so the Morse phase is an overall constant that
  cancels in the ratio `J/J0` (which is why `a0`/`b1` carry over unchanged
  with `det_a < 0` flipping both signs). It does NOT cancel in the carrier
  itself: the origin is `sqrt(|mu_macro|) * exp(-1j*pi/2)`, not a real
  amplitude.
- **Band split at `w * Delta_tau ~ 4`**, `Delta_tau` = Fermat-delay difference
  of the two real images, read from the partition. SAME criterion and SAME
  named constant (`RHO_END`) as positive parity — this is why the positive
  build was required to key on `w * Delta_tau` rather than `w * r0_sq`.
  Confirmed both directions on the saddle: at `gamma=1.6, |y|=4.24,
  theta=0.9`, `Delta_tau = 0.294` so `w_split = 13.6` and the carrier still
  works at `w = 8` (N=7); at `gamma=1.2, theta=0.3`, `Delta_tau = 35.3` so
  `w_split = 0.113` and the carrier has already failed by `w = 0.5` (N=161).
  `w * r0_sq` mispredicts both by two orders of magnitude
  (`r0_sq/(2*Delta_tau)` spans 0.16 to 35.6 here).
- **Above the split: ppGO over both real images at full C1/C2, and REFUSE the
  complex ghost.** Measured `gamma=1.6, |y|=4.243, w=5`: ppGO alone gives
  residual 1.4e-3 at N(theta)=4; adding the admitted ghost gives 4.2e-2 at
  N(theta)=14. Two causes — the admission set flips across theta inside a tile
  (43-54 of 65 sample points admitted), and `geometry.ghost_kernel` pins its
  sqrt branch with `reference_amplitude = exp(-0.5j*pi)`, justified in its own
  docstring by "the two real images continue into a Morse-index-1 saddle".
  That is a POSITIVE-PARITY statement: on the macro saddle both real images
  are ALREADY index-1, so the branch reference has not been derived for
  `det A < 0`. Refusing costs almost nothing — ppGO alone is better everywhere
  measured.
- **The census in the exterior annulus is (1,1)** — two images, both Morse
  index 1 — at every sampled point for `gamma in [1.05, 1.6]`. The (0,1,1,1)
  census occurs only in the `gamma <~ 1.03` region, which is interior and out
  of scope.
- **The fence is exact and is a BAND** (F026). One closed form covers both
  parities: `|y|**2(u) = 2u - 3 + 2*gamma**2/u + (1-gamma**2)/u**2` with
  `u = 1/|x|**2`, stationary at `u = 1` and `u_c = (sqrt(4*gamma**2-3)-1)/2`,
  and `u_c > 0` iff `gamma > 1`. Hence

      max|y|_saddle = sqrt(max( 4*u_c + 1/u_c - 2,  4*gamma**2/(gamma+1) ))

  the two candidates being the off-axis cusp (`u_c`) and the on-axis cusp.
  The outermost point switches from off-axis to on-axis at
  `gamma = 1.177651`, where the extent MINIMISES at 1.596072 — the
  non-monotonicity is a real cusp switch, not noise. Crossings:

      inner edge |y| = 3.0        gamma = sqrt((189 - 15*sqrt(105))/32)
                                        = 1.0502342
      outer edge |y| = 3*sqrt(2)  gamma = sqrt(63 - 24*sqrt(6))/2
                                        = 1.0261879
      RE-ENTRY at the inner edge  4*gamma**2 - 9*gamma - 9 = 0, gamma = 3

  So the annulus is exterior for `1.0502342 < gamma < 3`. Our prior stops at
  1.6 (`max|y| = 1.9846`, clear by 1.51x) — write the fence as a BAND anyway,
  so a widened prior cannot silently serve a region that is interior again.
- F024's MEASURED extent table (`3.71` at gamma=1.02 etc.) is RETIRED: its
  241^2 grid missed the thin spike (true values 4.886, 3.008, 9.939). Use the
  closed form only.

## In scope

- A saddle branch in `_born.py`'s lead-only carrier: the `sqrt(|mu_macro|) *
  exp(-1j*pi/2)` origin for `det_a < 0`. `_born_factors` currently assumes a
  positive radicand — extend it, do not fork it.
- The saddle fence as a named refusal, written as the BAND
  `1.0502342 < gamma < 3` using the closed form above, not a magic number.
- A saddle branch in `channels.born_carrier_from_partition`: same
  `w * Delta_tau = RHO_END` split, lead-only below, ppGO over both real images
  above, and the complex ghost REFUSED (not merely absent — refuse it
  explicitly with a comment naming the underived branch reference).
- The `'born'` census predicate extended to the saddle: it is currently
  `det_A > 0 AND gamma < 0.75 AND 3.0 < |y| <= 4.2426`. Add the saddle arm.
  It was written kappa-aware for exactly this.

## Out of scope

- Re-deriving `ghost_kernel`'s Morse branch reference for `det A < 0`. That is
  its own commission; here the ghost is simply refused.
- Per-column (per-theta) admission. The scalar fence costs only 3.1 % of the
  shear range on this branch (against 15.6 % on the positive side), so it is
  not worth spending here. Keep the fence parameterised so per-column can
  replace it later without restructuring.
- `gamma <= 1.0502342` — interior or straddling; a different problem.
- Wiring the likelihood serve slot. Residual chart is a driver artifact.
- Both carrier-continuity guards. Not implicated.

## Acceptance (build-level)

1. The saddle carrier equals `sqrt(|mu_macro|)*exp(-1j*pi/2)*exp(1j*w*phi_geo)`,
   checked against an INDEPENDENT reconstruction of `mu_macro` from
   `det A = lam**2 - gamma**2 < 0`, not a copy of the code expression.
2. `|F_carrier|` is `w`-INDEPENDENT and equals `sqrt(|mu_macro|)` to machine
   precision — the saddle counterpart of the F009 pin that `a0` violated on
   the positive branch.
3. The fence refuses at `gamma = 1.0502342` exactly (both sides of the
   crossing tested) and at `gamma >= 3`, and serves between. Cross-check the
   closed form: `max|y|_saddle == 3.0` at the lower root to 1e-10.
4. The split is the SAME named `RHO_END` constant in `w * Delta_tau` as the
   positive branch — a reachable-red must show `w * r0_sq` mis-splits at
   `gamma=1.2, theta=0.3` where `r0_sq/(2*Delta_tau) = 35.3`.
5. The complex ghost is refused on `det_a < 0`: a reachable-red shows that
   admitting it inflates the azimuthal node count (measured signature:
   N(theta) 4 -> 14 at `gamma=1.6, |y|=4.243, w=5`).
6. Residual node counts within a factor ~2 of N=4 on `log_w` and N=4 per
   y-axis in the low band, measured AZIMUTHALLY as well as radially. Radial
   sweeps hid a pathology for two rounds on the positive branch; do not repeat
   it.
7. `classify_fallthrough` returns `'born'` for a non-served saddle annulus
   draw, with a reachable-red showing the pre-build logic gives `out-of-box`.
8. Positive-parity behaviour is BYTE-IDENTICAL. State the driver-run recipe
   (config sweep + which outputs and npz byte streams to diff against the
   pre-build tree). Do NOT write a committed test importing a module from a
   git revision — retired in 8901b0b (F022).
9. Full fast suite green, driver-verified post-build.

## Constraints

- Branch `claude-dev` only.
- Slow tests NEVER run in-build; the tier env vars stay unset in agent envs.
  In-build tests must be FAST — small synthetic configs, few-eval oracles.
- `_born.py` is a pure float64 scalar path — keep it so, no `fastmath` (the
  phase must stay reproducible), no numpy objects on the scalar path.
- Verify existing tests for backward compatibility BY READING, including gated
  ones. The positive-parity Born tests must keep passing unchanged; if one
  moves, the saddle branch has leaked.
- The pre-commit drift hook blocks on gated tests referencing changed APIs and
  a build CANNOT satisfy it (it needs tier runs the driver owns). If you hit
  it, report the flagged list and STOP — do not `--no-verify`.
- If the Inspector loop reports the same finding set twice, it will now
  terminate on its own (fixed in `31ee133`); do not work around it.
