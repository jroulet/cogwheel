# Serving-coverage design, derived from physics

Written 2026-07-28 as a FRESH-EYES document: the decomposition below is derived
from the lens physics, not from what the repository currently implements. Part
IV then audits the repo against it. Where the two disagree, this document is
the claim and the repo is the defendant.

Motivation for writing it this way: the far-annulus rung was scoped around
`ANNULUS_INNER_RADIUS = 3.0`, which is `_Y_SCALE_CAP` — the PRIOR BOX half
width — not any physical scale. Two gamma fences and a refusal band were then
derived to defend that boundary. The physics question was never asked.

---

## Part 0 — the governing principle

**No absolute length may appear where the only scale is the caustic.**

The reduced (mass-sheet) Chang-Refsdal lens has exactly one shape parameter,
`gp = gamma / lam`. Every length in the source plane is measured in Einstein
radii, and the caustic's size varies by orders of magnitude across the prior
(`max|y|` runs from ~0 as `gp -> 0` to divergent as `gp -> 1`). Any constant
with length units is therefore a bug unless it is normalised by a local
structural scale.

The normalising quantity is region-dependent, and choosing it correctly is the
whole content of the principle:

| region | normalise by |
|---|---|
| caustic interior | `r_caustic(theta)` — directional radius, giving `rho = \|y\|/r_caustic in [0,1)` |
| near-caustic shell | LOCAL CURVATURE RADIUS `R_c(theta)` — the fold's own scale, giving `eta / R_c` |
| exterior, caustic encloses origin | `r_caustic(theta)`, additive outside: `rho = 1 + \|y\| - r_caustic` |
| exterior, caustic does NOT enclose origin (saddle) | no enclosing directional radius exists — additive scalar gauge only |
| far zone, `gp -> 0` | the caustic degenerates; normalise by the EINSTEIN radius instead |

**Falsifiable form of the principle:** grep the lensing package for a float
with length units and ask what sets it. If the answer is "the prior box", "a
round number", or "it worked at one gamma", it violates this document.

---

## Part I — the regions, from physics

The source plane decomposes by **image census and proximity to the caustic**;
the frequency axis decomposes by **image resolution**. These are independent,
so the full decomposition is their product.

### I.a Position classes (per parity)

Positive parity, `gp < 1`: one astroid caustic, 4 cusps, ENCLOSES the origin.
Macro saddle, `gp > 1`: two 3-cusp deltoid lobes, off-origin, neither encloses
the origin. `gp = 1` is `det A = 0`, a measure-zero named refusal.

| # | class | defining condition | census |
|---|---|---|---|
| P1 | deep interior | `rho << 1` | 4 images |
| P2 | interior near-fold | `eta / R_c << 1`, inside | 4 images |
| P3 | cusp neighbourhood | within the cusp's own scale of a cusp point | 4 <-> 2 transition |
| P4 | exterior near-fold | `eta / R_c << 1`, outside | 2 images |
| P5 | near exterior | `rho` slightly `> 1` | 2 images |
| P6 | far zone | `\|y\| >> ` caustic extent | 2 images + 1 complex pair |

The saddle has the same six classes PER LOBE, plus one class the astroid does
not have:

| P7 | inter-lobe corridor | equidistant between the two lobe centroids | 2 images |

### I.b Frequency classes

The physical discriminant is whether the stationary points are RESOLVED, i.e.
whether the phase difference between images exceeds the diffraction scale.

| # | class | condition | behaviour |
|---|---|---|---|
| W1 | diffractive | `w * Delta_tau <~ 1` | images unresolved; `F -> sqrt(mu_macro)` as `w -> 0` (F009) |
| W2 | transitional | `w * Delta_tau ~ 1` | the SACR-C switch band |
| W3 | geometric | `w * Delta_tau >> 1` | stationary phase valid; ppGO |

`Delta_tau` is the Fermat-delay difference of the two nearest images. It is the
ONLY correct currency (F024): `w * r0_sq` coincides with it at positive parity
by the accident `Delta_tau ~ r0_sq/2`, and mispredicts by two orders of
magnitude on the saddle. The measured switch is `w * Delta_tau ~ 4`, which is
SACR-C's own `RHO_END`.

---

## Part II — what serves each cell, and in which coordinate

The universal structure is CARRIER + CHARTED RESIDUAL. The carrier is an
analytic form; a spline interpolates `F_exact - F_carrier`. The carrier does
not need to be accurate — it needs to make the residual CHEAP TO SPLINE. This
is the same decomposition SACR-C already uses (analytic image kernels + one
interpolated envelope).

| cell | carrier | chart coordinate |
|---|---|---|
| P1 x W1-W3 | SACR-C switched-analytic (`tau_c`-demodulated) | `(gp, rho, theta, ln w)` |
| P2/P4 x W1-W3 | fold-uniform (Airy) in the fold's own scaled variable | `(gp, u = sqrt(eta/R_c), theta_arc, ln w)` |
| P3 x W1-W3 | cusp-uniform (Pearcey) in cusp-scaled variables | cusp-adapted, 2/3-power scaling |
| P5 x W3 | ppGO: coherent sum over real images, full C1/C2 | `(gp, rho, theta, ln w)` |
| P5 x W1 | lead-only macro carrier `sqrt(\|mu\|) e^{i n pi/2} e^{i w phi}` | same |
| P6 x W1 | same lead-only carrier | same |
| P6 x W3 | ppGO + complex-saddle ghost, decay-gated | same |
| P7 | (saddle) undecided — see C7 | — |

**Note what is NOT in this table:** any region defined by `\|y\| > 3`. The far
zone P6 is defined by `\|y\|` LARGE COMPARED TO THE CAUSTIC, which is a
`gp`-dependent, direction-dependent statement. The prior box corner is where
the SAMPLING stops, not where a physical regime changes.

---

## Part III — the checklist

Each item states a physics claim, the CHEAPEST check that would falsify it, and
what we already have. Work through in order; each is independently checkable.

### C1. The carrier for unresolved images is the lead term alone
CLAIM: for `w * Delta_tau <~ 4`, the cheapest-residual carrier is
`sqrt(|mu_macro|) * exp(i * n_macro * (-pi/2)) * exp(i w phi_geo)` with NO
amplitude corrections, at BOTH parities.
CHECK: node counts of the demodulated residual, azimuthal AND radial, lead-only
vs lead+corrections.
HAVE: F025 (positive, `a0` costs 11-44 nodes vs 4), F024 (saddle, 23-65 vs 4).
STATUS: **CHECKED**, both parities. Mechanism: `a0` violates F009.

### C2. The band split is `w * Delta_tau ~ RHO_END`
CLAIM: one criterion, both parities, no re-keying.
CHECK: a config where `Delta_tau != r0_sq/2` and the two currencies disagree.
HAVE: F024 — saddle `r0_sq/(2 Delta_tau)` spans 0.16 to 35.6.
STATUS: **CHECKED**.

### C3. The caustic extent has a closed form on both parities
CLAIM: `|y|^2(u) = 2u - 3 + 2 gp^2/u + (1-gp^2)/u^2`, stationary at `u=1` and
`u_c = (sqrt(4 gp^2 - 3) - 1)/2`, with `u_c > 0` iff `gp > 1`.
CHECK: against a dense caustic parametrisation, both parities.
HAVE: F026 — agreement to 4 decimals at 16 gammas from 0.45 to 3.0.
STATUS: **CHECKED**. This is the function every region boundary should be
expressed relative to.

### C4. The ghost's sqrt branch is parity-independent
CLAIM: `exp(-i pi/2)` is correct for `det A < 0` because `tr Hess = 2 lam > 0`
forbids index-2 images, so every A2 fold annihilates an (0,1) pair on both
branches.
CHECK: fold census across the caustic + `+G` vs `-G` against the exact engine.
HAVE: F027 — `-G` never best; `+G` wins for `w <= 1.41`.
STATUS: **CHECKED**.

### C5. The ghost needs a DECAY gate, not only a separation gate
CLAIM: `Im tau_c -> 0` on the principal axes, so the ghost stops decaying and
swamps a small ppGO residual; separation (near-cusp coalescence) is orthogonal
and does not catch it.
CHECK: `|G|/|F|` and residual vs `theta` approaching an axis.
HAVE: F027 — 1000x worse at `gp=0.45, theta=0.02`; separation never binds
(`min|x_a - x_c| in [0.94, 2.42]`).
STATUS: **CHECKED**. Fix: gate on `w_band_floor * Im tau_c >= ~2`, pinning
`w_min` to the CHART BAND FLOOR so train and serve see the same number.

### C6. The tube shell must be curvature-relative
CLAIM: the fold expansion's validity scale is the local caustic curvature
radius, so `eta_max` must be a FRACTION of `R_c(theta)`, not an absolute
0.05.
CHECK: at fixed `eta/R_c`, is the fold-uniform residual `gp`-independent? Sweep
`gp` over two decades and compare residual size at matched `eta/R_c` vs matched
absolute `eta`.
HAVE: circumstantial only — `_min_curvature_radius` already SKIPS a tube chart
when `eta_max` exceeds half the local curvature radius, i.e. the code already
knows the shell should be curvature-relative and enforces it by refusing.
STATUS: **UNCHECKED — highest-value next measurement.** If it holds, the
small-gamma collar disappears and the tube becomes prior-universal.

### C7. The exterior gauge must be additive where the caustic does not enclose
the origin
CLAIM: on the macro saddle, rays that miss both lobes have no enclosing
directional radius, so a multiplicative `|y|/r_caustic` is ill-posed and an
additive offset is the only correct gauge.
CHECK: exhibit a ray at `gp > 1` that intersects neither lobe and show
`r_caustic` refuses.
HAVE: already the implemented design and documented in SPEC; F026 confirms the
two lobes never enclose the origin.
STATUS: **CHECKED** (as a negative result — the principle has a genuine
exception here, by topology).

### C8. The far zone is defined relative to the caustic, not to the prior box
CLAIM: there is no physical boundary at `|y| = 3`. The weak-deflection carrier
becomes the cheapest description at large `rho`, and where that happens is
`gp`- and direction-dependent.
CHECK: sweep the carrier/ppGO/chart residual costs INWARD from the box corner
at several `gp` and find where the lead-only carrier stops winning. That
crossover is the real P5/P6 boundary.
HAVE: F023/F025 measured node counts only for `|y| in [3.05, 4.24]` — i.e.
only inside the box-artifact annulus.
STATUS: **UNCHECKED.** This is the measurement that would have prevented the
whole far-annulus mis-scoping.

### C9. `min_gamma_band` is a structure-spacing threshold, not a length
CLAIM: it exists because a topology-UNSTABLE band cannot be tiled coherently,
so it should scale with the local metamorphosis spacing — which differs by
parity (6 deltoid cusps on two lobes vs 4 astroid cusps).
CHECK: dropped mass vs `min_width`, both parities.
HAVE: measured 2026-07-28 — at `min_width = 0.02` the saddle drops 40.6% of its
branch against the astroid's 4.7%, an 8.7x asymmetry; every dropped sliver is
EXACTLY one `min_width` wide, so these are bands the splitter would keep if the
floor allowed.
STATUS: **MEASURED, TREATMENT UNDECIDED.** Not a length-scale violation — a
different threshold with the same "tuned on one branch" pathology.

### C10. `w > 60` has no exact evaluator, so nothing can be trained there
CLAIM: structurally different from every other item — not "the rung is
missing" but "no reference exists to build a rung against".
CHECK: none needed; it is a statement about the Schwinger ceiling.
STATUS: **KNOWN.** Blocks the full-box campaign, not any individual rung.

---

## Part IV — audit of the repository against Part 0

Constants with length units, and what sets them:

| constant | value | set by | verdict |
|---|---|---|---|
| `ANNULUS_INNER_RADIUS` | 3.0 | `_Y_SCALE_CAP`, the PRIOR BOX half-width | **VIOLATION** — a sampling bound used as a physical boundary |
| `_DEFAULT_ETA_MAX` | 0.05 | absolute | **VIOLATION** — should be a fraction of `R_c` (C6) |
| `_DEFAULT_ETA_FLOOR` | 0.02 | absolute | **VIOLATION** — same |
| `_SADDLE_CUSP_MIN_HALFWIDTH` | 0.08 | absolute floor | **VIOLATION** — should scale with the cusp's own scale |
| `GAMMA_FENCE` (3/4) | derived | from `ANNULUS_INNER_RADIUS` | consequence of the first violation; dissolves with it |
| saddle fence (1.0502342) | derived | same | same |
| `_GHOST_SEPARATION_MIN` | 0.7 | absolute, lens-plane | OK — lens-plane Einstein-radius-normalized; Einstein radius is the physical scale in the image plane (not caustic-relative); measured gap (refuse 0.29, admit 0.94) stable across gamma; traces to geometry, not the prior box (Build 7) |
| `_INTERLOBE_CORRIDOR_ETA_SCALE` | x `eta_max` | relative | OK — already relative |
| `rho`, `rho_lobe` | — | caustic-relative by construction | OK |
| `min_gamma_band` | 0.02 | gamma-axis, not a length | out of scope for Part 0; see C9 |

**Order of work implied by the audit:** C6 first (it is unchecked, it is the
highest-value, and it converts a refusal into a serve), then C8 (it re-derives
the P5/P6 boundary and retires the annulus concept), then the two fences fall
out for free. C5 is independent and owed on both branches regardless.
