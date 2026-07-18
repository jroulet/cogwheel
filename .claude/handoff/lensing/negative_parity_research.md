# Negative-parity (macro saddle) Chang-Refsdal: the treatment exists

Professor, research commission, 2026-07-18.  Engine frozen at ec8a276.

Verdict: **(i) THE TREATMENT** - with one genuine and precisely
characterized obstruction inside it.  The geometry layer, the deep-band
limit, the geometric branch, and the SACR-C switched-analytic +
single-envelope decomposition all carry over to the macro-saddle domain
`1 - kappa < |gamma|` (with `1 - kappa > 0`), verified numerically
end-to-end.  The one piece that does NOT survive is the wave branch's
shear-operator series: it has a hard convergence radius at the parity
boundary `|gamma'| = 1` and is numerically divergent (best truncation
error O(1)) everywhere in the saddle domain.  Its replacement is an
EXACT one-dimensional Schwinger-parameter representation of the same
diffraction integral, derived below, validated against an independent
high-precision 2D lens-plane oracle to 2.2e-15, and carrying a single,
`y`-independent cancellation channel `L_S = pi*w/4` that double-double
arithmetic holds to the 1e-10 target out to `w ~ 64` - almost exactly
the existing `DD_PRODUCT_CEILING`-band ceiling of the positive-parity
engine.  Envelope node counts on the saddle domain: **N = 20-25 across
all 15 tested configurations**, identical to the positive-parity
SACR-C band (19-26).

All numerics in this note were actually run; scripts in the session
scratchpad (`np_exp1_geometry.py` .. `np_exp9_deepphase.py`),
interpreter `cogwheel-newlal`, engine at ec8a276 (HEAD 905869b tree,
untouched).

---

## 1. Setup, conventions, and verified code pins

Paper/engine convention (tex Eq. fermat/diffraction-integral;
`geometry.delay`, verified):

    tau(x; y) = x.A.x/2 - y.x + |y|^2/2 - ln|x|,
    A = (1-kappa) I - gamma Q(beta),   eigenvalues a = 1-kappa-gamma,
                                       b = 1-kappa+gamma  (beta = 0),
    F(w, y) = (w / (2 pi i)) Int d^2x exp[i w tau(x; y)].

Macro saddle: `a < 0 < b`.  Guards enforcing `1 - kappa > |gamma|`
exist at exactly four sites (verified): `geometry.macro_matrix`,
`geometry._centered_source_images` (y = 0 only),
`geometry.critical_point` / `nearest_caustic_point`, and
`operator._mass_sheet_map` (hence all of `F_op`/`F_op_grid` and
`channels.ChangRefsdalChannels.evaluate`).  Everything else in
`geometry.py` - the quartic solver `find_images_quartic`, `hessian`,
`delay`, `magnification`, `morse_index`, `saddle_coefficients`,
`image_kernel` - is parity-agnostic algebra and works UNMODIFIED on an
indefinite matrix (all claims below verified by calling those internals
with a hand-built `diag(1-g, 1+g)`, `g > 1`).

## 2. Mass-sheet reduction and the domain split

The mass-sheet identity is untouched by parity as long as
`lam = 1 - kappa > 0`:

    F_{kappa,gamma}(w, y) = (1/lam) exp[i w (ln(lam)/2
        - kappa |y|^2 / (2 lam))] F_{0, gamma/lam}(w, y / sqrt(lam)),

now with reduced shear `g' = gamma/lam > 1`.  Verified on the saddle
domain with the 1D representation evaluating both sides at their
distinct arguments: relative residual 9.6e-17 / 1.2e-17 (w = 3, 0.5).
The saddle domain therefore reduces exactly to the ONE-parameter family
"pure shear `g' > 1`", same as the positive-parity engine's reduction.

Domain split (proposal):

* `lam > |gamma|`  - existing positive-parity branch (unchanged).
* `0 < lam < |gamma|` - NEW saddle branch (this note).
* `lam <= 0` - NAMED REFUSAL (`LensDomainError`).  `sqrt(lam)` is
  imaginary, the reduction dies, and `kappa >= 1` configurations
  (including Type III maxima, `lam < -|gamma|`) need a genuinely
  different treatment.  No clean conjugation identity exists (negating
  the quadratic part does not negate the `-ln|x|` term).
* `lam = |gamma|` (det A = 0) - refusal boundary between branches; the
  physical branch point (see Sec. 6.3).  F004's float64-exact-boundary
  lesson applies verbatim to the new guard's test points.

## 3. Image topology: census, critical curves, caustics

### 3.1 Morse census (derived, then verified)

Index theorem for the lens map `V(x) = A x - x/|x|^2 - y`: the sum of
signed image parities equals `sign(det A)` (index at infinity) minus 1
(index of the point-mass singularity).  For a saddle,
`sum_a sign(mu_a) = -1 - 1 = -2`.  Since
`tr(Hess tau) = 2 lam > 0` (the log potential is harmonic - traceless
Hessian), NO maxima can occur (`n = 2` impossible, same as positive
parity).  Hence exactly:

* 2-image region: **(1, 1)** - both images are saddles (Type II-like);
* 4-image region: **(0, 1, 1, 1)** - one minimum plus three saddles.

Verified: 4000 random sources at `g = 1.3` through
`find_images_quartic` + `morse_index`: census (2,(1,1)) x 3830,
(4,(0,1,1,1)) x 170, ZERO anomalies (index sum always -2, residuals
< 1e-7, positions Newton-polished).  The quartic path needs NO
modification for off-center sources; only `_centered_source_images`
(exact `y = 0`) needs a saddle case (images `+-b^{-1/2} e_b` on the
positive-eigenvalue axis only - two saddles, consistent with the
census; the current code raises there).

### 3.2 Critical curves: the engine's formula extends verbatim

`det Hess tau = (lam^2 - gamma^2) + 2 gamma cos(2 theta')/r^2 - 1/r^4`
(polar `theta' = theta - beta`).  Zeros in `v = 1/r^2`:

    v(theta') = gamma cos(2 theta') +- sqrt(gamma^2 cos^2(2 theta')
                + lam^2 - gamma^2)

which for `gamma < lam` is exactly the engine's astroid formula
(`critical_point`: `+` root only, all angles).  For `gamma > lam` the
discriminant requires `|sin(2 theta')| <= lam/gamma`, i.e. two angular
WEDGES around the NEGATIVE-eigenvalue axis, and BOTH roots are
positive: the critical set is **two closed lobes** (one per wedge,
mirror images under `x -> -x`), each parametrized by
`theta' in [-theta_max, theta_max]`, `theta_max = arcsin(lam/gamma)/2`,
with the `+-` branches joining at the wedge edges.  Verified:
`det Hess = 0` to machine precision on both branches across the wedge
(np_exp1).

### 3.3 Caustics: two 3-cusp deltoids on the far side

Each lobe maps to a closed source-plane caustic with exactly **3
cusps** (tangent-reversal count on a 8000-point traversal: 3, at
`y = (-1.7144, 0)` and `(-1.0492, +-0.9524)` for `g = 1.3`, right
lobe) - the classic Chang-Refsdal `gamma > 1` "deltoid" (An & Evans
2006 picture), replacing the positive-parity astroid (4 cusps, one
curve).  The right lens-plane lobe (`theta ~ 0`, the `a < 0` axis) maps
to the caustic on the NEGATIVE `y1` side (bbox `y1 in [-1.71, -1.05]`,
`y2 in [-0.95, +0.95]`): sources inside either deltoid have 4 images,
outside have 2.  Fold and cusp crossings behave exactly like the
astroid case (2 <-> 4 with a merging pair at the fold; merging triple
at the cusp), confirmed by the crossing configs of Sec. 8.

## 4. Deep band (w -> 0): the F009-S limit

Derivation: the Gaussian/Fresnel integral of the indefinite quadratic
gives one `e^{+i pi/4}` and one `e^{-i pi/4}` direction, so

    F(w -> 0) -> e^{-i pi/2} / sqrt(gamma^2 - (1-kappa)^2)
              = e^{-i pi/2} sqrt(|mu_macro|),

i.e. F009's magnitude law with `|det A|` and a frequency-independent
**Morse phase `e^{-i pi/2}`** (the macro image is a saddle).  Verified
(1D rep, mpmath dps 35-40):

* magnitude: `|F| - |det A|^{-1/2}` vanishes LINEARLY in `w` (fitted
  exponent p = 1.00); at `w = 1e-4`, rel 4.4e-5 (config
  `a=-0.3, b=2.3, y=(0.4,0.3)`); kappa != 0 case
  (`kappa=0.3, gamma=1.0`): |F| -> `1/sqrt(gamma^2-lam^2)` confirmed.
* phase: intercept `-1.570956` vs `-pi/2 = -1.570796` (1.6e-4 abs);
  the O(w) drift is fully modeled by
  `arg F = -pi/2 + w [tau_G + (1/2) ln(w/2) + c0] + O(w^2)` with
  `tau_G = tau(A^{-1} y; y)` (full Fermat delay at the macro stationary
  point, log term included) and a config constant `c0` (0.611 here;
  residual/w constant to 1e-3 over `w in [1e-4, 3e-3]`).  The
  `(w/2) ln(w/2)` term is the point-mass core normalization - the same
  term that sits in the engine's `C(w)` prefactor at positive parity.
  It is NOT a defect and must not be "fixed".

Consequences carried over from F009: an "unlensed limit" test fixture
must still use `gamma = kappa = 0`; a saddle lens's amplitude never
relaxes to 1; and NO small-w short-circuit may be added.  New content
for F009-S: the limit is complex (`-i` times a positive real), so any
magnitude-only pin must be complemented by the Morse-phase pin.

## 5. The obstruction: the shear-operator series cannot cross parity

Radius-of-convergence argument: the operator exponential IS the Taylor
series of `F` in `gamma'` about 0 (term `n` carries `gamma'^n`).  As
`w -> 0` its sum must reproduce `1/sqrt(1 - gamma'^2)`, whose Taylor
series has radius EXACTLY 1: the parity boundary is a genuine branch
point of the amplification in the shear parameter, and no re-centered
or re-split expansion around a positive-definite seed can reach past it
(the singular locus `|1 + delta| = |gamma|` sits at distance <= 1/sqrt2
from any isotropic-seed expansion center; checked analytically).

Measured (np_exp4; engine's own table/weights/dd-kernel ladder with the
guard bypassed, `max_order = 42`, truth = validated 1D rep,
`y = (0.4, 0.3)`):

| g'   | best relative error over ALL truncation orders, w = 0.5..20 |
|------|--------------------------------------------------------------|
| 0.90 | 1.2e-2 (w=0.5) .. 3.2 (w=20)  - the known certified edge     |
| 1.05 | 0.24 .. 2.2   - terms grow from order 1; no usable window    |
| 1.30 | 1.5 .. 5.4                                                   |
| 1.50 | 1.8 .. 5.5                                                   |

Past `g' = 1` the best achievable truncation error is O(1) at EVERY
frequency - not an accuracy degradation but a structural divergence
(min |term| at order 1, monotone growth after).  Borel/Pade resummation
was not pursued: the target sits on the real axis beyond the branch
point, exactly where Pade pole strings land (dead end; do not re-try
without new mathematics).

## 6. The replacement wave branch: exact 1D Schwinger representation

### 6.1 Derivation

Insert `r^{-i w} = (1/Gamma(iw/2)) Int_0^inf dt t^{iw/2 - 1} e^{-t r^2}`
(Schwinger/heat-kernel identity, defined at `Re s = 0` by analytic
continuation) into the diffraction integral; the `x` integrals become
exact Gaussians for ANY signature:

    F(w,y) = (w / (2 pi i)) e^{i w |y|^2 / 2} (pi / Gamma(iw/2))
             Int_0^inf dt t^{iw/2-1} h(t),
    h(t) = (t - iwa/2)^{-1/2} (t - iwb/2)^{-1/2}
           exp[ -w^2 y1^2 / (4(t - iwa/2))
                -w^2 y2^2 / (4(t - iwb/2)) ],

principal square roots (both factors separately; `Re(t - iw./2) > 0` on
the contour).  The `t -> 0` endpoint is one integration by parts
(continuation in `s = iw/2`):

    Int_0^T t^{s-1} h dt := T^s h(T)/s - (1/s) Int_0^T t^s h'(t) dt,

`h'` in closed form; both remaining integrals are absolutely convergent
after `t = e^u` (tail decays as `t^{-2}`, take `T = w(|a|+|b|+2)/2`
past both branch points).  Branch points sit at `t = iwa/2` (LOWER half
plane for `a < 0`) and `t = iwb/2` (upper); the real-`t` contour is
clean for the saddle.  As `a -> 0^-` the branch point migrates through
`t = 0` INTO the contour endpoint - the det A = 0 parity boundary
appears as an explicit contour pinch: the representation itself names
its refusal boundary.

### 6.2 Verification chain (all run this session)

* vs the point-mass closed form (`a = b = 1`): 0 at dps 30 display
  precision, three (w, y) points (np_exp2a).
* vs engine `F_op` at positive parity: 3.9e-15 .. 1.9e-14 at
  `(w, gamma, kappa)` = (3, 0.2, 0.1), (10, 0.2, 0.2), (1, 0.35, 0);
  high-w: 6.5e-14 at w=30, 3.4e-11 at w=45, 4.2e-8 at w=58 (the w=58
  figure is the mutual engine+quadrature agreement at L_1F1 = 31,
  where the engine itself sits near its own dd degradation band).
* vs the INDEPENDENT 2D lens-plane oracle on the SADDLE domain: the 2D
  rotated-contour integral (per-axis Fresnel rotation by
  `alpha = pi/8`, sign following each eigenvalue; branch points of
  `log(x1^2 + x2^2)` provably avoided for `alpha != pi/4` - the
  rotation legality argument is in np_exp2_oracle.py's construction)
  evaluated with mpmath adaptive quadrature at dps 15:
  **rel diff 2.2e-15** at (w=3, y=(0.4,0.3), a=-0.3, b=2.3); plus a
  float64 tensor-GL version of the same contour agreeing within its
  own measured grid-convergence error (1e-4 .. 6e-5) at seven saddle
  configs including 4-image, soft-axis, and kappa != 0 (np_exp3).
* mass-sheet identity across the reduction (Sec. 2): ~1e-16.
* deep-band closed form (Sec. 4).

### 6.3 The new cancellation law (F001-S) and evaluation ceilings

The prefactor `1/Gamma(iw/2)` grows as `e^{+pi w/4}` while the
oscillatory `t`-integral supplies the compensating `e^{-pi w/4}`:
quadrature in float64 loses `L_S = pi w / (4 ln 10)` digits.  Measured
(np_exp7 A, saddle config): rel err = 7.4e-15 (w=5), 5.3e-13 (w=10),
3.2e-11 (w=15), 2.3e-9 (w=20), 2.8e-7 (w=25), 7.5e-6 (w=30) - tracking
`e^{pi w/4} * 1e-16` within a factor 2-8.  CRUCIALLY, the channel is
**y-INDEPENDENT** (measured at |y| = 0.1 .. 3.0 at w = 10 and 20: no
trend, all within a factor ~6 of the w-only law; np_exp8): the F001
two-channel law collapses to ONE channel for the saddle branch - there
is no `L = w|y'|` exponent and no 1F1 ladder at all.

Ceilings this implies:

* float64 scratch evaluator: 1e-10 target holds to `w ~ 18`, 1e-6 to
  `w ~ 30`.  Cost measured: 0.4 - 1.4 ms/node (`~4w` GL points per
  unit ln t, two integrals).
* production double-double integrand (the engine's existing `_dd`
  substrate; phases are O(w ln t) ~ O(300), well inside dd range):
  `pi w/4 <= (31.9 - 10) ln 10` gives **`w <= 64` at the 1e-10
  target** (`w <= 74` at 1e-6) - within a whisker of the
  positive-parity branch's `DD_PRODUCT_CEILING = 60` band edge.  The
  saddle branch's certified `w`-ceiling is a hard number of the SAME
  magnitude the engine already lives with, but now in `w` alone
  (`W_MAX_CERTIFIED = 500` at small `y` has no saddle analogue -
  named refusal above the ceiling unless resolved, where the
  geometric branch takes over).
* refusal margin at the parity boundary: as `|a| -> 0` the pinch
  approaches the contour; certify-or-refuse should be driven by a
  MEASURED quadrature-error estimate (paired coarse/fine rules), not
  an a-priori `|a|` cut - the random scan below certified down to
  `g' = 1.05` without special handling.

### 6.4 The steepest-descent route (documented alternative, not built)

Substituting `t = (iw/2) v` extracts `e^{-pi w/4}` ANALYTICALLY
(`(iw/2)^{iw/2}`) and turns the integrand into pure stationary-phase
form `v^{iw/2-1} [(v-a)(v-b)]^{-1/2} e^{i w Phi(v)}`,
`Phi = y1^2/(2(v-a)) + y2^2/(2(v-b))` (plus the log), whose stationary
points satisfy EXACTLY the engine's image quartic: **`v* = u = 1/|x|^2`
of the geometric images** - the 1D representation is a Lefschetz-thimble
package of the image sum.  A production implementation on deformed
`v`-contours would remove the `e^{pi w/4}` cancellation entirely
(float64-safe to arbitrary w) at the price of one-dimensional
Picard-Lefschetz bookkeeping near the essential singularities at
`v = a, b`.  Recommended only if the dd-quadrature ceiling `w ~ 64`
ever binds in production; the correspondence `v* = u` is also the
natural bridge for a saddle-branch `select_branch` consistency test.

## 7. Geometric branch: works verbatim

`geometry.image_kernel` already carries `sqrt|mu| e^{-i pi n/2}
(1 + i C1/w + C2/w^2)` with `n` from the Hessian eigenvalue count -
nothing assumes positive parity, and the saddle census (Sec. 3.1) only
uses `n in {0, 1}`, which the existing Morse-phase code covers.
Measured against the 1D rep (two-image saddle, `g = 1.3`,
`y = (0.4, 0.3)`, pair separation `d tau = 0.385`):

    |F - sum_a e^{i w tau_a} H_a| / |F| = 0.60 (w=1), 7.3e-2 (w=4),
    8.9e-3 (w=8), 2.3e-4 (w=13);  kappa=0.4 config: 9.3e-5 at w=18.

i.e. the stationary-phase sum converges onto the exact amplification
at the usual `(w * separation)^{-3}`-flavored rate.  C1/C2 values are
O(1) away from caustics (e.g. -0.64/-0.70, -0.55/-0.67).  The
`select_branch` structure (resolution AND strong-cancellation) carries
over with the cancellation condition replaced by the `w`-ceiling of
Sec. 6.3.

## 8. SACR-C carry-over: certified on the saddle domain

Construction identical to the positive-parity SACR-C:
`F = sum_a e^{i w tau_a} S_a H_a + e^{i w tau_c} E(w)` with
`S_a = smootherstep(w |tau_a - tau_c|, 0.5, 4)`, `tau_c` the Fermat
delay (relative to `t_min`) of the nearest caustic point, `E` the one
interpolated envelope.  The ONLY generalization needed:
`nearest_caustic_point` must search the TWO lobes with BOTH `+-`
branches of the `v(theta')` formula (Sec. 3.2) instead of one astroid -
implemented in scratch as a 720-angle scan over both signs + bounded
refinement; everything else (`smootherstep`, `image_kernel`, envelope
projection, `_assign_labels` parking virtual labels at the critical
point) is engine code reused as-is.  The bounded-phase theorem is
parity-blind (it is carrier algebra: switch scale == demodulation
distance), and the deep band carries F009-S verbatim
(`E -> e^{-i w tau_c} F`, all phases `< RHO_START`).

Greedy node counts, eps(w) = |dF| / max(|F|, 0.15 max|F|) < 1e-3,
cubic spline in ln w on Re/Im E, 2-decade windows
`[0.2, 20]/delta_key` (4-image capped at w = 30 by the scratch
float64-truth ceiling; still covers every switch completion with 2.2x
margin), np_exp7 C:

| config (g = 1.3 unless noted)            | n_img | N  | max|S H| |
|------------------------------------------|-------|----|----------|
| two-image y=(0.4,0.3)                    | 2     | 24 | 0.34     |
| four-image y=(-1.31,0)                   | 4     | 20 | 1.46     |
| near-fold eta=+0.002 / -0.002            | 2 / 4 | 22 / 22 | 1.34 |
| near-cusp eta=+0.002 / -0.002            | 4 / 4 | 22 / 22 | 1.30 |
| kappa=0.4, gamma=1.2, y=(0.4,0.3)        | 2     | 25 | 0.40     |
| random scan, 8 configs, g in [1.05,1.6]  | 2     | 20-25 | <=2.80 |

**N = 20-25 everywhere** - inside the positive-parity certified band
(19-26), config-independent, including fold and cusp crossings at
eta = +-0.002 on both topology sides.  Decomposition identity residual
~1e-15-1e-16 on every config (algebraic, switch-independent).
`max |S_a H_a| <= 1.46` on all crossing configs (positive-parity gate
was 2); two random 2-image configs reached 2.4-2.8 (moderately
magnified images mid-switch) - bounded, splines unaffected; propose
the saddle-scan gate at 4 with the crossing gate kept at 2.

Scratch-experiment caution transferred: the envelope must be built
with F demodulated by the SAME `t_min` used for the relative carriers
(a mismatched convention reproduces exactly the beat disease at
`t_min`-scale and inflated a first run to N = 72; engine `channels.py`
already does this correctly via `exact_total`'s `t_min` demodulation -
verified in `_exact_total`'s signature use).

Likelihood/ratio layer: unchanged in structure.  `w prop f` still makes
image delays pure time shifts; the 4-channel-with-per-frequency-weights
or 5th-envelope-channel forms and the pair-summary algebra never
reference parity.  The Type II overall `e^{-i pi/2}` phases live inside
`H_a`/`E` exactly as the positive-parity Morse phases do today.  Label
continuation: initial labeling and path-based continuation carry over;
NEW risk - `nearest_caustic_point` can now jump BETWEEN LOBES as
proposals move (astroid never had disconnected critical sets).  Each
evaluation stays exact regardless (same telescoping argument); only
kernel-ratio smoothness across proposals is affected, and only via
`tau_c` jumps - same class as the fold-to-cusp jumps already accepted
for the astroid, but worth one crossing-path spot check in-build.

## 9. Proposed certified domain and named refusals (F005-S discipline)

Certified (saddle branch):

* `lam = 1 - kappa > 0` and `gamma > lam` (reduced `g' > 1`), any
  `beta` (eigenframe rotation unchanged - one rotated config should be
  in the build gates), `y` unrestricted in the scanned range
  (|y| <= ~3 tested; no y-cancellation channel exists);
* wave branch: `w <= ~60` (dd quadrature; exact ceiling from the
  measured `e^{pi w/4}` law and the 1e-10 target, `w <= 64`, rounded
  down to the engine's familiar 60), refusal via a MEASURED
  coarse/fine quadrature-error estimate, never a silent return;
* geometric branch: resolved (`w * delta_min >= RHO_END`) AND
  `w` above the wave ceiling - the same two-condition gate;
* refusal wedge: unresolved AND `w > ceiling` - named
  `CancellationError`-analogue, exactly like today's F005 wedge.

Named refusals: `lam <= 0` (Type III / over-critical sheet);
`|1 - kappa| = |gamma|` boundary band (contour pinch; boundary test
points must be float64-exact per F004); quadrature-error cut;
`y = 0` Einstein-ring degenerate cases as today.

## 10. FINDINGS addenda required (if built)

* **F001-S**: the saddle wave branch has ONE cancellation channel,
  `L_S = pi w/4` in the Schwinger quadrature, y-independent; the 1F1
  `L = w|y'|` channel and the operator `L_op = w gamma'/2` channel do
  not exist there.  dd holds 1e-10 to `w ~ 64`.
* **F005-S**: certified-or-refuse for the saddle branch is
  quadrature-error-driven; the F_op refusal cuts do not apply.
* **F009-S**: deep-band limit `e^{-i pi/2}/sqrt(gamma^2 - lam^2)`
  with O(w) magnitude correction and the modeled
  `w[tau_G + (1/2)ln(w/2) + c0]` phase drift; magnitude-only pins are
  insufficient - pin the Morse phase too.
* **F008 note**: the criticality-separation switch generalizes with
  the two-lobe nearest-caustic search; no keying change.
* **F004 note**: applies to the new `lam <= 0` and parity-boundary
  guards.

## 11. Build-brief-ready summary

**Mission.** Extend the Chang-Refsdal engine to negative-parity
(macro-saddle) hosts: saddle-capable geometry (guard split, centered
-source case, two-lobe critical utilities), a new certified
saddle wave-branch evaluator (dd Schwinger quadrature), and the SACR-C
channel construction over the saddle domain.  Positive-parity behavior
byte-identical.

**Shape: two sequential builds** (honest decomposition exceeds 3 WPs
otherwise).

*Build S1 - engine (geometry + wave branch).*  In scope:
`geometry.py` (parity-aware `macro_matrix` domain split with `lam <= 0`
refusal; `_centered_source_images` saddle case; `critical_point` /
`_caustic_source` / `nearest_caustic_point` two-lobe, two-branch
extension using the existing `v(theta')` formula), new module
`_schwinger.py` (dd-integrand 1D quadrature evaluator, certified-or-
refuse via paired quadrature rules, ceiling `w <= 60`), `operator.py`
saddle dispatch (`F_op` routing by parity, `select_branch` with the
w-ceiling condition; existing positive-parity path BYTE-FROZEN).
Fast gates: (1) census test - 200-source scan, index sum -2, census
sets exactly {(1,1),(0,1,1,1)}; (2) evaluator vs an mpmath dev-oracle
(the 1D rep at high dps, F002-clean: independent implementation, not
the production code's own path) at <= 1e-10 over a (w, g', y) grid
w <= 60 incl. g' = 1.05; (3) the 2.2e-15-class independent 2D-oracle
anchor reproduced at one saddle point; (4) mass-sheet identity on
observables; (5) deep-band pins: |F| closed form AND -pi/2 phase
intercept; (6) geometric-branch agreement at resolved w; (7)
positive-parity regression: full existing suite untouched and green.

*Build S2 - channels + likelihood + prior.*  In scope: `channels.py`
(saddle-domain `evaluate`; virtual labels on the nearest lobe;
crossing-scenario fixtures built from geometry+_gauge only, F002),
`likelihood.py` (branch plumbing only - the envelope/LOO machinery is
parity-blind), `lensing/prior.py` (domain description: either
positive-parity-only unchanged, or the two-domain prior with the
parity-boundary refusal band mapped to lnL = -inf at proposal level -
coordinate with the Build-4 sampling layer, whose serialization work
is concurrent).  Fast gates: (1) SACR-C node-count gate N <= 30 for
eps < 1e-3 on the Sec. 8 anchor set (2-decade windows); (2) identity
residual <= 1e-13; (3) max|S H| <= 2 on fold/cusp crossings at
eta = +-0.002, <= 4 on the random scan; (4) lobe-jump spot check
(kernel-ratio continuity along one path crossing lobes); (5) RB lnL
vs brute force on one saddle config within the standard tolerance.
Post-build, driver-verified: 25-config saddle scan, warm-lnlike
timing, full-suite regression.

**Out of scope (both builds).** The v-plane steepest-descent
evaluator (Sec. 6.4); `lam <= 0` (Type III); any change to the
positive-parity operator/1F1/refusal constants; ratio-layer speedups.

## 12. Dead ends and residual risks

* Operator-series resummation past `g' = 1` (Pade/Borel): branch point
  sits exactly where Pade lays its cut; do not re-try without new
  mathematics (Sec. 5).
* Naive float64 Schwinger quadrature above `w ~ 20`: the
  `e^{pi w/4}` law is unforgiving; the first 4-image run silently
  produced garbage truth at `w = 69` (values 1e7) - in production this
  band must exit through the quadrature-error refusal, never a return.
  This is the saddle branch's version of the F005 silent-nan lesson.
* Envelope built with mismatched `t_min`/carrier conventions
  reproduces the beat disease (N 72 vs 24 on the same config) -
  covered by the identity gate.
* mpmath's own adaptive quadrature under-resolves the t-integral at
  high w unless dps is scaled (~30 + w used here); the dev-oracle in
  S1 must inherit that scaling, and above w~45 its agreement with the
  engine's positive-parity branch degrades into the engine's OWN dd
  band (4e-8 at w=58) - certify the oracle against the closed forms,
  not only against F_op.
* Two random-scan configs show max|S H| ~ 2.4-2.8 (vs <= 1.3 at
  positive parity): bounded and harmless here, but the in-build scan
  gate should measure it rather than assume the old constant.

## 13. Scripts (session scratchpad)

np_exp1_geometry.py (census/critical/caustic), np_exp2_oracle.py
(mpmath 2D rotated-contour + 1D rep oracles), np_exp2a_1d.py,
np_exp2b_mp2d.py (high-precision 2D anchor), np_exp3_2d64.py (float64
2D + mass-sheet), np_exp4_series.py (operator-series divergence),
np_exp5_saddle_pipeline.py (float64 1D evaluator + first SACR-C),
np_exp6_diagnose.py (t_min bug + carrier variants), 
np_exp7_consolidated.py (cancellation law, high-w validation, cusps,
deep pins, final SACR-C table), np_exp8_ydep.py (y-independence),
np_exp9_deepphase.py (phase model).
