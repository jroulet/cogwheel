# Envelope research: beat-free decomposition of the Chang-Refsdal transition-band kernels

Professor, research commission, 2026-07-18.
Verdict: **(i) THE DECOMPOSITION EXISTS** and is computable from existing engine
primitives at ~0.41 ms per coarse node. Numerically certified on all five
anchors plus a 25-configuration scan: null-safe relative error < 1e-3 with a
config-independent node count (greedy-oracle 19-26; self-certifying production
placement 30-44) on 2-decade bands, versus 40-53 for the current kernels under
the *same oracle placement* (and 50-90 as measured in Build 3d). No new engine
primitive is required. The Build-3e per-image residual R_j remains nonexistent
and is *not needed*.

All numerics in this note were actually run; scripts in the session scratchpad
(`envelope_exp1..6.py`), interpreter `cogwheel-newlal`, engine at HEAD
(905869b).

---

## 1. What the prototype actually interpolated (golden-lead answer)

Read: `.claude/spec/lensing_paper/code/{exact_gauge_partition.py,
chang_refsdal_exact_partition.py, chang_refsdal_topology_stable.py}` and tex
Sec. "Numerical results".

1. The prototype's partition is **block-structured**: persistent (resolved)
   images always carry the *analytic* saddle kernel `image_kernel`
   (`sqrt|mu| e^{-i pi n/2} (1 + iC1/w + C2/w^2)`) under their own carriers;
   only the **cluster residual** `exact_total - persistent_total`, demodulated
   at the *critical-point delay*, is split among cluster channels
   (`exact_transition_channels` with cluster-local weights). The residual
   projection never smears O(1) resolved-separation content into a channel
   demodulated far from its carrier.
2. The famous **6-11 node** figure is for greedy-adaptively placed nodes
   interpolating the **candidate/fiducial ratios** `q_a = (h_U/h_U0) K_a/K_a0`
   over the *sub-decade* band `5 <= w <= 40`, error metric
   `eps(w) = |dF| / max(|F|, 0.15 max|F|)` (tex Eq. error-profile;
   `data/channel_benchmark_summary.txt`: separated-ratio 11 vs direct 135).
   It is *not* a claim about raw kernels over a multi-decade band.
3. Scaled per decade, the prototype's ~7-12 nodes/decade matches exactly what
   the decomposition below achieves on the engine (~9-12/decade).

## 2. Root cause of the Build-3d beat disease (verified in engine source)

`channels.py::ChangRefsdalChannels.evaluate` treats all four labels as ONE
flat cluster: `exact_transition_channels(w, exact_total, mean(delays), delays,
physical, switch)` with **uniform weights 1/4**. Two structural consequences
(verified in `_gauge.py::exact_transition_channels` / `_member_split`):

- The unresolved trial for channel `a` is `(1/4) e^{-i w tau_a} F(w)`: the
  **full** total, containing every carrier `e^{i w tau_b}`, demodulated at
  `tau_a`. Any pair already resolved in-band (`w |tau_a - tau_b| >> 1`) beats
  at O(1) amplitude inside every unswitched kernel.
- The residual projection weights are uniform, so O(1) transition-band
  residual (which contains all carriers) is injected into *resolved* channels
  as well.

Additionally, the F008 switch separation `delta_a = min_b |tau_a - tau_b|`
stalls on **accidental** delay degeneracies: at the crown anchor the four
delays form two near-degenerate pairs with intra-pair separation 1.9e-3
(images 0.23 apart in the lens plane — *not* merging), so switches never
complete in band and the full-F artificial split persists, beating at the
resolved cross-pair separations. Measured control: even with oracle (greedy)
node placement the current kernels bind at N = 45 (crown), 40 (near-cusp),
53 (near-fold) — consistent with Build 3d's 50-90 under practical placement.

## 3. The decomposition

Per parameter point (labels, delays `tau_a` relative to `t_min`, real mask as
already built by `channels.py`):

    F(w) = SUM_a e^{i w tau_a} S_a(w) H_a(w)   +   e^{i w tau_c} E(w)

with, all closed-form:

- `H_a(w)` = `geometry.image_kernel(w, image_a, matrix)` for real channels,
  0 for virtual channels (verified symbol, includes C1/C2 corrections);
- `tau_c` = Fermat delay of the parked/critical carrier:
  `geometry.delay(geometry.nearest_caustic_point(...).image, source, matrix)
  - t_min` (exactly the engine's `virtual_delay`);
- switch `S_a(w) = smootherstep(w * delta_a, RHO_START, RHO_END)` with the
  **criticality separation** `delta_a = |tau_a - tau_c|` (virtual channels:
  `S == 0`); `smootherstep, RHO_START=0.5, RHO_END=4.0` from `_gauge` /
  `operator` (verified);

and ONE interpolated object, the **transition envelope**

    E(w) := e^{-i w tau_c} * ( F(w) - SUM_a e^{i w tau_a} S_a(w) H_a(w) ),

computed at coarse nodes from `F_op_grid` (the engine's certified evaluator)
plus closed forms, splined (cubic in ln w on Re/Im), and reconstructed densely.
The identity is algebraic: reconstruction == `exact_total` at machine
precision at every w where E is exact (measured 1.7e-16..3.3e-16 relative on
all anchors), independent of switch quality — same telescoping argument as
`_gauge`'s residual projection.

**Four-channel drop-in form** (if the likelihood keeps 4 channels): kernels

    K_a(w) = S_a H_a + u_a(w) e^{-i w (tau_a - tau_c)} E(w),
    u_a = (1 - S_a + eta) / SUM_b (1 - S_b + eta),   eta ~ 1e-2,

satisfy `F = SUM_a e^{i w tau_a} K_a` *identically* for any per-frequency
weights summing to one (same `_gauge` algebra; the current
`_normalized_weights` is static per-member — a per-frequency weights extension
or the 5-channel form is needed). The **5-channel form** (4 analytic channels
+ 1 envelope channel at carrier `tau_c`, 15 RB pair summaries instead of 10)
avoids `u_a` entirely and is the cleaner production shape: the only
engine-node-dependent object is E.

### Why E is beat-free (bounded-phase argument)

1. Content appears in E at O(1) amplitude only from channels with
   `S_a < 1`, i.e. `w |tau_a - tau_c| < RHO_END`. Its demodulated phase
   against the `tau_c` carrier is `w |tau_a - tau_c| <= RHO_END = 4 rad`.
   **The switch scale and the demodulation distance are the same quantity by
   construction** — that is the theorem the F008 nearest-neighbour rule lacks.
2. Switched channels contribute only their saddle-asymptote error, relative
   amplitude O(w^-3) past `1 + iC1/w + C2/w^2`; it beats at
   `|tau_a - tau_c|` but crosses below 1e-3 within a bounded number of visible
   cycles (in the scaled variable `w delta_a`), config-independently.
3. Deep-unresolved band: all `S_a = 0`, `E = e^{-i w tau_c} F`; every phase
   `w |tau_a - tau_c| < RHO_START`; F009's exact macro limit
   `F -> 1/sqrt((1-kappa)^2 - gamma^2)` is carried verbatim (no small-w
   surgery anywhere).
4. F008 intent preserved and sharpened: images merging at the critical point
   have `tau_a -> tau_c`, so `delta_a = |tau_a - tau_c| ~ delta_pair/2` —
   the switch is *at least as conservative* as the full-cluster rule for
   genuine mergers (measured `max_w,a |S_a H_a| <= 1.30` on all anchors,
   including fold/cusp crossings at eta = +-0.002: no kernel inflation),
   while **accidental** delay degeneracies between non-merging images
   (the crown pair) no longer stall the switch. In the crown's case the
   accidental pair also sits at `tau ~ tau_c`, so it stays unswitched — and
   that is *harmless*, because its demodulated phase in E is equally tiny:
   accidental degeneracy = small carrier separation = no beat, ever.

## 4. Numerical certification (all runs executed this session)

Truth = engine `exact_total` (switch-independent; `_exact_total` =
`F_op_grid` wave branch + `geometric_amplification` geometric branch).
Metric = paper's null-safe `eps(w) = |dF| / max(|F|, 0.15 max|F|)`; on the
anchor windows the greedy N is **unchanged** at floors 0.05 and 0.01, so the
floor is not doing the work. Interpolant: cubic spline in ln w on Re/Im of E.

### 4.1 Full transition-covering bands (2.7-4.6 decades, incl. deep-unresolved)

Greedy node placement (oracle), N to reach eps < 1e-3:

| anchor      | band (w)         | current kernels | E (this work) |
|-------------|------------------|-----------------|---------------|
| crown       | [0.125, 74.5]    | 45              | 25            |
| near_cusp   | [0.088, 46.6]    | 40              | 20            |
| two_image   | [0.027, 1046]    | (>90 log-nodes) | 40            |
| near_fold   | [0.043, 29.1]    | 53              | 24            |
| sheared_sw  | [0.046, 1389]    | (>90 log-nodes) | 42            |

The near-fold band is capped at w ~ 31 by the engine's own F005 refusal
(`CancellationError` at max_order 42) — the certified-or-refuse contract is
untouched; nodes live in the certified band by construction.

### 4.2 Production 2-decade windows, worst-case placement over the transition

Window `[0.2/delta_key, 20/delta_key]`, `delta_key = max_a |tau_a - tau_c|`
(the analogue of 506 dense frequencies over a detector band):

- Five anchors: greedy N = 19, 21, 21, 23, 24 (eps 5e-4..9.5e-4).
- 20-config scan: fold and cusp crossings at eta = +-0.002 and +-0.01 (both
  topology sides, 2 and 4 images) plus 12 random configs
  (gamma in [0.05, 0.25], kappa in [0, 0.05], |y| in [0.05, 0.9]):
  **greedy N = 19-26, every config < 1e-3.** The count is config-independent;
  N ~ 9-12 per decade of band, matching the paper's 6-11 on its 0.9-decade
  band.

### 4.3 Production node placement (no dense oracle available per candidate)

Greedy needs dense truth, so it is certification-only. Two production facts:

- **Transplanting node positions across configs fails** (donor crown ->
  random configs: eps up to 1.2e-1). Only the *count* is universal; positions
  must adapt to the config's delay scales.
- **Leave-one-out adaptive refinement** uses only node data: seed 8
  log-spaced nodes, iteratively split the intervals flanking the worst LOO
  error, stop at `max LOO < 4e-3` (calibrated: LOO overestimates the true
  spline error by ~4-16x). Result on the anchor windows:
  N = 30-44, true eps = 1.6e-4 .. 9.4e-4, all pass, self-certifying (the LOO
  statistic itself is the online error monitor). Looser stops (8e-3, 1.6e-2)
  start to fail near-fold — keep 4e-3.

### 4.4 Cost arithmetic

`F_op_grid` measured this session: 12.3 ms per 30-node batch = **0.41
ms/node** (crown window; consistent with the 0.37 ms/node contract figure).
Closed forms (images, delays, `image_kernel`, smootherstep, splines) are
microseconds. Per warm likelihood evaluation:

- greedy-oracle bound: 19-26 nodes = 8-11 ms;
- LOO production: 30-44 nodes = 12-18 ms (incremental refinement batches
  amortize slightly worse than one big batch);
- current Build-3d binding: 50-90 nodes = 20-37 ms; dense direct: 506 nodes
  = ~207 ms.

The 12 ms relaxed gate of the 3d consult is met at the low end; the 10 ms
aspirational gate needs either tuned LOO seeding (multi-scale a-priori seeds
instead of 8 log-spaced) or the paper's candidate/fiducial **ratio layer**
(`q_a`, tex Eq. slow-component-ratio) on top of this decomposition — the
ratio benefits are proposal-proximity-dependent and are deliberately *not*
claimed here.

## 5. Dead ends explored (do not re-try without new evidence)

- **Parametric tail envelopes** (fit `A_a/w^3 + B_a/w^4` at pair carriers
  from the same coarse nodes): unnecessary (greedy numbers above are without
  it) and fragile — with an O(1) unswitched background the LS fit is biased
  (near-cusp eps blew up to 1.0). If ever needed, C3/C4 belong in
  `geometry.saddle_coefficients` as closed forms, not in a fit.
- **F008 nearest-neighbour switch inside this decomposition** (variant B):
  indistinguishable from variant C on the anchors, but it lacks the
  bounded-phase guarantee (its switch scale is not the demodulation distance)
  and it stalls on accidental degeneracies; variant C is strictly better
  founded.
- **Per-image wave residual R_j** (Build-3e premise): still does not exist in
  the tree, and this construction shows it is not required — the *sum-level*
  residual E, demodulated at the critical carrier, is the smooth object.

## 6. Residual risks

1. `nearest_caustic_point` can jump between astroid folds/cusps as proposals
   move. Each evaluation is self-contained (total exact regardless of
   `tau_c`), but kernel-ratio smoothness *across* proposals inherits the
   label-continuation story unchanged. No new failure mode identified; spot
   plots across a fold-to-cusp path are a cheap build check.
2. `beta != 0` untested here (enters only via `macro_matrix` rotation);
   include one rotated anchor in the build gates.
3. Accidental `|tau_a - tau_c|` ~ 0 for a non-merging image delays that
   channel's switch (sheared_sw: 0.029). Harmless — measured: content stays
   in E with bounded phase; no accuracy or boundedness impact.
4. LOO stop constant 4e-3 is calibrated on the anchors; the build must
   re-verify it on its own grid (margin ~2x exists: true eps mostly <= 5e-4).
5. Likelihood integration choice: 5-channel form needs 15 pair summaries and
   a carrier at `tau_c`; 4-channel form needs per-frequency projection
   weights in `_gauge`. Either is a contained change; pick one, not both.
6. Node bands must respect F005 refusals (unswallowed `CancellationError` /
   `HypergeometricDomainError` at the DD ceiling `w sqrt(s) = 60`); the
   experiments hit and honored both.
7. Supersedes the F008 *switch-separation rule* (full-cluster min ->
   criticality separation `|tau_a - tau_c|`) while preserving its intent;
   needs a FINDINGS addendum and migration of the `_channel_switch`
   docstring narrative.

## 7. Build-brief-ready summary

**Mission.** Replace the flat artificial-split channel construction in
`cogwheel/lensing/chang_refsdal/channels.py` with the switched-analytic +
single-envelope decomposition (Sec. 3), expose the envelope to the RB
likelihood, and evaluate candidates from N ~ 30-44 adaptive engine nodes
instead of ~100 fixed ones.

**In scope.** `channels.py` (switch separations `|tau_a - tau_c|`; kernel
assembly `S_a H_a` + envelope; envelope accessor), `_gauge.py` (per-frequency
weights OR 5th-channel plumbing), `likelihood.py` (coarse-node engine evals,
closed-form dense reconstruction, LOO refinement loop, stop 4e-3).
**Out of scope.** `operator.py` / `_hyp1f1.py` / refusal thresholds
(byte-frozen), geometry closed forms, ratio-layer (`q_a`) speedups, sampler
integration.

**Fast in-build gates** (all run in seconds):
1. Reconstruction identity `<= 1e-13` relative on the five anchors.
2. Greedy-oracle N `<= 26` for eps < 1e-3 on each anchor's 2-decade window
   (dense truth = 506-point `exact_total`, ~0.2 s/anchor).
3. Production LOO path reaches eps < 1e-3 with N `<= 48` on all anchors.
4. `max |S_a H_a| <= 2` on fold/cusp crossings at eta = +-0.002 (both sides).
5. Deep-band check: reconstruction matches the F009 constant at the window's
   low end to < 1e-6 relative for a sheared config.

**Post-build, driver-verified.** 25-config scan (Sec. 4.2 grid), warm-lnlike
timing vs the 12 ms gate, lnlike-vs-brute-force regression on the standard
suite.
