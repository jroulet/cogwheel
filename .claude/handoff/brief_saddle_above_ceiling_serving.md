# Build Brief: serve the far-from-caustic macro saddle from the analytic channels

## Mission

For the macro saddle (`gamma > 1`), a measured 1236/10000 prior draws reach no
serving rung and fall through to direct evaluation. Production must never
reach direct evaluation, so these are UNSERVED, not merely slow.

Measurement (below) shows the largest sub-population needs NO chart: the
SACR-C transition envelope is negligible there, so the switched analytic
channels alone reproduce the total. Wire that as a named rung, with the gauge
choice that makes it hold.

## Measured facts (driver, 2026-08-13, at HEAD 1d7487a)

Reproduce with `scripts/census_dry_run.py --n-samples 10000 --seed 42` plus
the per-branch instrumentation in
`todo.d/lensing_saddle_coverage_gap_breakdown.md`.

1. **The envelope is negligible at moderate-to-high `w`.** `|E| / |F_total|`
   over draws sampled from the real gap population:

       w = 24   p50 3.60e-05   p90 1.47e-04    93.3% below 1e-3
       w = 40   p50 4.98e-06   p90 2.76e-05    93.3% below 1e-3
       w = 58   p50 2.97e-06   p90 1.17e-05    96.7% below 1e-3

2. **The exceptions are a GAUGE artifact, not physics.** The excluded cases
   are FARTHER from the caustic than the served ones (`eta` p50 1.169 vs
   0.992), 0% sit inside the Airy fold fence, and `r(eta, delta_min) = -0.118`
   — uncorrelated. They are not merging images. `delta_min = |tau_a - tau_c|`
   compares a real image's delay to the CRITICAL CARRIER, which far from the
   caustic is a virtual reference where no image sits. The switch
   `S_a = smootherstep(w*|tau_a - tau_c|, 0.5, 4)` then turns OFF and hands
   that channel to the envelope by construction. The failure threshold
   matching `RHO_END = 4.0` exactly is the tell.

3. **The SACR-C identity, verified to 3.8e-16** from a partition's own fields:

       envelope = conj(exp(1j*w*tau_c)) * (F - sum_a exp(1j*w*tau_a) * S_a * SK_a)

   where `SK` is `partition.saddle_kernels`. NOT `partition.kernels` — those
   already have the envelope apportioned back in with weights
   `1 - S_a + _ENVELOPE_WEIGHT_FLOOR` (`_gauge.py`).

4. **Re-gauging works, and it is a TAIL fix.** Choosing `tau_c` to saturate
   every switch, on the worst measured case (`gamma=1.5859`,
   `y=(-1.1208,-0.9002)`, `w=58`): `|E|/|F|` 4.165e-01 -> 5.126e-04, an 813x
   improvement. The value PLATEAUS for every `tau_c` that saturates the
   switches — once saturated the residual is fixed and only a phase changes,
   which cannot alter a magnitude — so 5.13e-04 is that source's
   gauge-independent intrinsic error.

   Across the whole excluded population (`w*delta_min < 16`, n = 37):

                      p50        p90        MAX      frac < 1e-3
       shipped     4.53e-05   1.31e-02   6.71e-01      81.1%
       re-gauged   3.55e-05   2.45e-04   9.65e-03      97.3%

   MEDIAN IMPROVEMENT IS 1x. Most of that population was already fine; what
   re-gauging fixes is the TAIL (p90 53x, max 70x, coverage 81% -> 97%).

5. **The residual is NOT a gauge problem — it needs a CHART, and it is mostly
   chartable.** Over 300 (source, w) pairs probed at `w in {24, 58}`,
   re-gauged `|E|/|F|` is p50 6.11e-06, p90 1.22e-04, p99 1.45e-02,
   max 2.99e-02; 8 pairs (2.67%) exceed 1e-3.

   THAT 2.67% IS A PER-PROBE-FREQUENCY FIGURE, NOT THE TIER-2 SIZE. It shares
   the sampling flaw corrected in fact 8: `w in {24, 58}` oversamples high
   frequencies against the population's true `LogU(5, 148)`. The tier-2 size
   over the real `w` distribution is **~23%** (fact 8). Use that number for
   sizing; the distribution shape below is still valid, since it describes
   WHICH sources fail, not how many. Since the re-gauged value is the gauge-independent floor,
   their analytic trials genuinely fail to reproduce `F`, so the envelope must
   be carried — i.e. splined, exactly as every other chart does.

   That residual is a DISTINCT REGION, not scattered noise:

       residual  eta p50 0.426 (range 0.233-1.037)   |y| p50 1.797   gamma p50 1.486
       served    eta p50 0.984                        |y| p50 0.915   gamma p50 1.369

   Closer to the caustic AND farther out — the deltoid's outer edge.

   Chartability (`w*|y| <= _DD_PRODUCT_MARGIN = 58` for a training node):

       w*|y|  p50 44.4   min 30.6   max 103.0
       chartable: 75% (6 of 8)
       w = 24: n=7, 85.7% chartable      w = 58: n=1, 0% chartable

   The two constraints pull APART in our favour: the envelope shrinks with `w`
   (p50 3.6e-5 at w=24 -> 3.0e-6 at w=58), so the population that still needs
   it thins exactly where charting gets hard. 7 of the 8 sit at `w = 24`.

6. **A ~0.7% sliver is still open.** 2 of 300 pairs both exceed the bar after
   re-gauging AND have `w*|y| > 58`, so no training node can exist for them
   and no gauge choice fixes them. That is the honest residual of this whole
   approach. It is OUT of scope: it must be REFUSED by name, not served
   wrongly, and not used to justify stretching tier 1 or tier 2.

7. **THE GAUGE RULE IS DETERMINED, and it makes the envelope splineable.**
   `tau_c` must be `w`-INDEPENDENT or each node of a chart's `w` axis stores a
   differently-gauged object. Three candidates measured on a residual-region
   source (`gamma=1.486`, `|y|=1.798`, real delays `[0, 0.2509]`, band
   `w in [12, 58]`), reporting sign changes and `max|2nd diff| / max|v|` —
   the quantities a spline actually cares about:

       per-w minimisation of |E|   6-7 sign changes   2nd diff 1.09 / 1.45
       tau_c = image midpoint      5   sign changes   2nd diff 0.111 / 0.250
       tau_c OUTSIDE the cluster   0   sign changes   2nd diff 0.023 / 0.022

   THE RULE: place `tau_c` outside the real-image cluster, at distance
   `RHO_END / w_min` from the nearest image, where `w_min` is the chart
   band's LOWEST frequency:

       tau_c = min_a(tau_a) - RHO_END / w_min

   `w_min` binds because the switch argument `w*|tau_a - tau_c|` is smallest
   there; satisfying it at `w_min` saturates every switch across the whole
   band (measured 4.0 -> 19.3 for the band above). A `tau_c` BETWEEN the
   images cannot work when the pair is closer than `RHO_END / w_min` — here
   the images are 0.251 apart and 0.333 is needed — which is exactly the case
   the midpoint gauge fails.

   WHAT THE RULE BUYS: switch saturation across the band, which is what makes
   tier 1 work (fact 8). It does NOT by itself give a smooth envelope.

   THE RESIDUAL OSCILLATION IS THE GAUGE'S OWN PHASE WINDING, NOT PHYSICS.
   Tested and REFUTED first: that it is the image-pair beat. Predicted
   crossings `(w_max - w_min)*|dtau|/pi` span 12 to 1204 across sources while
   the observed count sits at 4-5 for every one of them — off by up to 240x.

   The observed count is instead set by the DEMODULATION OFFSET. With
   `tau_c = tau_min - RHO_END/w_min` the offset is 0.3333 for every source, so
   `E` rotates at that rate: `(58-12)*0.3333/pi = 4.9` crossings predicted,
   4-5 observed, for all of them. Pushing `tau_c` far enough to saturate the
   switches is exactly what makes `E` wind.

   DECOUPLE THE TWO ROLES. `tau_c` plays two independent parts — it sets the
   switch argument AND it demodulates `E`. The identity holds for any
   demodulation phase, so use different values:

       tau_switch = tau_min - RHO_END / w_min     (far: saturates every switch)
       tau_phase  = tau_min                       (near: minimal winding)

   Measured over 6 sources, sign changes in (Re, Im) and max|2nd diff|/max|v|:

       coupled     4-5 / 4-5 changes    2nd diff 0.18 - 0.91
       decoupled   0 / 0 on 5 of 6      2nd diff 0.06 - 0.22

   The build must store `tau_switch` and `tau_phase` (or `tau_min` plus the
   band's `w_min`, from which both derive) so the served gauge reproduces the
   trained one exactly.

   CAVEAT ON THIS MEASUREMENT, do not skip: those 6 sources were selected by
   `|y| > 1.4` as a proxy for the tier-2 region. That is NOT the tier-2
   definition, which is `|E|/|F| > 1e-3`. The one source that did not improve
   has `|E| p50 = 4.4e-06` — it is a TIER 1 source whose smoothness never
   matters. Before sizing tier 2's `w` axis, re-measure smoothness on sources
   selected by the ACTUAL tier-2 criterion.

   NOTE the two tiers share ONE gauge rule, so there is no gauge
   discontinuity between them — the handover is only about whether the
   envelope is large enough to need charting.

8. **THE RULE COSTS NOTHING vs the (invalid) per-`w` optimum, and TIER SIZES
   DEPEND ON THE `w` DISTRIBUTION.** Over 220 (source, w) pairs probed at
   fixed `w in {24, 58}`:

       per-w minimisation (NOT a valid gauge)   p50 8.48e-06  p90 1.11e-04  96.8% < 1e-3
       tau_c = min(tau) - RHO_END/w_min         p50 8.48e-06  p90 1.20e-04  96.4% < 1e-3

   The implementable `w`-independent rule is essentially optimal — that
   comparison stands.

   BUT 96.4% IS NOT THE TIER-1 SIZE. Those probe frequencies oversample high
   `w`; the population's own `w` is `LogU(5, 148)` and reaches down to 5,
   where the envelope is NOT negligible. Re-measured with EACH SOURCE'S OWN
   `w`:

       w <= 60        n=65   p50 1.12e-04   p90 4.53e-01   tier 1 serves  69.2%
       60 < w <= 148  n=25   p50 1.97e-06   p90 1.98e-05   tier 1 serves  96.0%

   Weighted over the beyond-shell population: **tier 1 ~77%, tier 2 ~23%** —
   tier 2 is SIX TIMES the 3.6% an earlier draft of this brief claimed. Size
   the work accordingly: the chart is a substantial part of this build, not a
   corner case.

   Note the inversion vs cost intuition: the EXPENSIVE mpmath band
   (`60 < w <= 148`) is the part tier 1 serves almost completely (96%), while
   the cheap-to-evaluate `w <= 60` band is where the chart is actually needed.

9. **THE `w` AXIS DOES NOT NEED A COORDINATE CHANGE — `log w` is near-optimal.**
   F064 records the hard-won positive-parity lesson: minimising oscillation was
   not enough, and only a COORDINATE CHANGE (`u = d^(2/3)`, worth 171x) made
   the structure splineable. Tested here on a genuine tier-2 source
   (`gamma=1.535`, `|y|=1.499`, `|E|/|F|` max 2.10), as spline interpolation
   error `max|spline - truth| / max|E|`:

       coordinate      n=8        n=12       n=16
       log w        5.34e-04   1.02e-04   2.58e-05
       w            2.11e-02   7.13e-03   2.89e-03     40-100x worse
       1/w          9.90e-04   2.83e-04   1.27e-04
       w^-1/2       2.14e-04   6.06e-05   1.26e-05     best, ~2x over log w

   The axis choice MATTERS — linear `w` is badly wrong — but `log w`, which
   the charts already use, is within ~2x of the best candidate found. That is
   nothing like the 171x of the positive-parity angular fix, so do NOT spend a
   work package inventing a `w` coordinate.

   WHY the pathology is absent here: F064's disease is a NORMALISING
   DENOMINATOR dragging a cusp's `theta^(2/3)` across every radius. Nothing
   analogous acts in `w` — the re-gauged residual is a smooth asymptotic
   decay. The axes that COULD carry that pathology in this region are the
   SPATIAL ones, and they already use the lobe-local cusp-adapted `u`.

   Note `1/w` — the coordinate an asymptotic-series argument predicts — is
   WORSE than `log w`. The measurement, not the argument, settled it.

   LIMIT: one source. If tier 2's chart misses its eps bar, re-run this
   comparison across the tier-2 population before adding nodes.

10. **TIER 2's NODE BUDGET IS ORDINARY — the 8-node seed suffices.** Selecting
    by the ACTUAL tier-2 criterion (`|E|/|F| > 1e-3` under the decoupled
    gauge, not the `|y|` proxy of fact 7): 3 of 70 scanned gap sources qualify
    at `w = 24`. Cubic-spline interpolation error in `log w`, as
    `max|spline - truth| / max|E|`:

        gamma  |y|    |E|/|F|     n=8       n=12      n=16      n=24
        1.519  1.787  8.70e-02  8.51e-04  1.69e-04  2.56e-05  9.36e-06
        1.585  1.851  3.56e-02  9.69e-04  1.00e-04  4.37e-05  1.82e-05
        1.518  1.559  1.83e-03  1.39e-02  7.57e-03  1.63e-03  4.63e-04

    Those are errors in `E`. The SERVED error is that times `|E|/|F|`, and at
    `n = 8` it is 7.4e-05, 3.4e-05 and 2.5e-05 respectively — all ~1e-5,
    two orders under the 1e-3 bar.

    The envelope's own smallness suppresses its interpolation error in the
    served quantity, so the residual oscillation of fact 7 costs almost
    nothing. Do NOT spend a work package enlarging the `w` axis; the standard
    `_LOO_SEED_NODES = 8` with the usual LOO-adaptive stop is enough, and the
    stop will terminate early here.

    Note the tier-2 population is SMALL and its members are strongly
    envelope-weighted (`|E|/|F|` up to 8.7e-02), so tier 2 is a narrow chart
    over a well-conditioned target, not a broad one.

## Scope

IN — a THREE-TIER ladder for the far-from-caustic macro saddle, with NO tier
falling through to direct evaluation:

  1. re-gauged switched analytic channels (serves ~77% of the beyond-shell
     population, no chart);
  2. for sources the gate rejects, a CHART of the RE-GAUGED ENVELOPE — the
     lowest-order physics is already removed by construction, since `E` is
     what remains after the switched analytic trials are subtracted, so the
     chart target has small dynamic range, and fact 10 measures its node
     budget as ORDINARY (the existing 8-node seed suffices);
  3. a named refusal ONLY for what neither tier can reach (see below).

Plus the validity gate, the serve-path and census wiring, and tests.

OUT — the ~0.7% sliver of fact 6 (2 of 300 pairs, `w*|y| > 58` AND needing the
envelope: unchartable by construction, still open — do NOT invent a rung for
it here, and do NOT let it justify widening any other tier); the
`w <= 60` chartable population (separate, and NOT optional — see
`todo.d/lensing_saddle_gap_is_a_routing_failure_not_coverage.md`); the
`gamma -> 1` degenerate-band question (recorded); the cusp/fold carve-out
population (326 draws, belongs to the uniform arms); any training run.

## Design points the Architect must settle

These are DESIGN decisions. The measurements above are DONE — do not
re-derive them and do not send agents to re-measure them.

- **Where the gauge choice lives.** `tau_c` must be a pure function of the
  source, computed identically at train and serve time. A gauge chosen one
  way when the envelope is stored and another when it is served is a
  train/serve skew — the bug class already recorded for the ghost gate. Name
  the single authoritative home.
- **Regional handover.** Near the caustic the critical-point `tau_c` is
  CORRECT: the merging pair's delays converge to it and the envelope handles
  them smoothly. This is therefore a REGIONAL gauge switch keyed on distance
  from the caustic, and the handover needs a continuity requirement stated as
  a test.
- **The gate.** It must key on something that degrades with the served
  quantity. `eta` alone does NOT (fact 2). The measured discriminator is
  post-re-gauge switch saturation, `min_a w*|tau_a - tau_c| >= RHO_END` under
  the CHOSEN gauge. State how a source that cannot be re-gauged into
  saturation is refused.
- **Refusal, not degradation.** A source failing the gate falls to the next
  ladder rung by a NAMED refusal. The `_airy_fold` fence is permanent
  precisely because a self-certificate blind to what degrades it read 1.2e-2
  where the true error was O(1) (F028, confirmed against GLoW in F032; no
  amplitude refinement removes it, F033).

## Acceptance

1. A named rung serves far-from-caustic saddle sources from the analytic
   channels: no chart, no engine call, no fall-through to exact.
2. Accuracy vs the exact Schwinger engine reported as a DISTRIBUTION
   (p50/p90/max) with the worst-sample locus — never a bare max. Bar: 1e-3.
3. The gate REFUSES the residual population rather than serving it wrongly; a
   test drives a known-bad source and asserts the refusal.
4. Train/serve gauge identity: a test asserts the stored and served `tau_c`
   agree for the same source.
5. `scripts/census_dry_run.py` models the rung; the saddle `exact_engine`
   bucket drops by what the rung claims, reported against the six-way
   per-cause breakdown, not just the total.
6. Astroid (`parity == 1`) byte-identical.

## Constraints

- Branch `claude-dev`. Fast tests only; no training run; slow tiers stay empty.
- Every domain-test description MUST begin with its SHARD letter and target
  suite FILE PATH; shards DISJOINT, one file per shard (F057).
- The oracle for `gamma > 1` is the exact Schwinger path. `operator.F_op`
  DIVERGES for the saddle and must NOT be used.
- `w <= 60` keeps the engine cheap (~0.2 s/call); `60 < w <= 148` is the
  mpmath band at ~85-120 s/call (F061). Certify in the cheap band.
- Keep the WP count at or below 3.
