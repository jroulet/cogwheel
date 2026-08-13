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

5. **A residual 2.67% is NOT a gauge problem — it needs a CHART, and it is
   mostly chartable.** Over 300 (source, w) pairs, re-gauged `|E|/|F|` is
   p50 6.11e-06, p90 1.22e-04, p99 1.45e-02, max 2.99e-02; **8 pairs (2.67%)**
   exceed 1e-3. Since the re-gauged value is the gauge-independent floor,
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

   Under this rule the envelope decays MONOTONICALLY (1.4e-01 -> 6.1e-04
   across the band) with no oscillation, so it is a well-conditioned spline
   target. This is why tier 2 is a chart of the RE-GAUGED envelope
   specifically: in the shipped gauge the same object oscillates and is not
   splineable.

   NOTE the two tiers share ONE gauge rule, so there is no gauge
   discontinuity between them — the handover is only about whether the
   envelope is large enough to need charting.

## Scope

IN — a THREE-TIER ladder for the far-from-caustic macro saddle, with NO tier
falling through to direct evaluation:

  1. re-gauged switched analytic channels (serves ~97.3%, no chart);
  2. for sources the gate rejects, a CHART of the RE-GAUGED ENVELOPE — the
     lowest-order physics is already removed by construction, since `E` is
     what remains after the switched analytic trials are subtracted, so the
     chart target has small dynamic range and no carrier oscillation;
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
