# Build Brief: serve the saddle band where the engine is expensive or refuses

## Mission

For the macro saddle (`gamma > 1`), a measured population of prior draws is
served today by the SLOWEST path in the codebase, and about half of it cannot
be charted at all because the kernel refuses to produce a training node there.
Give that region a named serving rung.

Do NOT re-open the deltoid exterior coordinate work (`LobeExteriorChart`,
lobe-local `(rho_lobe, u)`) — it shipped in `4c7dc92` and is passing.

## Measured facts (driver, 2026-08-12, at HEAD 4e72409)

Instrument: `scripts/census_dry_run.py --n-samples 10000 --seed 42`, then
per-branch instrumentation. Structural only, no engine calls, ~1 min to
reproduce.

1. Whole-prior structural coverage is **87.61%**. Of the 1239-draw gap, 1236
   are macro-saddle. The largest single cause is 901 draws with
   `rho_lobe > rho_outer` — beyond the charted lobe-exterior shell.

2. Those draws are FAR FROM THE CAUSTIC, not near it:
   `eta` p10 0.617, p50 **0.971**, p90 1.380; `eta >= ETA_MIN_GEOMETRIC = 0.3`
   for **99.2%**. `rho_lobe` p50 5.70 is an artefact of dividing by the small
   `r_deltoid` — `|y|` p50 is only 0.732. They sit near the origin, between
   the lobes, a full unit from the caustic.

3. None of them reach the geometric arm. The saddle takes stationary phase
   ONLY when resolved AND `w > W_CEILING_SCHWINGER_QD = 150`
   (`operator.py` module docstring ~L105). This population's `w` is
   p50 28.0, p90 104.4, **max 147.5** — zero draws clear 150.

4. COST IS CONCENTRATED. Splitting by the wave-branch boundary
   `W_CEILING_SCHWINGER = 60`:

       w <= 60         564 draws   Schwinger double-double  ~0.2 s/call
       60 < w <= 150   216 draws   Schwinger MPMATH        ~85-120 s/call  (F061)

   Measured against the SERVING BUDGET, which is the denominator that
   matters — warm single-thread lnlike is 9.8 ms (ratio layer), the
   exact-engine crown is 751 ms:

       double-double     200 ms/call =     20x the ENTIRE lnlike budget
       mpmath        100,000 ms/call =  10204x the ENTIRE lnlike budget

   A single such node does not fit inside an evaluation; it IS the
   evaluation. Amortised at the measured hit rates:

       dd      5.64% x 200 ms =   11.3 ms/draw =   1.2x budget
       mpmath  2.16% x 100 s  = 2160.0 ms/draw =   220x budget

   So the mpmath band is ~190x worse and is correctly this build's target,
   but the double-double population is NOT cheap either — on its own it more
   than doubles every likelihood evaluation.

5. HALF THE EXPENSIVE POPULATION CANNOT BE CHARTED. A training node needs the
   1F1 kernel, which refuses above `_DD_PRODUCT_MARGIN = 58` (`w*|y| <= 58`).
   Over the 216: `w*|y|` p50 61.2, p90 188.8, max 394.1 —
   **48.1% under the ceiling (chartable at ~100 s/node), 51.9% over it (no
   training node can exist)**.

## Scope

IN — a named serving rung for the saddle region above the DD product ceiling,
its validity gate, its wiring into the serve path, and tests.

OUT — the deltoid exterior coordinate (shipped); extending `rho_outer` for the
`w <= 60` population — a SEPARATE build, but NOT optional: at 200 ms/call and
a 5.64% hit rate it amortises to 11.3 ms/draw against a 9.8 ms lnlike budget,
so it more than doubles every evaluation on its own; the `gamma -> 1`
degenerate-band question (recorded, deliberately open);
any training run; the cusp/fold carve-out population (326 draws, separate).

## The idea to evaluate (Professor must adjudicate before the Coder codes)

`todo.d/lensing_ppgo_extrapolation_beyond_engine_reach.md` proposes: where a
direct fit is impossible, train where the engine is CHEAP and extrapolate in
`w` with the known analytic scaling divided out. The `w`-dependence in the
resolved regime is carried by known factors (`exp(i w tau_a)` carriers,
`w^{1/6}` / `w^{-1/6}` Airy weights, `w^{1/2}` / `w^{3/4}` Pearcey control
arguments), so splining the DEMODULATED, scaling-stripped residual — flat in
`w` by construction — may extrapolate where the envelope itself cannot.

The Professor must decide:
- whether that residual is genuinely `w`-flat for the macro saddle, or whether
  the saddle's own asymptotics differ from the positive-parity case the
  existing arms were built for;
- what the VALIDITY GATE is, keyed on something that actually degrades (the
  `_airy_fold` fence is permanent precisely because a self-certificate blind
  to caustic distance read 1.2e-2 where the true error was O(1) — F028, F032,
  F033);
- whether the existing fold-ppGO handoff (`likelihood.py`,
  `_XI_FOLD_THRESHOLD = 4.0` plus a per-pair uniform error estimate) is the
  right precedent to extend, or whether the saddle needs its own.

## Acceptance

1. A NAMED rung serves a measured subset of the 216, with a gate that
   REFUSES outside its validity domain rather than degrading.
2. Accuracy certified against the exact engine INSIDE the reachable band
   (`w*|y| <= 58`), reported as an eps DISTRIBUTION (p50/p90/max) plus the
   worst-sample locus — never a bare max.
3. The error DECREASES with `w` across the certified band. This is the only
   honest basis for trusting the rung above the ceiling, and it is a
   falsifiable claim: report the trend, do not assert it.
4. `scripts/census_dry_run.py` models the new rung, and the saddle
   `exact_engine` bucket drops by the amount the rung actually claims. Report
   the per-cause breakdown (the six-way split above), not just the total.
5. Astroid (`parity == 1`) behaviour byte-identical.

## Constraints

- Branch `claude-dev`. Fast tests only; no training run; slow tiers stay empty.
- Every domain-test description MUST begin with its SHARD letter and target
  suite FILE PATH, and shards must be DISJOINT — one file per shard (F057).
  A plan was rejected at this gate on 2026-08-12 for omitting exactly this.
- The oracle for `gamma > 1` is the exact Schwinger path. `operator.F_op`
  DIVERGES for the saddle and must NOT be used.
- Do not raise `_DD_PRODUCT_MARGIN`. It is the 1F1 kernel's certified domain,
  not a tunable.
- Keep the WP count at or below 3.
