# Build 8f — Serving micro-levers: ~2x floor, parallel exact, Pearcey table, guarded L_MAX

## Mission

Close the performance program: take the served path from 4.1x the
unlensed floor toward ~2x, multiply exact-evaluation throughput for
the training campaign, make cusp serving genuinely ms-scale, and
graduate the measured-only geometric-gate relaxation. Five levers,
each independently certified, value-preserving in the 8b-levers mold
(HEAD side-by-side where bytes are claimed; physics tolerances
untouched; F010 reachable-red falsifications; F017 gauge discipline).

1. **geometry_partition residual (~2.0 ms).** Post-8b the caustic
   search is 0.24 ms; the residual is the quartic image solve, delays,
   kernels, switch. WP-FIRST: profile the split (the 8e census-first
   pattern), then optimize the dominant term only. Value-preserving to
   1e-10; branch decisions byte-identical.
2. **Likelihood contraction overhead (~2.3 ms).** The data/norm
   moment contraction. Profile-first, then the dominant term. Same
   preservation contract.
3. **Node-parallel exact evaluation (owner-spotted).** The
   per-node Schwinger loops in `_positive_parity_grid`/`_saddle_grid`
   are serial over independent nodes (~90 ms/node). Parallelize
   across nodes: either an njit prange driver returning
   (values, certificate_flags) with the NAMED raise hoisted to the
   Python wrapper (the refusal exception must never cross a thread
   boundary; the refusal CLASS and message contract are unchanged),
   or a thread pool over a `nogil=True` call chain — Coder verifies
   which the numba version supports cleanly and audits the chain for
   shared mutable state. CERTIFICATION: per-node byte identity vs the
   serial path across a both-parity sweep; refusal-decision identity;
   deterministic ordering. Payoff: the training campaign and the
   brute tier divide by ~core count.
4. **Universal Pearcey table (REQUIRED — measured 45 ms/call warm).**
   `P(x, y)` is lens-independent. Ship a precomputed table artifact
   (2-D not-a-knot spline over the certified quadrature on a bounded
   (x, y) box; the existing large-argument asymptotics take over
   outside), REGISTERED as a data product (DATA_CONTRACTS +
   data_registry + LOADERS, the 8c pattern) with a regeneration
   script; loader verifies a stored hash. The arm consults the table
   first and falls back to live certified quadrature outside the box
   or on any load anomaly — the never-serve-where-wrong contract
   keeps its live-certificate backstop. Target: <= 50 us/call served
   from the table, certified vs the quadrature at 1e-8 on held-out
   points.
5. **L_MAX relaxation, guarded (owner-approved graduation).** The 8d
   headroom audit + census fraction (b) (13.9% of high-w nodes) fund
   relaxing the positive-parity cancellation gate from 48 toward the
   audit's certified floor with the MANDATORY guards: image-count
   match vs the quartic solve, parity-sum (Morse) check, and a 1.5x
   safety margin on the measured worst-config 1e-4 crossing. The
   relaxed threshold is a named constant with its provenance; the
   audit test graduates from documentation to enforcement (asserts
   the shipped constant respects the measured floor + margin).

## Measured facts (pre-answered — do not re-derive)

- Floor ledger (post-8b, corrected generic-proposal protocol):
  unlensed 1.56 ms; served lensed 6.37 ms (4.1x); budget: partition
  ~2.0 + envelope 0.35 + reconstruct 0.11 + contraction ~2.3 ms.
- Exact path ~90 ms/node (Schwinger, dd); crown lnlike ~751 ms
  default; brute ~138 s/call. Caustic search 0.24 ms warm.
- Pearcey primitive 45 ms/call WARM (certified rotated-contour,
  measured 2026-07-21); Airy 3.4 us. The 45 ms is certification-node
  count, not asymptotic-regime cost — the table amortizes it once.
- Census (1e5 draws): corner = 24.6% of draws; high-w nodes 31.2%
  geometric-now / 13.9% relaxed-L_MAX (b) / 54.9% arms-or-hard;
  fold:cusp topology 3:1; arms serve ~45% of fold-type cd nodes at
  the strong/saddle bar. Hard core concentrates at small arguments.
- Test tiers are LAW; exact-heavy/brute tests born gated. The
  tree-wide fast gate now runs automatically as a COMMIT PRECONDITION
  (SDK) — in-build test specs must keep the fast tier fast.
- This build is the PROVING RUN for two fresh SDK fixes: the
  revision-loop test-dev route and the tree-gate commit preflight.

## Out of scope — hard fences

- NO changes to `_schwinger`/`_dd`/`_hyp1f1` numerical internals
  (lever 3 restructures the CALLING loop only; the evaluator bodies
  are untouched).
- NO surrogate retraining; NO full-box training (it rides AFTER this
  build, on the node-parallel engine). NO sampling/PP. NO
  enable-by-default changes.
- NO quad-double work (separate owner decision, pending the
  hard-core w-distribution measurement).
- Refusal vocabulary unchanged; no new exception classes; every
  refusal stays certified-or-refuse.

## Acceptance (two-tier)

1. In-build (FAST): profile reports for levers 1-2 with the optimized
   dominant terms certified value-preserving; node-parallel path
   byte-identical to serial per node with refusal-decision identity
   and a measured speedup factor on a small grid; Pearcey table
   registered + certified at 1e-8 vs the quadrature on held-out
   points with the live-quadrature fallback exercised both directions
   (F010 mutation on the stored hash must fall back, not serve);
   L_MAX ships at the guarded value with the enforcement test green;
   fast tier stays fast.
2. POST-BUILD (driver): floor ledger re-measured (target ~2x);
   training-throughput probe (nodes/sec serial vs parallel); full
   sweep under the flag; then the full-box training campaign on the
   final engine.
