# Build 3d — Kill the 1F1-ladder cost: finish the factorization, surrogate only if needed (10 ms)

## Mission

Take `LensedRelativeBinningLikelihood.lnlike` from the current 41 ms/eval
(warm, pinned single-thread, post-Build-3c) to AT OR UNDER 10 ms — the
owner's standing requirement. The remaining cost is ~85% one thing: the
exact 1F1 derivative ladder evaluated per w-node (~0.37 ms/node x ~100
nodes ~= 35 ms; sequential dd recurrence, NOT batchable). TWO levers,
in order of preference (owner directive 2026-07-18: the `h_L = F*h_UL`
factorization is not yet exhausted — do not reach for the heavy table
while the cheap structure remains unused):

A. **Finish the factorization: kink-aware interpolation, ~10x fewer
   nodes.** The paper's prototype needed only ~6-11 global kernel
   nodes; Build 3b needs ~100 because a global cubic spline on a
   log-grid fights the C2-only kinks — whose LOCATIONS AND FUNCTIONAL
   FORM ARE KNOWN ANALYTICALLY (the smootherstep gauge hand-over at
   RHO_START/delta and RHO_END/delta over full-cluster separations, and
   the branch switch). Exploit that knowledge: peel the analytic
   switch/kink factor before interpolating (interpolate only the
   genuinely smooth remainder, multiply back exactly), and/or segment
   the band at the known kink frequencies with per-segment
   Chebyshev/spline. Target the paper's node regime. Arithmetic: at
   ~0.37 ms/node, ~12 nodes => ladder ~4.5 ms => lnlike ~7-9 ms with NO
   table, fully per-event (no lens-mass domain problem, no cache).
   IMPORTANT RANKING CORRECTION the plan must absorb: the 3c plan
   dismissed node reduction ("saves <0.5 ms") using its ~70 ms
   non-engine estimate, which measurement falsified (~2.5 ms). With the
   true split, node count multiplies the DOMINANT remaining cost — this
   lever is high-leverage, not negligible. The 3b production interp
   gate (null-safe 1e-3 on the production grid) must hold on whatever
   grid ships; the gate moves to the new interpolation scheme's
   production configuration, tolerance UNCHANGED.
B. **The 3D post-contraction surrogate** (design below) — take it ONLY
   for whatever gap A leaves, or if the Professor shows A cannot hold
   the accuracy gate at a node count that reaches 10 ms. B carries the
   research-grade burdens (global domain, cache, provenance) that A
   avoids entirely.

If the honest combined arithmetic still cannot reach 10 ms, follow the
step rule (gate at the plan's own predicted floor with the derivation,
name the residual lever) — the burden of proof is on the arithmetic.

## Settled facts (do not re-derive)

1. Post-3c cost split (measured, crown config, pinned): lnlike 41.1 ms =
   engine 38.6 (of which 1F1 ladder ~35, weight-vector contraction ~2,
   caustic ~1.9) + non-engine ~2.5 (data/norm 2.5 dominates it).
2. The surrogate target is the POST-contraction output — the smooth O(1)
   per-image channel kernels `K_a` (equivalently the reconstructed `F`)
   — NEVER the internal ~85-output derivative ladder (that 2D table was
   REJECTED for cause: ~100 decades of dynamic range through a
   cancellation regime; the rejection does NOT apply to the O(1)
   post-contraction kernels).
3. Reduced domain: kappa eliminated EXACTLY by the mass-sheet identity,
   beta by rotation into the shear eigenframe (locked design decision 5)
   => 3 real dimensions (w > 0, y' >= 0 in the shear frame, gamma').
   The reduction must be VERIFIED against the engine at build time (a
   dedicated test), never assumed — a wrong reduction bakes a bias into
   every table cell.
4. DOMAIN CONSTRAINT — applies ONLY if lever B (the table) is taken:
   `w = xi(M_L, z_L) * f` — the per-event w-grid MOVES with the sampled
   lens mass and redshift, so a global table's w-dimension must cover
   the full w range samplers will visit (log-w spanning the certified
   band up to `w <= 500`, `w*sqrt(s) <= 60`), not one event's node
   grid. Lever A is per-event and has no such constraint — one of the
   reasons it is preferred.
5. Fallback contract (from the archived 3c launch-1 plan — reuse these
   design elements): per-topology-regime domain boxing with the
   transition-frequency surfaces the full-cluster machinery already
   locates; guard bands around caustics/transitions; refused-cell masks
   (any oracle point that raises `CancellationError`/`LensDomainError`
   marks its cells REFUSED); in-domain masks returned to the caller;
   out-of-box / guard-band / refused nodes are evaluated by the EXACT
   batched engine (`F_op_grid` path); engine refusals propagate
   unswallowed and symmetrically on RB and brute paths. Correctness is
   therefore guaranteed independent of surrogate coverage — only timing
   depends on coverage, and timing is gated.
6. The oracle for table construction AND certification is the exact
   engine (now 7x faster after 3c — table builds are cheap) and mpmath
   at the boundary; F002: never the surrogate itself; enforce oracle
   independence with the AST-guard idiom.
7. Tables are a LAZILY-BUILT, VERSIONED CACHE (the coherent-score
   `LookupTable` idiom already in cogwheel): engine-version provenance
   hash, rebuilt on mismatch, stored under a gitignored cache path —
   NEVER committed (memory budget well under ~250 MB). A DATA_CONTRACTS
   artifact entry (+ contracts_changelog fragment) is required for the
   new cached product.
8. Surrogate accuracy tolerance (Professor, 3c launch-1): rel ~1e-6
   in-domain vs the exact engine (>=100x below the noise-limited
   likelihood precision); end-to-end |delta lnlike| < 0.01 nats vs the
   exact path across ~1000 parameter points including fallback-
   straddling ones. Explicit gate with provenance.

## Scope fences

IN: new `cogwheel/lensing/chang_refsdal/_surrogate.py` (or equivalently
placed module); dispatch wiring in `channels.py` /
`cogwheel/lensing/likelihood.py` (thin — no new physics in the
likelihood layer); the cache path + provenance machinery; tests via
`domain_test_descriptions`; DATA_CONTRACTS artifact + fragments.

OUT (do not touch): every refusal threshold and message; `F_op_grid` /
`operator.py` semantics (it is the fallback and the oracle — leave it
alone); `_dd.py`/`_gauge.py`/`geometry.py`; the 3b node-grid accuracy
machinery and its gates; the stall-ringdown/template builders;
priors/sampling/folding (Build 4); NO tolerance widening anywhere.

## Constraints

- All existing gates green at ORIGINAL tolerances (production interp
  null-safe 1e-3, RB-vs-brute max(1.5, 1e-2|bf|) on every config, numba
  preservation, macro limit 7.85e-9, near-cusp pin, certification
  battery, refusal symmetry).
- No silent extrapolation, ever: outside the box or in a guard band the
  surrogate returns not-in-domain and the exact engine serves the node.
- Zero false accepts: the surrogate must never return a value where the
  oracle refuses (test with refused-region probes).
- F010: any new njit code keeps py_func-chain-reachable falsification;
  self-falsification tests must demonstrably go RED under seeded
  perturbations.
- Timing: warm, best-of-N, threads pinned to 1. In-build tests FAST
  (table build for the test fixture's regimes only if full-domain build
  is slow — but the shipped lazy-cache must cover the certified domain;
  say in the plan how build time is bounded).
- In-build tests: minutes, not hours. Full suite = driver post-build.

## Acceptance (build-level)

- In-build: warm pinned best-of-5 `lnlike <= 10 ms` on the crown
  4-image config with the surrogate serving in-domain nodes (HARD,
  subject only to the step rule above); surrogate-vs-oracle rel 1e-6
  in-domain on off-grid probe sets per regime; zero false accepts;
  reduction-exactness test green; end-to-end |delta lnlike| < 0.01 nats
  vs the exact path (sampled points incl. fallback-straddlers); every
  existing gate green at original tolerances; cache provenance test
  (stale-version table is rebuilt, never silently reused).
- Commit lands hook-clean: SPEC.md + DATA_CONTRACTS.yaml (+ fragments,
  rendered) reflect the new module and cached artifact; no table file
  in git.
- Post-build (driver): full suite minus XODE trio green, detached.

## Environment facts

- Interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server nereid; SDK SSE port 8323 via .env).
- HEAD 37c760f: full suite minus XODE trio 222 passed in 44 s at -n4;
  fast-path suite green; batched-operator suite green.
- numba 0.58.1, mpmath 1.3.0, pytest-xdist 3.8.0, scipy present.
