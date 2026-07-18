# Build 3f — SACR-C: switched-analytic + single-envelope channels (the beat-free decomposition)

## Mission

Implement the Professor's numerically-certified SACR-C decomposition
(research report: `.claude/handoff/lensing/envelope_research.md` — READ
IT FIRST; it is the design authority for this build). Replace the flat
artificial-split channel construction with: persistent resolved images
carried by the ANALYTIC saddle kernel `geometry.image_kernel` under
their own carriers with smootherstep weights `S_a(w) =
smootherstep(w*|tau_a - tau_c|, 0.5, 4)`, plus ONE interpolated smooth
envelope `E(w) = e^{-i w tau_c} (F - sum_a S_a H_a e^{i w tau_a})`
demodulated at the parked critical-carrier delay `tau_c` (from
`nearest_caustic_point`), computed at coarse nodes from `F_op_grid`.
Evaluate candidates from N ~ 30-44 LOO-adaptive engine nodes instead of
~100 fixed. Exactness rides the existing `_gauge` telescoping algebra
(identity verified 2e-16 in the report's experiments).

## Measured facts (Professor research, ALL actually run — do not re-derive)

1. Beats are impossible in E by construction: the switch scale IS the
   demodulation distance, so all O(1) content in E carries <= 4 rad of
   demodulated phase (report Sec. 3, the key theorem).
2. Certified node counts, null-safe eps < 1e-3, 2-decade production
   windows, 25 configs (5 anchors + fold/cusp crossings both sides +
   12 random): greedy-oracle N = 19-26 (CONFIG-INDEPENDENT); production
   LOO-adaptive placement (stop 4e-3) N = 30-44, self-certifying.
   Control under the SAME oracle placement: current kernels need 40-53
   (Build 3d measured 50-90 in production).
3. Cost: ~0.41 ms/coarse node (F_op_grid) => ~12-18 ms/eval projected
   (oracle bound 8-11 ms) vs 20-37 ms today. The remaining path to the
   owner's 10 ms is the (now-trivial) surrogate of the SINGLE smooth
   envelope — named residual lever, NOT this build.
4. Merging images: max|S_a H_a| <= 1.3 measured through eta = +-0.002
   fold/cusp crossings — the S_a gate is MORE conservative than F008's
   switch where images merge; accidental delay degeneracies (the
   crown's actual disease) no longer stall convergence.
5. The Build-3e per-image wave residual R_j does not exist and is NOT
   needed. The paper prototype's partition is block-structured
   (persistent images analytic; only the cluster residual split among
   cluster channels, demodulated at the critical-point delay) — SACR-C
   is that scheme in the current engine's language.
6. Dead ends documented in the report (do not retry): parametric tail
   fits; node transplanting between configs.

## Scope (from the report's build-ready section)

IN: `cogwheel/lensing/chang_refsdal/channels.py` (switch separations
`|tau_a - tau_c|`; kernel assembly `S_a H_a` + envelope; an envelope
accessor for the likelihood), `_gauge.py` (per-frequency weights OR
5th-channel plumbing — the report discusses both; Architect picks),
`cogwheel/lensing/likelihood.py` (coarse-node engine evals, closed-form
dense reconstruction, LOO refinement loop with stop 4e-3), tests via
`domain_test_descriptions`.

OUT (byte-frozen / untouched): `operator.py` and `_hyp1f1.py`
internals and EVERY refusal threshold/message (F_op_grid is consumed
as-is); `geometry.py` closed forms (`image_kernel`,
`nearest_caustic_point` are consumed as-is); the ratio-layer (`q_a`)
speedups; sampler integration (Build 4); the stall-ringdown/template
builders; NO tolerance widening anywhere.

## Constraints

- `exact_total` and `lnlike_bruteforce` remain the untouched oracles;
  refusals (geometry.LensDomainError, operator.CancellationError)
  propagate unswallowed and symmetrically on RB and brute paths.
- The F008 full-cluster rule is superseded ONLY where the report says
  the S_a gate is provably more conservative — the crossing-scenario /
  label-continuity tests must stay green (F002 fixture independence:
  scenario builders from geometry/operator/_gauge only, never
  channels.py).
- F001: carrier phases reduced mod 2*pi before complex exp where
  w*delay reaches large values; F009: the deep-unresolved limit must
  reproduce the exact macro constant.
- F010: py_func-chain falsification for any new njit (the report's
  recipe needs none — reconstruction is vectorized numpy).
- In-build gates FAST (seconds each, from the report): (1)
  reconstruction identity <= 1e-13 relative on the five anchors; (2)
  greedy-oracle N <= 26 for eps < 1e-3 on each anchor's 2-decade
  window (dense truth = 506-point exact_total, ~0.2 s/anchor); (3)
  production LOO path reaches eps < 1e-3 with N <= 48 on all anchors;
  (4) max|S_a H_a| <= 2 on fold/cusp crossings at eta = +-0.002 both
  sides; (5) deep-band: reconstruction matches the F009 constant at
  the window's low end to < 1e-6 relative on a sheared config. PLUS
  the existing suite's crown anchors (RB-vs-brute at original gates,
  near-cusp pin, zero-noise floors) green unchanged.
- Timing gate: structural-first (node count <= 48 across anchors,
  config-independent; public-entry-point speedup vs brute), absolute
  ceiling arithmetic-derived at 18 ms warm pinned best-of-5 (report's
  projected upper bound; the 10 ms program target is NOT this build's
  gate — its named finisher is the envelope surrogate).

## Acceptance (build-level)

- All five report gates + existing suite green at ORIGINAL tolerances;
  commit lands hook-clean (SPEC.md updated for the new channel
  construction + envelope accessor; fragments rendered).
- Post-build (driver): 25-config scan per the report's Sec. 4.2 grid,
  warm-lnlike timing, full suite minus XODE trio — detached.

## Environment facts

- Interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server nereid; SSE port 8323 via .env). HEAD 2a69e74 (post-3e-abort;
  amended): 222 tests green in 44 s at -n4.
- The report's experiment scripts (envelope_exp1..6.py) are in the
  driver session scratchpad — facts from them are inlined above; the
  report itself is the durable record.
- numba 0.58.1, mpmath 1.3.0, pytest-xdist 3.8.0.
