# Build 3b — Close Build 3: production-accurate node grid + the two measured hot spots

## Mission

Build 3 delivered a 215x lnlike speedup (14.79 s -> ~70 ms) with the numba
engine correct at original tolerances, but it is NOT committable: the
Professor found the shipped default kernel-node grid UNDER-RESOLVED (real
production accuracy defect, masked by mis-aimed gates), the few-ms target
was missed, and the spec/doc pre-commit hook blocks the commit. All Build 3
deliverables sit UNCOMMITTED in the working tree — this build fixes forward
from that state and lands ONE commit that is accurate, fast, honestly
gated, and hook-clean. Accuracy first: a fast result that is wrong is
worthless, and Build 4 sampling cannot launch on a likelihood with
undetected >1.5-nat bias.

## What is wrong (Professor review + driver probes, 2026-07-17)

1. ACCURACY DEFECT. `_DEFAULT_KERNEL_NODES = 10` (likelihood.py:139) has
   O(1) interpolation error off the crown config; production RB-vs-brute
   leaks up to 3.44 nats on the kappa config (gate: RB_ATOL = 1.5). The
   inline comment claiming the default "clears interpolation gate (F
   within 1e-8 with margin)" is FALSE — delete it. The suite masked this:
   the interpolation gate ran on a 400-node PROXY grid (never production),
   production lnL was gated only on interpolation-benign configs, and
   `SelfFalsificationTestCase.test_interpolation_gate_rejects_the_default_
   sparse_grid` even PROVES the default fails. Gates must exercise the
   PRODUCTION grid.
2. TIMING SHORTFALL. Warm lnlike ~66-70 ms (engine ~94%), not few-ms. The
   test dev replaced the 10 ms in-build ceiling with a machine-calibrated
   `MS_CEILING = 0.25` — a documented deviation, but the owner's target
   stands.
3. HOOK BLOCK. New module `cogwheel/tests/test_lensing_fast_path.py` is
   not reflected in SPEC.md; the pre-commit spec/doc discipline hook
   refuses the commit (this killed the Build 3 pipeline at the commit
   step). The todo fragment `engine_hyp1f1-surrogate.md` also still needs
   its completion workflow (completed.d + spec_changelog fragment).

## Measured facts (pre-answered; do not re-derive)

Node-convergence of max|F_interp - F_dense|/|F_dense| (and the null-safe
max|dF|/max|F| in parens) vs ACTUAL grid size n, driver-measured with the
shipped `_interp_and_dense` (oracle = direct dense engine, F002-clean):

  two-image:  n=12 0.99 (0.51) | n=42 0.18 (2.8e-2) | n=82 3.5e-3 (1.1e-3) | n=162 1.6e-4 (6.4e-5)
  four-image: n=18 9.7e-3      | n=48 1.5e-4        | n=88 9.5e-6          | n=168 7.1e-7
  near-cusp:  n=12 1.1  (0.29) | n=42 4.4e-3        | n=82 5.5e-4          | n=162 4.3e-5
  kappa:      n=12 3.3  (0.28) | n=42 1.3e-2        | n=82 5.1e-4          | n=162 4.0e-5
  rot-shear:  n=12 6.7e-2      | n=42 2.8e-3        | n=82 1.3e-4          | n=162 5.9e-6

  The scheme converges everywhere (no stall); two-image is the slowest
  config. Whether the answer is a larger default, config-adaptive count,
  or better PLACEMENT (the current grid = log-spaced base UNION smoother-
  step transition frequencies; two-image's slow convergence suggests the
  base grid misses in-band structure) is the Architect's design call —
  bound it by measurement, not hope. Professor: the pointwise-relative
  1e-8 metric is ILL-POSED at interference nulls (|F_dense| -> 0); use a
  null-safe metric (e.g. normalized by max|F|) for the gate.

Engine cost split, crown params, 12-node grid, warm (driver cProfile):

  evaluate() total ~50-59 ms of which:
  - `geometry.nearest_caustic_point`: ~29 ms PER EVALUATE — a frequency-
    independent scipy `minimize_scalar` search making ~330 Python-level
    `critical_point` calls. Half the engine cost is w-INDEPENDENT geometry
    that should be near-free (closed-form/parametrized astroid distance,
    njit, or coarse-then-polish); it scales with NOTHING (same cost at any
    node count).
  - `F_op` ~2.3 ms/call, of which `_contract_orders` (ALREADY numba-njit,
    fastmath=False) is 1.93 ms — the per-point floor is the real FLOP
    count of ~order-40 explicit 85x85 contractions, NOT missing JIT. The
    plan's 0.2-0.4 ms/point prediction underestimated the work; cutting it
    means restructuring (e.g. stacked BLAS/tensor contraction across
    orders, exploiting table sparsity/zero-coefficient corners, or
    tightening `_series_length`/order truncation) — while preserving
    bit-level or <=2-ULP agreement and every F005 refusal check.
  - Timing is thread-insensitive: pinned OMP/MKL/NUMBA_NUM_THREADS=1
    measures the SAME ~70 ms lnlike (owner: production is a parallel
    sampler — every core busy — so gates must hold single-threaded).

Per-lnlike budget arithmetic: lnlike = caustic-search + n_nodes x
per-point + ~4 ms (contraction+ratio, already fine). Few-ms requires BOTH
the caustic search eliminated AND (n_nodes x per-point) driven to a few
ms — e.g. 80 nodes needs ~0.03 ms/point, 30 nodes ~0.1 ms/point. If the
honest floor after both fixes is above 10 ms, SAY SO in the change report
with the measured floor; do not widen accuracy tolerances to buy time,
and do not silently re-calibrate the ceiling (a moved gate is not a met
gate — driver escalates the residual to the owner for the 2D-table
decision).

## Scope fences

IN: `cogwheel/lensing/likelihood.py` (node default/placement, false
comment, gating surface), `cogwheel/lensing/chang_refsdal/geometry.py`
(`nearest_caustic_point` fast path ONLY — the astroid-distance
computation), `operator.py` (`_contract_orders` restructuring),
`channels.py` (only as needed to wire the above), `_hyp1f1.py` (only if
order truncation lives there), `cogwheel/tests/test_lensing_fast_path.py`
(re-aim gates at the production grid; null-safe metric; pinned-thread
timing), SPEC.md + spec_changelog/changelog/completed fragments (the hook
must pass).

OUT (do not touch): `_dd.py`/`_gauge.py` semantics; every refusal
threshold and the certified-or-named-refusal contract (F005); the
smootherstep switch and F008 fix; the stall-ringdown/template builders;
waveform.py; priors/sampling (Build 4); NO tolerance widening anywhere;
MS_CEILING-style machine recalibration of the few-ms goal.

## Constraints

- The working tree ALREADY HOLDS Build 3's uncommitted deliverables (WP1
  numba engine + WP2 spline grid + test_lensing_fast_path.py). Fix
  FORWARD from this state; do not revert or re-implement what is correct.
- Accuracy oracles unchanged and F002-clean: dense-engine oracle for
  interpolation, mpmath for kernels/F_op; never the code under test.
- `nearest_caustic_point` and `_contract_orders` changes must be
  value-preserving: same returned values within <=2 ULP on certified
  inputs (the caustic distance feeds branch/switch decisions — verify the
  same branch is taken across the existing suite), all F005/refusal
  behavior byte-identical.
- Timing gates: warm, best-of-N, measured with OMP/MKL/NUMBA threads
  pinned to 1 (production = parallel sampler; single-thread cost is the
  real number). Ratios (SPEEDUP_MIN, contraction subdominance) stay.
- The engine suites + fast-path suite must stay green at ORIGINAL
  tolerances; in-build tests stay FAST (single-eval brute anchors,
  sampled grids — no sweeps, no hour-scale specs).

## Acceptance (build-level)

- In-build: production RB-vs-brute |lnlike - lnlike_bruteforce| <= 1.5 on
  EVERY `_LENS_CONFIGS` row (one eval per path per config; brute ~10.5 s
  each — ~1 min total) with the SHIPPED production default node grid;
  interpolation gated on the PRODUCTION grid with a null-safe metric and
  a margin the Architect states with provenance; `SelfFalsification`
  updated so a genuinely under-resolved production default FAILS; warm
  pinned best-of-5 lnlike <= 10 ms on the crown config OR the measured
  honest floor documented in the change report (never a silently moved
  ceiling); engine suites + fast-path suite green at original tolerances.
- Commit lands: SPEC.md reflects the new test module (+ fragments,
  `python scripts/render_fragments.py` run), the spec/doc pre-commit hook
  passes WITHOUT --no-verify, todo fragment
  `engine_hyp1f1-surrogate.md` completed per the workflow.
- Post-build (driver, NOT in-build): full crown suite + full suite minus
  the XODE trio, detached, at original tolerances.

## Environment facts

- Suite interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server; SDK_CONDA_ENV routes it).
- Server baseline pre-Build-3 was fully green (163/163 engine+waveform in
  3m50s; crown 19/19 in 59m59s serial). numba 0.58.1, mpmath 1.3.0,
  pytest-xdist 3.8.0.
- Ignore test_gw_prior / test_posterior / test_waveform (XODE gap).
