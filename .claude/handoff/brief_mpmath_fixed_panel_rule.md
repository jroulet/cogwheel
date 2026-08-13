# Build Brief: Bounded mpmath fixed-panel rule for the Schwinger QD band

## Mission

Replace the adaptive per-panel `mp.quad` in `_f_schwinger_mpmath`
(`cogwheel/lensing/chang_refsdal/_schwinger.py`, the `60 < w <= 150`
arbitrary-precision exact-engine path) with a FIXED-panel Gauss-Legendre
rule evaluated at mpmath precision, making the band bounded and O(n_panels
· order) like the sub-60 DD path. This is the deferred production fix from
`lensing_mpmath_band_fixed_panel_rule.md`.

## Why

Measured (2026-08-11, driver): the current mpmath path runs
`mp.quad(..., maxdegree=5)` — adaptive tanh-sinh — PER PANEL at
`dps = 30 + w` on the strongly oscillatory integrand `e^{iwu/2}·kernel`.
Two compounding problems:

1. **Panel count grows ~w²**: `margin = πw/4 + 34` (cancellation depth),
   panel width `= 8π/w`, so `n_panels ≈ w²/32` (309 at w=80, 907 at w=150).
2. **Adaptive refinement diverges at some (w, y)** — the "6-hour freeze".
   Measured: `f_schwinger(80, y=[0.106,0.146], γ'=0.5)` ≈ 160 s; `w=61,70,100`
   exceed 60 s; the fast-tier hang cluster was only dodged by moving
   ladder-node frequencies above the QD ceiling.

Contrast the DD path (`w <= 60`): FIXED `_PANEL_ORDER = 24` Gauss-Legendre
composite rule per panel, precomputed once (`_dd_gl_rule`), no adaptivity,
~0.5 s regardless of w. The mpmath path should reuse this structure with
mpmath nodes/weights so the `e^{πw/4}` cancellation stays certified above
`w = 60`.

## The fix (single function: `_f_schwinger_mpmath`)

Replace `_raw_integral_mp`'s two `mp.quad(..., breakpoints, maxdegree=5)`
calls with a fixed-order composite Gauss-Legendre rule in `u = ln t`:
compute mpmath GL nodes/weights for order `_PANEL_ORDER` (or a dedicated
mpmath order) ONCE per call (or cache), map each panel
`[u_i, u_{i+1}]` to the GL nodes, sum `Σ w_k f(u_k)` at `dps = 30 + w`.
Keep:
- the N/2N paired-rule certification on the RECONSTRUCTED F (lines 176-190)
  — this is the accuracy contract;
- the IBP split at `T = w(|a|+|b|+2)/2`, part A on `[0,T]` (t^{s-1}
  singularity removed), tail on `[T, inf)` (lines 151-175);
- the `e^{πw/4}`-cancellation `margin = πw/4 + _U_MARGIN_CONST` and the
  panel-count formula (`_panel_count`) — these bound the truncation.

The fixed rule must be verified to reproduce the adaptive result to
`_CERTIFICATION_TOL` (3e-10) on a spot grid, OR raise `SchwingerCertificationError`
(refusal-conservative). Do NOT weaken the certification.

## Measured facts (at HEAD 72028a2)
- `_f_schwinger_mpmath` lines: `_raw_integral_mp` ~751-175 (mp.quad at 156,
  164), paired certification ~176-190. Constants: `_PANEL_ORDER=24`,
  `_MIN_PANELS=16`, `_WAVELENGTHS_PER_PANEL=2.0`, `_U_MARGIN_CONST=34.0`,
  `_CERTIFICATION_TOL=3e-10`, `W_CEILING_SCHWINGER=60`,
  `W_CEILING_SCHWINGER_QD=150`.
- DD path (`f_schwinger` w<=60) uses `_dd_gl_rule(_PANEL_ORDER)` — see lines
  500-560 for the panel loop + `_raw_t_integral_core` to mirror.
- mpmath GL nodes: `mp.quad` internals are tanh-sinh; a fixed GL rule needs
  `mp.gauss(24)` or equivalent at the working dps. Verify node accuracy at
  `dps = 30 + w`.

## Acceptance
1. `_f_schwinger_mpmath` for every `w in (61, 150]` and y in the served box
   completes in O(seconds) — NO divergence (the previous 160 s / 6-hour
   cases).
2. Paired-rule certification agrees with the CURRENT adaptive result to
   `_CERTIFICATION_TOL` on a spot grid spanning `w in {61, 80, 100, 120,
   150}` × several y (run both old and new, compare). If the fixed rule
   cannot meet the bar, it must refuse (named `SchwingerCertificationError`),
   never serve wrong.
3. `test_refusal_precedes_coherent_score`
   (`cogwheel/tests/test_lensing_marginalized_likelihood.py::RefusalContractTestCase`)
   completes fast (< 60 s) — it was hanging because CANCELLATION_LENS has
   hard-core nodes in the band. (If it still needs a band node, it now
   completes.)
4. Full-file fast-tier gates: `python -m pytest cogwheel/tests/test_lensing_fast_path.py cogwheel/tests/test_lensing_airy_fold.py -q --no-header --timeout=120 --timeout-method=thread` still green.
5. `test_lensing_schwinger.py`, `test_lensing_levers.py` (the mpmath-oracle
   accuracy tests) still pass.

## Constraints
- Fast tests only; NO slow-tier inside the build.
- Refusal-conservative: never serve a finite value the adaptive path would
  refuse; prefer a named refusal over a wrong number.
- numba-compatible not required (mpmath path is pure Python) but keep the
  module import lazy (`_mpmath` global).
- Do NOT touch the DD path, `_reconstruct`, or the w<=60 routing.
- Keep the existing N/2N certification contract intact.
- Optional (post-fix, NOT required): the parameter moves
  (`_CUSP_NODE_W`/`_GEOMETRIC_NODE`/`FOP_REFUSALS`/`LEVER5_ABOVE_CEILING_W`)
  may be reverted once the band is fast — but that is a follow-up, not this
  build's scope.

## Plan-gate note (relaunch)

The prior launch failed the plan gate: the Architect's `domain_test_descriptions`
bundled a test in `test_lensing_marginalized_likelihood.py` with the
`test_lenschwinger.py` suite, creating a Test Dev shard write-ownership
conflict (one shard must own exactly one primary test file).

RESOLUTION for the Architect: `test_refusal_precedes_coherent_score` ALREADY
EXISTS in `test_lensing_marginalized_likelihood.py` — it is NOT to be
re-authored. This build's verification is:
- PRIMARY suite: `test_lenschwinger.py` (the production fix lives in
  `_schwinger.py`; the mpmath-oracle accuracy tests live there).
- The refusal-precedence test is a CONSUMER, not a target: run it as
  verification (`python -m pytest ...RefusalContractTestCase`) but do NOT
  list it in any `domain_test_descriptions`. If a domain test is required,
  put it in `test_lenschwinger.py` (the suite this build owns).
