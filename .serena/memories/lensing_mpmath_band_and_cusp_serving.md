# Interior cusp serving + mpmath-band hang cluster (Build, 2026-08-11, commit a8361be)

Consolidated record of the "interior cusp serving barrier" build: the cusp arm's
calibration bypass, the cross-arm double-serve gate, and the mpmath-band fast-tier
hang cluster (test-level fix; production fix deferred). Measured facts — reusable.

## 1. Interior cusp serving (deliverable)

`cusp_amplification` (_pearcey_cusp.py) now serves INTERIOR cusp sources — 3 real
stationary points, `rho < 1` — by skipping `_calibration_certified` when
`len(stationary_values) == 3`. Exterior (1-stationary) sources still certify
(delay-to-image alignment). Physics justification (Professor-ratified):
- the uniform-error gate `R >= radius_min` bounds the answer to the envelope bar;
- the uniform ratio `P/P_asymp` is SELF-CALIBRATING to leading order — both
  evaluated at the same `(x, y)`, so a control miscalibration cancels at first order.
The Pearcey table's real role is interior cusp serving (R ~ 1-4, diffraction
regime); exterior cusp configs have R >= 71.6 > r_ppgo_min=71.1 so ppGO fires
first there. Context: `mem:professor_code_observations` (CUSP PPGO FAST RUNG) and
`mem:coder_knowledge` (CUSP PPGO FAST RUNG IMPLEMENTATION).

## 2. Cross-arm double-serve regression: ppGO fold-band gate

The source-plane-nearest cusp routing fix (b64480c) made the cusp arm's ppGO rung
fire at a FOLD-region node (w=500, r=0.06, gamma=0.3) with 44% disagreement vs
the fold arm. Fix: gate the ppGO fast rung on
`nearest.distance >= _airy_fold._ETA_MAX_FOLD` (= 0.3) — the fold arm owns its
serving band (serves at `nearest.distance < 0.3`), the cusp ppGO rung serves at
`>= 0.3`; no overlap. Uses the `nearest` value already computed earlier in
`cusp_amplification` — no new geometry call.

GOTCHA (rejected alternative): a naive proximity gate (`|src-vertex| < 0.2`) was
proposed and REJECTED — it breaks legitimate EXTERIOR ppGO serving: the
`_PPGO_ASTROID_SOURCE` fixture has |src-vertex| = 0.69, fold_dist = 0.302 and
must still serve. The fold arm's OWN admission radius (`_ETA_MAX_FOLD`) is the
correct partition boundary; never substitute a source-proximity heuristic.

## 3. mpmath-band hang cluster root cause (biggest reusable lesson)

`_f_schwinger_mpmath` for `60 < w <= 150` is SLOW and DIVERGENT:
- adaptive per-panel `mp.quad` (tanh-sinh, maxdegree=5) at `dps = 30 + w` on a
  strongly oscillatory integrand (`e^{iwu/2}·kernel`);
- panel count grows ~w²: `n_panels ≈ w²/32` (309 at w=80, 907 at w=150);
- measured: w=80 → ~150-160 s; w=61/70/100 exceed 60 s; genuine DIVERGENCE
  (not mere slowness) at some (w, y) — the "6-hour freeze".
The sub-60 DD path is fast (~0.5 s) and bounded: FIXED 24-pt Gauss-Legendre
composite rule (`_dd_gl_rule`, `_PANEL_ORDER = 24`).

FIX = TEST-LEVEL parameter choice (driver + agents, no production code touched):
move ladder-node frequencies ABOVE the QD ceiling (150) so the engine hard-refuses
instantly (`w > 150` → immediate `SchwingerCertificationError`, no mpmath eval):
- `_CUSP_NODE_W` 80 → 160 (test_lensing_airy_fold.py)
- `_GEOMETRIC_NODE` w 100 → 200 (test_lensing_airy_fold.py)
- `FOP_REFUSALS` / supra grids 63 → 160 (test_lensing_fast_path.py)
- `LEVER5_ABOVE_CEILING_W` 62 → 160 (test_lensing_levers.py; old 62 sat in-band
  where the evaluator CERTIFIES instead of refusing)
This resolved 10 of 12 red-guard items in `lensing_serving_ladder_guards_are_red.md`.
PRODUCTION fix (bounded fixed-panel mpmath GL rule preserving N/2N certification)
is DEFERRED by user decision — tracked in `lensing_mpmath_band_fixed_panel_rule.md`.
Context: `mem:coder_knowledge` (MPMATH LAZY IMPORT + PAIRED N/2N CERTIFICATION
PATTERN — incl. the `mp.linspace` mpf-endpoints bug and QD dispatch order).

## 4. Two REMAINING red guards — genuine production issues, NOT parameter-fixable

Tracked in `lensing_serving_ladder_guards_are_red.md` (STILL RED section):
- `test_refusal_precedes_coherent_score`: CANCELLATION_LENS has hard-core nodes
  in (60, 150] that the engine evaluates via mpmath BEFORE refusing — no mass
  choice avoids the band (engine processes all in-band nodes first). Needs the
  deferred mpmath production fix.
- `test_thresholds_have_one_home`: `select_branch` says 'wave' for a saddle node
  (`w*delta_min < RHO_END`) but the grid serves the cusp arm's ppGO value, 1 ULP
  below `geometric_amplification`; the test's bit-identity probe reads 'geometric'.
  Pre-existing at HEAD (verified), unrelated to the cusp changes (nearest.distance
  = 0.84 passes the new ppGO gate either way). A routing/bit-identity
  adjudication, not a parameter fix.

## 5. Process lessons (pytest-timeout vs mpmath)

- pytest-timeout with `--timeout-method=thread` does NOT interrupt the mpmath
  C-level computation — a hanging mpmath test must be AVOIDED (by parameter
  choice / fixture design), not timed-out.
- A test can look fast in isolation but hang in-class (shared setUpClass/state)
  — always run the FULL class/file, not just the failing test.

Build review verdicts: `mem:inspector_knowledge` (CUSP PPGO FAST RUNG REVIEW +
the ppGO fold-band gate review — INS-16-001/002 trivial findings) and
`mem:professor_code_observations`. Spec anchor: `.claude/spec/SPEC.md` microlensing
engine row, `_pearcey_cusp.py` paragraph (INTERIOR CUSP SERVING + ppGO rung gate).
Completed fragments: `.claude/spec/completed.d/2026-08-11_interior_cusp_serving_barrier.md`,
`.claude/spec/completed.d/2026-08-11_mpmath_hang_fast_tier.md`.
