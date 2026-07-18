# Build 3c — Few-millisecond lensed lnlike: the requirement, not the stretch goal

## Mission

Take `LensedRelativeBinningLikelihood.lnlike` from the current ~0.3 s/eval
(warm, single-thread, accuracy-correct after Builds 3/3b) to FEW
MILLISECONDS at UNCHANGED accuracy tolerances. Owner ruling (2026-07-18):
few-ms is NON-NEGOTIABLE — it is the package's competitive requirement
(comparable tools like GLoW evaluate amplification curves at ms scale;
our differentiators — embedded Chang-Refsdal, certified-or-refuse
accuracy, RB-likelihood integration — only matter at competitive speed).
10 ms is the PROGRAM requirement. Owner clarification (2026-07-18): it
may only be attainable by COMBINING levers — including further
exploitation of the `h_L = F * h_UL` factorization — so this build is
judged as a STEP: it must deliver a large MEASURED reduction with the
timing gate set at the plan's own predicted floor (stated with
arithmetic, then held to), plus a named next lever if that floor is
still above 10 ms. What is not acceptable: a moved/machine-calibrated
gate dressed as progress, or a plan with no quantified lever. If the
predicted floor reaches <= 10 ms, gate there.

## Where the milliseconds are (measured, 2026-07-17, this machine)

Warm single-thread lnlike ~0.3 s at the production default
`n_kernel_nodes = 100` (~105 effective nodes with full-cluster
transitions):

1. Engine `_amplification_coefficients` ~= n_nodes x ~2.3 ms/point.
   Per point: `_contract_orders` (ALREADY numba-njit, fastmath=False)
   is ~1.9 ms — an order-40 sum of 85x85 contractions, ~290k complex
   ops. This is a FLOP floor, not JIT overhead: more numba does nothing.
2. njit caustic search ~1.9 ms/eval (already fixed by 3b; fine).
3. `_data_term`+`_norm_term` contraction ~2.5 ms, ratio path ~0.5 ms —
   already inside a few-ms budget; leave them alone unless the timing
   gate says otherwise.

Budget arithmetic: 10 ms total needs the per-eval engine cost at or
under ~5 ms — i.e. ~0.05 ms/node at 105 nodes, or far fewer engine
evaluations per lnlike, or no per-node engine evaluation at all.

## The derivative structure (facts; do not re-derive)

The engine needs ~85 derivative orders of the point-mass 1F1 kernel
because `F_op`'s shear expansion is a derivative ladder. Producing them
is CHEAP and exact: the dd shared-numerator `P_n` table is built once
per point (F001 — dd lives only there) and the reciprocal-binomial
k-ladder reuses it for all orders (complexity pinned by
`LadderComplexityTestCase`). CONSUMING them — the 85x85 order
contraction per point — is the entire bottleneck. Consequence for any
surrogate: it must reproduce the POST-contraction output, never the
85-derivative ladder itself. The previously REJECTED 2D derivative-
ladder table (85 outputs spanning ~100 orders of magnitude, certified
interpolation through a cancellation regime) stays rejected — that
rejection was about tabulating the INTERNAL representation, and it does
not apply to the levers below.

## Candidate levers (Architect owns the design; Professor consult expected)

A. **Batched per-eval contraction.** Within one lnlike the lens params
   are FIXED; only ``w`` varies over the ~105 nodes. The per-node
   `z_powers @ (table*radial) @ zbar_powers` contractions share the
   z-power vectors, so the whole node set can be restructured as stacked
   BLAS/tensor contractions (one big matmul instead of 105 x 40 small
   explicit loops). This changes accumulation order — the reason it was
   deferred in 3b — so it must be RE-CERTIFIED against the 70-dps
   mpmath oracle across the F005 boundary band (the calibration
   methodology from F005 exists and is the template); the per-node
   refusal quantities (`positive_total`, `max_term`, `converged`,
   thresholds byte-unchanged) must still be computed and still fire on
   the same inputs. Expected 5-20x — necessary but possibly not
   sufficient alone.
B. **Reduced-space POST-contraction surrogate.** Tabulate/surrogate the
   contracted, smooth, O(1) channel kernels ``K_a`` (or ``F`` itself)
   over the REDUCED lens space: kappa is eliminated EXACTLY by the
   mass-sheet identity and beta by circular symmetry (locked design
   decision 5), leaving ``(w, y' in the shear frame, gamma')``. Domain-
   boxed per topology regime (the full-cluster transition machinery
   already knows where the kinks are); the EXISTING engine is the
   F002-clean oracle; out-of-box inputs fall back to the exact engine
   path or raise the existing named refusals — never silent
   extrapolation. Table construction cost is offline/setup, not
   per-eval.
C. **Fewer engine nodes via a smarter interpolant** (segmented
   Chebyshev between the known kink frequencies instead of a global
   cubic spline) — secondary; only worth a WP if A+B leave the node
   count binding.
D. If the Professor identifies a per-eval CURVE evaluation (e.g.
   recurrence/ODE marching of the kernel along the ``w`` ray, since
   ``zz`` is linear in ``w`` and 1F1 satisfies Kummer's ODE) as sounder
   than B, it is in scope as B's replacement — same oracle and refusal
   obligations.

## Scope fences

IN: `cogwheel/lensing/chang_refsdal/operator.py` (batched contraction,
re-certification), `_hyp1f1.py` (batched ladder feeds if needed),
`channels.py` (curve-level evaluation surface), a new module for the
surrogate/table if it lives naturally there, `cogwheel/lensing/
likelihood.py` (wiring only), tests via `domain_test_descriptions`.

OUT (do not touch): `_dd.py`/`_gauge.py` semantics; `geometry.py` (3b's
caustic path is done); every refusal THRESHOLD (the certified-or-named-
refusal contract must survive re-certification unchanged); the node-grid
accuracy machinery from 3b (production interp gate at null-safe 1e-3
stays green as-is unless lever C is taken); the stall-ringdown/template
builders; priors/sampling (Build 4); NO tolerance widening anywhere.

## Constraints

- Every accuracy gate at ORIGINAL tolerances: production-grid null-safe
  interp < 1e-3, RB-vs-brute <= max(1.5, 1e-2|bf|) on every
  `_LENS_CONFIGS` row, numba-vs-mpmath preservation, macro-limit
  7.85e-9, near-cusp pin, refusal symmetry. A fast result that is wrong
  is worthless.
- Surrogate-vs-engine-oracle error gets its OWN explicit gate with
  provenance (F002: the oracle is the exact engine/mpmath, never the
  surrogate).
- F010 discipline: after any compilation/restructuring change, the
  self-falsification tests must still be able to go RED (py_func-chain
  idiom for any new njit code); re-run them, do not assume.
- Timing gates: warm, best-of-N, OMP/MKL/NUMBA threads pinned to 1
  (production is a parallel sampler). The 10 ms ceiling is HARD.
- In-build tests FAST (single-eval brute anchors, sampled oracle grids,
  minutes not hours); the full suite is the driver's post-build step.

## Acceptance (build-level)

- In-build: warm pinned best-of-5 `lnlike` at or under THE PLAN'S OWN
  PREDICTED FLOOR on the crown 4-image config (10 ms if the levers
  reach it; otherwise the predicted floor becomes the gate and the
  remaining path to 10 ms is named in the plan and change report);
  every existing fast-path and engine gate green
  at original tolerances; the new surrogate/batching gates green with
  stated provenance; re-certification evidence for the accumulation-
  order change recorded (F005-style calibration table in the change
  report or FINDINGS).
- Post-build (driver): full suite minus the XODE trio green, detached;
  FINDINGS gains the measured story; fragments rendered; spec hook
  passes without --no-verify.

## Environment facts

- Suite interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server nereid; SDK_SERENA_PORT=8323 via .env — do not touch gw's 8322).
- Full suite minus XODE trio currently 207 passed in ~3 min at -n4;
  fast-path suite 20 passed in ~3.5 min serial. HEAD 32fe82a.
- numba 0.58.1, mpmath 1.3.0, pytest-xdist 3.8.0.
