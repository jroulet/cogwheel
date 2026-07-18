# Build 3 — Few-millisecond lensed lnlike (engine surrogate + coarse kernel grid)

## Mission

Cut `LensedRelativeBinningLikelihood.lnlike` from ~20 s/eval to FEW
MILLISECONDS — the standard cogwheel relative-binning regime for unlensed
frequency-domain models — at UNCHANGED accuracy. Owner directive
(2026-07-17): performance FIRST; sampling (lens coordinates, folding) is
the NEXT build, not this one. Accuracy gates do not move: a fast result
that is wrong is worthless.

The measured cost structure (facts below) gives three multiplicative
levers, in order of leverage:

1. **Cheap kernel evaluations.** Naive per-point 1F1 evaluation (the DD
   shared-numerator series ladder in `_hyp1f1.py`) dominates everything.
   Build a tabulation or surrogate (e.g. precomputed tables with
   certified interpolation error, or Chebyshev/spline) for the channel
   kernels / 1F1 over the certified domain, with the DD ladder retained
   as the accuracy oracle and as the fallback for out-of-table inputs.
2. **Fewer kernel evaluations.** Exploit the factorization
   `h_L = F * h_UL`: `F(f)` is far smoother than `h_UL(f)`, so the smooth
   kernels deserve their OWN coarse global node grid (splined /
   interpolated to bin centers), decoupled from the waveform's bin grid —
   today the kernels inherit the bin grid and pay `h_UL`'s resolution for
   a quantity that needs a fraction of it. The image-delay phases stay
   ANALYTIC (already exact) — only the smooth kernels ride the coarse
   grid. The Professor's Build-2 consult measured the paper's prototype
   needing only ~6-11 global nodes.
3. **Contraction subdominance (conditional).** The `_data_term` +
   `_norm_term` contraction measures 142 ms on the laptop but only
   1.6 ms on this machine (many-core BLAS), so it does not block few-ms
   HERE. Treat it as conditional cleanup: if, with the fast engine, the
   existing contraction-subdominant-to-engine gate or the 10 ms ceiling
   fails, tighten it (numba is the house tool); do not spend a work
   package on it preemptively.

If an honest decomposition exceeds ~3 WPs, deliver levers 1-2 in this
build and leave the rest for a small follow-up (Build 3b) — say so in
the plan rather than widening it.

## Measured facts (pre-answered; do not re-derive)

Laptop (2026-07-17, crown 4-image config `y=(0.08,0.06)`, `gamma=0.20`,
`kappa=beta=0`, seeded fixture of `test_lensing_likelihood.py`):

1. lnlike ~20 s/eval: engine (`_amplification_coefficients`) 19.36 s =
   99.3%, contraction 0.142 s, ratio path ~1 ms.
2. 506 engine points/eval = 253 bins x `kernel_subsamples = 2` — the
   kernel grid currently inherits the waveform bin grid.
3. Speedup vs brute force ~8x (brute ~167 s); the RB advantage is
   entirely engine-bound.

Server (nereid, `cogwheel-newlal`, same fixture, 2026-07-17):

4. lnlike 14.79 s/eval (best-of-3, warm): engine 14.87 s (~100%; the
   two best-of measurements jitter past each other), contraction 1.6 ms
   (many-core BLAS — 90x faster than the laptop's 142 ms), ratio path
   0.5 ms. Brute force 119.2 s -> speedup 8.06x. Same 506 engine
   points; the engine is the ENTIRE cost on this machine.
5. Server baseline at HEAD (905869b): the six engine suites + waveform
   suite pass 163/163 in 3m50s; crown likelihood suite 19/19 in 59m59s
   serial (the brute-force sweep dominates).

Settled facts from the engine program (do not re-litigate):

6. F001 two-channel law: double-double is required ONLY in the 1F1
   kernel series (`L_1F1 = w*|y'|`; 1e-10 out to ~50, ~1e-6 at the
   ceiling 60). The operator channel is guarded by measured-cancellation
   REFUSAL (`CancellationError` past ~1e13), not extended precision.
7. Certified engine domain: `w <= 500`, `w*sqrt(s) <= 60`, positive
   parity only; geometric branch above `w*delta_min >= 4` and `L > 48`;
   the band `L in [~45, 48]` exits via named refusal (F005 NARROWED).
8. F009: `F(w->0) = 1/sqrt((1-kappa)**2 - gamma**2)` EXACTLY (macro
   limit, mass- and frequency-independent), pinned by
   `MacroMagnificationLimitTestCase` to rel 7.85e-9 at w down to 1e-12.
   NEVER add a small-w short-circuit that breaks this closed form.
9. The delay phases are analytic and exact; the paper's contraction is
   additive `M^2 + n_img^2` over bins and is correct as shipped (F008
   closed the near-cusp/two-image accuracy story; crown gates green at
   ORIGINAL tolerances, `RB_ATOL = 1.5`).

## Scope fences

IN: `cogwheel/lensing/chang_refsdal/_hyp1f1.py`, `operator.py`,
`channels.py` (surrogate/tabulation and a coarse-grid evaluation
surface); a new sibling module for tables/surrogate if it lives naturally
there; `cogwheel/lensing/likelihood.py` (kernel node grid decoupled from
bins; contraction tightening); tests via `domain_test_descriptions`.

OUT (do not touch): `_dd.py` / `_gauge.py` semantics; `geometry.py`;
every refusal threshold (`_CONTRACTION_GUARD`, `_CANCELLATION_REFUSAL`,
`MAX_ORDER` policy) and the certified-or-named-refusal contract (F005);
`waveform.py` conventions; the stall-ringdown / template builders (the
8.962e-3 zero-noise floor is an UPSTREAM standard-RB todo, not ours);
priors / sampling / folding (Build 4); NO tolerance widening anywhere.

## Constraints

- Oracle discipline (F002-safe): the surrogate is judged against the DD
  ladder (and mpmath in tests) — never against itself; no test fixture
  may be built from the surrogate under test. Surrogate-vs-oracle error
  gets its OWN explicit tolerance gate with provenance.
- Out-of-domain inputs fall back to the DD ladder (slow-but-correct) or
  raise the EXISTING named refusals — never silent extrapolation, never
  a refusal where the exact path used to return a certified value.
- The tiny-w macro-limit gate (fact 8) must stay green through whatever
  fast path serves that regime.
- Hot-path additions stay numba-compatible; no FFTs on the hot path
  (setup only); no per-frequency Python loops.
- Every config the existing suite exercises must keep its lnlike within
  the EXISTING gates — accuracy regressions are build-killers.

## Acceptance (build-level)

- In-build gates must be FAST (minutes, not hours — owner directive):
  the engine suites (163 tests, ~4 min here); a new surrogate-vs-oracle
  gate with explicit tolerance judged on a SAMPLED domain grid (few-eval
  oracle, not a sweep); a few-ms timing gate on the crown 4-image
  config — warm, best-of-N `lnlike` at or under 10 ms on this machine
  (target is few-ms; 10 ms is the regression ceiling); and ONE
  RB-vs-brute agreement anchor on that same single config (one eval per
  path, ~3 min) at the ORIGINAL `RB_ATOL = 1.5`. Do NOT spec the full
  crown brute-force sweep, the full suite, or any hour-scale run as an
  in-build test.
- Post-build (driver-verified, NOT an in-build test spec): the full
  crown suite and the full suite minus the XODE trio green at ORIGINAL
  tolerances, run detached by the driver on this many-core box.
- FINDINGS gains the measured surrogate story (domain, node counts,
  error calibration); todo fragment `engine_hyp1f1-surrogate.md` retired
  per the spec workflow (completed.d + spec/changelog fragments
  rendered).

## Environment facts

- Suite interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server; `SDK_CONDA_ENV` routes it — do not hard-code elsewhere).
- mpmath 1.3.0, numba 0.58.1, pytest-xdist 3.8.0 present.
- Ignore test_gw_prior / test_posterior / test_waveform (pre-existing
  IMRPhenomXODE optional-dep gap, not this build's concern).
- Timing gates: warm best-of-N pattern (already the suite idiom); the
  box may carry background load — gates are ratios plus the single
  10 ms absolute ceiling above.
