# Build 2: LensedWaveformGenerator + multi-component relative-binning likelihood

## Context
Build 1b landed the complete Chang–Refsdal engine (`cogwheel/lensing/chang_refsdal/`,
all six modules, 126 tests green, Inspector PASS, Professor physics PASS). The
public entry point is `ChangRefsdalChannels` (topology-stable
`F(w) = sum_a e^{i*w*tau_a} K_a(w)`), which THIS build consumes. Design math is
locked in `.claude/handoff/lensing/META_PLAN.md` ("Design decisions") and the
`professor/microlensing_chang_refsdal` memory; the physics source is the
manuscript at `.claude/spec/lensing_paper/` (unpublished — stays under
`.claude/`). Where anything conflicts with this brief, THIS BRIEF WINS.

## Mission — four production deliverables
1. CLOSE FINDINGS F005 (engine prerequisite — the Professor made this binding
   before the likelihood may trust the high-magnification band). In
   `operator.py`: make the wave-branch contraction safe for cancellation
   exponents L = w*|y'| in [~30, 48] — an overflow-safe/rescaled contraction
   (e.g. factor the max magnitude out of `derivs` before the matmuls), and a
   NAMED refusal (`CancellationError`) whenever the contraction cannot certify
   its target, so the module keeps its "named error, never a silently wrong
   number" contract over the whole wave branch. Silent `nan` is the bug; a
   refusal is acceptable, an accurate answer is better. Update FINDINGS F005
   (resolved or narrowed) and the SPEC limitation line to match reality.
2. `cogwheel/lensing/waveform.py` — `LensedWaveformGenerator`: wraps cogwheel's
   existing waveform machinery and applies the lensing amplification per mode,
   h_lensed(f) = F(w(f)) * h(f), with w = 8*pi*G*M_L*(1+z_L)*f/c^3
   (DIMENSIONLESS, LINEAR in f; M_L in solar masses, f in Hz — keep the
   conversion constant in one place). Consumes `ChangRefsdalChannels`
   (tau_a, K_a(w)) and exposes the per-image decomposition the likelihood
   needs, not just the total. MACRO-SADDLE REJECTION AT THE API BOUNDARY
   (Professor, binding): positive-parity only is a scope limit — configurations
   violating 1 - kappa > |gamma| must RAISE (geometry.LensDomainError
   propagates; never swallowed into a warning or a nan).
3. `cogwheel/lensing/likelihood.py` — the multi-component relative-binning
   likelihood. The locked structure (decisions 2-4, do not relitigate):
   - Delay-continuous summaries T^(p)_mn,b(delta_t) and data-side
     A^(p)_m,b(delta_t): the rapid x rapid phase products stay INSIDE the
     frequency sum. NEVER product-of-summaries — summarizing F and h
     separately and multiplying the summaries is wrong math; the cross terms
     do not factor.
   - Slow fiducial K_a0 * conj(K_c0) Taylor-expanded within bins; the <h|h>
     norm term costs one extra moment (p <= 3).
   - Image-delay phases e^{i*w*tau_a} handled ANALYTICALLY; K_a(w)
     interpolated (it is smooth by construction of the channels).
   - Hot path: NO FFTs (setup only). Sequential contraction — modes first
     (M^2 x few delta_t-grid nodes x bins), then images via envelope
     interpolation at the n_img^2 pair delays. Additive M^2 + n_img^2 cost,
     NOT multiplicative; the contraction must stay subdominant to the
     coarse-node waveform call (assert this in a timing test).
   - delta_t vs Delta_t: fiducial ABSOLUTE delays are exact inside the
     summaries; the candidate's RESIDUAL delta_t is handled by linear RB with
     a lens-aware bin criterion plus a guard assert
     [pi * Delta_f_bin * delta_t_max < tol]. The overall time shift uses the
     BaseLinearFree idiom already in cogwheel.
   - Consume the engine's `branch` flag; the K-accuracy domain is the
     cancellation-exponent law (post-WP1: certified or named-refusal
     everywhere), not a box in (w, y).
4. Closeout: SPEC.md layer row for the new modules + spec_changelog fragment;
   changelog fragment; FINDINGS update per WP1; short overview.rst addition.
   KEEP `.claude/spec/todo.d/2026-07-16_lensing-program.md` (Build 3 pending).

If any deliverable produces/consumes an on-disk data product, register it in
DATA_CONTRACTS.yaml + contracts_changelog.d fragment; otherwise none is expected.

## Tests — put ALL of these in `domain_test_descriptions`
Name each spec's target suite explicitly (one suite per module):
`test_lensing_operator.py` (EXTEND the existing suite), `test_lensing_waveform.py`,
`test_lensing_likelihood.py`. Give each spec setup / operation / expected /
diagnostic.
- operator (extension, gates WP1): mpmath-oracle accuracy extended from the
  current L <= 25 up through L = 48, or a NAMED CancellationError wherever the
  contraction refuses — assert there is NO (finite, wrong, silent) outcome
  anywhere in the band; explicitly probe the former silent-nan configs near
  L ~ 40. The existing suite's L <= 25 gates must keep passing unchanged.
- waveform: unlensed limit (far from caustic / w -> 0: h_lensed -> h to
  roundoff, |F| -> 1); w(f) convention check (linear in f, correct constant,
  against a hand-computed value for a reference M_L, z_L); per-mode consistency
  (lensing acts on each mode's frequency array, not the 22-mode grid only);
  MACRO-SADDLE REJECTION raises LensDomainError at the API boundary (both the
  generator constructor and the likelihood path — never a warning or nan).
- likelihood (the crown gate, cogwheel value #1): relative-binning lnL agrees
  with the EXACT brute-force lnL (full frequency grid through the same
  LensedWaveformGenerator) within stated tolerance across a parameter grid
  covering: 2-image and 4-image regimes, near-fold and near-cusp
  configurations, candidate delays offset from fiducial (exercising the
  delta_t machinery and its guard), and mode content beyond 22. Plus: the
  delta_t guard assert actually fires when the bin criterion is violated
  (falsification, not just prose); a timing assertion that the contraction is
  subdominant to the coarse waveform call; and a structural regression that
  would FAIL under product-of-summaries (two images with near-degenerate
  delays where the cross term dominates — pin the exact answer).

## Settled facts — inputs, not questions
- Engine certification (current, pre-WP1): w <= 500; dd kernel to w*sqrt(s) <= 60
  (1e-6 at ceiling, 1e-10 to ~50); wave-branch contraction oracle-certified to
  L ~ 25-30; L in [~30,48] OPEN (F005 — silent nan near 40, no refusal);
  geometric branch above w*delta_min >= 4.0 AND L > 48. WP1 exists to close
  the gap; the likelihood may not paper over it.
- Positive parity only: 1 - kappa > |gamma| (macro saddles / Type II images out
  of scope, enforced by raising, per the Professor).
- The reconstruction identity and scale-aware bound live in `_gauge.py`
  (34 tests); channels reuse it — do not re-derive in likelihood code.
- mpmath is test-oracle only; it remains an undeclared test dependency
  (FINDINGS F003) — report, do not fix here.
- The brute-force reference must go through the SAME waveform generator as the
  RB path (isolating the binning approximation), and additionally against
  cogwheel's existing unlensed likelihood in the F -> 1 limit (catching a
  normalization error the self-referential comparison cannot).

## Scope
Sampled lens coordinates, priors, astroid folding, injection-recovery: Build 3.
No sampler runs in this build's tests (fast tests only — likelihood
evaluations, not posteriors).
