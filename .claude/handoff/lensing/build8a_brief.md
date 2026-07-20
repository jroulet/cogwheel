# Build 8a — The amplification surrogate: fast F(w) across the full prior box

## Mission

Build the surrogate speed layer for the Chang-Refsdal amplification so
the LENSED per-eval approaches the unlensed one across the WHOLE prior
box, both parities (owner objective, durable todos
`likelihood_envelope-surrogate.md` + `likelihood_schwinger-
homogenization.md`; owner ruling A parked all sampling until this
lands). The surrogate is an ADDITIVE layer: trained offline against
the certified engine, serving evaluations only inside its validated
domain, with the exact engine as fallback and every named refusal
preserved. This build does NOT homogenize the evaluator dispatch
(operator demotion to oracle duty and the Airy patch are Build 8b+).

## Measured price points (the baseline to beat — 2026-07-20)

- Crown (positive parity, warm memoized fiducial): 9.8 ms certified
  (suite gate). MUST NOT REGRESS — the surrogate must either leave the
  crown path untouched or beat it.
- Prior-box positive-parity draws: ~112-154 ms median (cold-fiducial
  ratio-layer misses + heavy-m_lens node counts), p90 ~700 ms.
- Saddle hosts: ~1.38 s warm (24-42 envelope nodes x 30-125 ms
  Schwinger each; fiducial build ~2.2 s).
- Target: lensed per-eval within ~2-4x of the unlensed RB lnlike
  across the box (owner-set), i.e. ~5-20 ms.

## Pre-answered design facts (do not re-derive)

- DIMENSIONALITY: lens mass and redshift enter F ONLY through the
  dimensionless frequency w (w = 8 pi G M_lens (1+z_lens) f / c^3).
  The amplification surface is F(w, gamma', beta, y1, y2) — FIVE
  dimensions, with kappa eliminated by the mass-sheet reduction the
  engine already performs (gamma' = gamma/(1-kappa); the sampled space
  has kappa = 0 so gamma' == gamma). An emulator over this 5-D space
  serves EVERY candidate.
- Symmetries that shrink the training domain: the u1/u2 reflection
  symmetry of the Fermat potential (F is invariant under y1 -> -y1 and
  y2 -> -y2 in the eigenframe; suites pin this at 1e-14) — train on
  ONE quadrant; the beta dependence is a rotation into the eigenframe
  (the engine reduces via exp(-i beta) — check whether beta can be
  eliminated exactly from the surrogate space the same way, leaving
  4-D).
- The SACR-C decomposition is the RIGHT surrogate target: the
  channel-layer envelope E(w) is beat-free and smooth BY CONSTRUCTION
  (that was its design point), while raw F(w) beats. Surrogate the
  envelope (or the per-node engine values the envelope build consumes)
  — never raw F on a fine w grid.
- Certified band: w <= 60 wave branch both parities (above: geometric
  branch when resolved; named refusals otherwise). The surrogate
  domain is the certified band; outside it the exact path serves
  unchanged.
- Accuracy currency: the existing envelope gate is reconstruction
  eps < 1e-3 relative on max|F| (research S2 gate; suites enforce).
  The surrogate must meet the SAME gate against the certified engine
  on held-out configs, and the lnlike-level effect must be gated:
  <= 0.01 nats at the crown, standard RB tolerance elsewhere
  (F016: strong-shear lnlike accuracy is RB-binning-limited — do not
  chase envelope precision past the binning floor).
- Refusal semantics: the engine's refusal SET must be preserved
  exactly. The surrogate never converts a refusal into a value or vice
  versa: candidate classification (refuse/evaluate) is decided by the
  same cheap domain checks as today (macro_matrix, band-limit, the
  fallback's gamma' > 0 rule); the surrogate only replaces the
  EXPENSIVE certified evaluations. Where the engine's refusal is only
  discoverable BY evaluating (CancellationError at w<=60 legacy,
  SchwingerCertificationError near the pinch), the surrogate's
  training set records the refusal boundary and the surrogate REFUSES
  CONSERVATIVELY (falls back to the exact engine near the boundary
  rather than guessing) — a surrogate answer where the engine would
  refuse is the F005 bug.
- The memoized-fiducial ratio layer stays: the surrogate accelerates
  the per-candidate envelope/node evaluations that the ratio layer
  still pays (the ~8-node candidate cost and every cache-miss fiducial
  build).
- Training/certification separation (F002): the training data
  generator and the accuracy-gate oracle must be independently seeded
  engine runs; the gate compares surrogate vs ENGINE (the engine is
  certified; the surrogate's contract is fidelity to it).

## Open design decisions (Architect owns; consult the Professor)

1. Emulator family for a smooth complex surface on a 4-5-D box
   (tensor splines / RBF / Chebyshev per-dimension / small NN) —
   numba-compatible evaluation under ~0.1 ms, deterministic,
   serializable to a versioned data file with a recorded training
   hash.
2. Offline-vs-lazy training: a shipped precomputed table for the
   sampled box vs build-on-first-use per event. (The prior box is
   fixed by the prior classes; shipped-offline is likely right; the
   training run itself is a POST-BUILD driver step — hours are fine —
   but the brief's in-build tests must use a SMALL training set on a
   reduced domain so gates run in minutes.)
3. Where the surrogate plugs in: at `_envelope_loo_nodes`' engine
   calls (per-node F), or one level up as a direct envelope emulator.
   Prefer the seam that leaves certify-or-refuse classification and
   the SACR-C reconstruction untouched.

## Out of scope — hard fences

- NO changes to the engine modules (geometry, operator, _schwinger,
  _hyp1f1, _gauge, _dd) — the surrogate consumes them for training.
- NO dispatch homogenization, NO operator demotion, NO Airy patch, NO
  v-plane (Build 8b+ per the homogenization todo).
- NO sampling runs (owner ruling A holds until this build LANDS and
  the owner re-opens sampling).
- Crown certified outputs: the exact path must remain available and
  byte-identical (the surrogate may SERVE the crown only if it beats
  9.8 ms AND meets the 0.01-nat gate; otherwise the crown keeps the
  exact path).

## Acceptance (build-level; two-tier per CLAUDE.md)

1. In-build (FAST, reduced training domain): surrogate-vs-engine
   envelope gate eps < 1e-3 on held-out configs BOTH parities; lnlike
   gates (0.01-nat crown-family / RB-tol strong-shear+saddle) on a
   small deterministic config set; refusal-set preservation test (a
   config the engine refuses must refuse through the surrogate path —
   with an F010 falsification proving the gate can go red); timing
   smoke: surrogate-served eval beats the exact path by >= 5x on a
   saddle config.
2. POST-BUILD driver-verified: full-box training run; the three price
   points re-measured (census rerun); full-suite regression.
