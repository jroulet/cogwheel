# Build 5 — Marginalized lensed likelihood: make the posterior samplable in hours, not days

## Mission

Wire the lensed likelihood into cogwheel's marginalization machinery so
the microlensed posterior samples in reasonable wall-clock. Owner ruling
(2026-07-18): full extrinsic marginalization with conditional draws is
REQUIRED — the plain path measures ~1-2 blind-draw evals/s (fresh XPHM
coarse-waveform generation per proposal + cold fiducial-lattice cells +
refusal overhead dominate; the warm fixed-candidate 9.8 ms does not
transfer), giving ~1 day per 15-D posterior. The unlensed pipeline
solves exactly this with the marginalized-extrinsic ("coherent score")
likelihoods + postprocessing conditional draws; lensing lives entirely
in the INTRINSIC sector (given lens params, `h_lensed = F * h` is just a
waveform), so the machinery should carry over — the build's job is the
lensed analogues of the summary data it consumes, plus d_app riding the
distance marginalization. Sampled space shrinks to intrinsic + lens
(~9-10 dims), extrinsics drawn from conditionals in postprocessing.

## Measured motivating facts (do not re-derive)

1. Plain-path sampling measured (2026-07-18, Nautilus, crown-config
   synthetic event, pool=4): ~1-2 evals/s blind; 15-D; ~1 day per
   converged posterior. Stack itself validated (1500+ evals, zero
   exceptions, refusal net exact -inf).
2. Warm fixed-candidate lnlike is 9.8 ms (ratio layer); the blind-draw
   gap is per-proposal XPHM coarse strain + cold fiducial cells +
   refusal exception overhead — marginalization amortizes/eliminates
   the extrinsic-parameter portion of proposals entirely.
3. The Professor's binding constraint (Build 2, still in force): the
   constant-lens-phase ~ orbital-phase degeneracy is 22-ONLY — phase
   marginalization must respect mode content (the existing qas/hm
   split in the marginalization layer encodes exactly this
   distinction; reuse it, do not blur it).
4. d_app enters the lensed strain exactly as d_L does the unlensed one
   (F009: the sqrt(mu_macro) constant folds into apparent distance) —
   the existing distance-marginalization `LookupTable` machinery
   applies with d_app in place of d_L.

## Where the machinery lives (verify signatures before planning on them)

- `cogwheel/likelihood/marginalized_distance.py`,
  `marginalized_distance_phase.py` — distance(+phase, 22-only)
  marginalization over the RB likelihood.
- `cogwheel/likelihood/marginalized_extrinsic.py` /
  `marginalized_extrinsic_qas.py` + `cogwheel/likelihood/
  marginalization/` (coherent score, skydict, lookup tables) — full
  extrinsic marginalization from per-detector matched-filter
  timeseries; numba-accelerated; conditional draws for postprocessing.
- `cogwheel/lensing/likelihood.py` — the lensed RB summaries
  (delay-continuous moments, ratio layer). The lensed analogue of the
  `(d|h)` timeseries / mode decomposition the coherent score consumes
  is THE design question: the lensed waveform's per-mode structure is
  `F(w) * h_m(f)` with image delays analytic — the original design
  list anticipated this ("data-side A^(p)_m,b(delta_t) ~= z_m(t)
  timeseries", design decision 2).
- `cogwheel/lensing/prior.py` / `posterior.py` — Build 4's sampled
  coordinates; the marginalized variant needs the extrinsic subpriors
  swapped out exactly as the unlensed marginalized priors do
  (mirror the existing registered marginalized prior classes).

## Design questions the plan must answer (Professor-consulted, pins verified)

- Which marginalization tier ships THIS build: (a) distance(+22-only
  phase) marginalization for the lensed likelihood — near-verbatim,
  low-risk, big win alone (kills d_app + phase dims and their proposal
  cost); (b) the full coherent-score extrinsic marginalization with
  the lensed timeseries — the complete fix. Professor decides whether
  (b) is one build or (a)-then-(b); an honest (a)-first with (b)
  immediately following is acceptable (step rule; never dropped).
- The lensed timeseries construction for (b): from the existing lensed
  RB summaries or a dedicated setup path; delay structure handled
  analytically (no FFTs on the hot path — standing design rule).
- Fiducial-cache interaction: marginalized evaluation changes the
  per-proposal parameter subset (intrinsic + lens only) — the lattice
  hit rate should IMPROVE (extrinsics no longer thrash proposals);
  confirm and record.
- Accuracy oracle: the marginalized lnL must be gated against direct
  numerical integration over the marginalized parameters on seeded
  configs (F002-independent quadrature, not the machinery itself),
  plus consistency with the plain path via importance reweighting on
  a small draw set.

## Scope fences

IN: a lensed marginalized likelihood class (or classes) in
`cogwheel/lensing/` composing the existing marginalization layer; the
marginalized lensed prior/posterior variants (registered); conditional
draw postprocessing wiring; tests via `domain_test_descriptions`.

OUT: the engine (`chang_refsdal/` untouched); the plain
`LensedRelativeBinningLikelihood` path and its gates (it remains the
oracle); the coherent-score numba internals and lookup-table formats
(consumed, not modified — if a genuine extension is unavoidable the
plan must call it out explicitly with its re-certification);
negative-parity (next program); NO tolerance widening.

## Constraints

- 22-only phase constraint respected structurally (hm vs qas split).
- Refusal semantics: marginalized evaluation must preserve the
  posterior-boundary -inf mapping; no refusal may be silently averaged
  over inside a marginalization integral (a refusing lens config
  refuses BEFORE extrinsic marginalization is attempted).
- Determinism: marginalized lnL bit-repeatable; conditional draws
  seeded.
- In-build gates FAST: marginalized-vs-direct-integration on seeded
  single configs (quadrature oracle, minutes); plain-vs-marginalized
  consistency spot checks; existing suites untouched and green.
- Post-build (driver): a REAL sampling run on the marginalized
  posterior (the wall-clock headline — target: hours, not days) and
  then the deferred injection-recovery/PP validation on the
  marginalized path.

## Acceptance (build-level)

- Marginalized lnL gated against the independent quadrature oracle at
  Professor-set tolerances; conditional-draw postprocessing reproduces
  the extrinsic posterior on a seeded config (KS-style check vs direct
  sampling of a cheap sub-case if feasible in-build); every existing
  suite green at original tolerances; commit hook-clean (SPEC row +
  fragments).
- Post-build: the measured sampling run — wall-clock and effective
  samples recorded in the journal; this number is the build's headline
  and the program's ship gate.

## Environment facts

- Interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server nereid; SSE 8323 via .env). HEAD 03a1787: 281 passed +
  2 designed xfails in 1:23 (-n4, minus XODE trio).
- The plain-path smoke run evidence: scratchpad lensed_run0.log
  (throughput lines) — facts inlined above.
- numba 0.58.1, mpmath 1.3.0, pytest-xdist 3.8.0; samplers dynesty,
  nautilus, zeus.
