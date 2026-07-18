# Build 3g — The ratio layer: heterodyne the envelope, land 10 ms

## Mission

Take warm, thread-pinned `lnlike` from the measured 29.5 ms/eval
(post-SACR-C, commit b2d80a0) to AT OR UNDER 10 ms by implementing the
paper's candidate/fiducial RATIO LAYER (`q_a`, tex Eq.
slow-component-ratio in `.claude/spec/lensing_paper/`) on top of the
SACR-C envelope: build the fiducial envelope ONCE at the reference
parameters (setup cost), and per candidate evaluate only the RATIO of
the candidate envelope to the fiducial — the ultra-smooth object the
paper's famous ~6-11-node figure was actually measured on (research
report `.claude/handoff/lensing/envelope_research.md`, Sec. 1-2: ~7-12
nodes/decade ON RATIOS over its 0.9-decade band). This is the
factorization completed: relative binning applied to the lens sector
itself. Owner preference on record: this lever BEFORE the surrogate
table (no cache machinery); the surrogate remains the named backstop if
the measured result leaves a gap.

## Measured facts (do not re-derive)

1. Post-3f baseline (driver harness, pinned, crown config): lnlike
   29.5 ms; engine 28.1 ms of it; the exact 1F1 derivative ladder
   inside `F_op_grid` node evaluations is ~89% of total cost (Professor
   review memo). LOO node count at fiducial-quality accuracy: ~21-44.
2. Ratios of beat-free envelopes at nearby lens parameters are smoother
   than either envelope (the beat carriers cancel in the ratio up to
   the small delay DIFFERENCES between candidate and fiducial — which
   the carriers absorb analytically). The paper measured ~6-11 greedy
   nodes for its ratio objects; the report confirms the per-decade
   economics transfer.
3. Cost arithmetic: per-proposal engine cost = N_ratio x ~0.4 ms +
   reconstruction (vectorized numpy, ~1-2 ms) + contraction/data/norm
   (~2.5 ms) + caustic (~1.9 ms). At N_ratio ~ 8-12: ~7-9 ms total.
   The 10 ms gate is therefore the honest in-build ceiling (step rule
   applies only through a Professor-analysis obstruction).
4. SACR-C invariants that must survive: reconstruction identity vs the
   untouched `exact_total` oracle; criticality-separation switch;
   config-independent node ceilings; every existing gate at original
   tolerances (RB-vs-brute max(1.5, 1e-2|bf|), near-cusp pin,
   zero-noise floors, macro limit, crossing boundedness, F001 carrier
   phase, F009 deep-band).

## Design questions the plan must answer (Professor-consulted, code-pins verified)

- The exact ratio object: candidate envelope over fiducial envelope at
  the SAME w-grid, or the paper's `q_a` per-channel slow-component
  ratio? What carriers absorb the candidate-vs-fiducial delay
  differences (tau_a and tau_c both move with lens params)? The paper's
  tex is the authority; the report's Sec. 1-2 reading is the map.
- Fiducial refresh policy: when the sampler wanders far enough that the
  ratio's node count degrades, when/how is the fiducial rebuilt?
  (Deterministic trigger — e.g. LOO node count exceeding a ceiling —
  never error-adaptive per-eval logic that breaks determinism.)
- Where the fiducial lives: per-LensedRelativeBinningLikelihood
  instance state alongside the existing `par_dic_0` reference-waveform
  machinery (the RB idiom already has exactly this shape — reuse it,
  do not invent a parallel cache).
- VERIFY every code-pin with find_symbol before planning on it (the
  Build-3e lesson; the driver will spot-check the plan's pins).

## Scope fences

IN: `cogwheel/lensing/likelihood.py` (ratio-layer state, per-proposal
evaluation, fiducial refresh); `cogwheel/lensing/chang_refsdal/
channels.py` ONLY if a ratio accessor belongs beside the envelope
accessor; tests via `domain_test_descriptions`.

OUT: `operator.py`/`_hyp1f1.py`/`_dd.py`/`geometry.py`/`_gauge.py`
(the engine is done); every refusal threshold; `exact_total` and
`lnlike_bruteforce` (untouched oracles); the SACR-C construction
semantics (consumed, not modified); the stall-ringdown/template
builders; priors/sampling (Build 4); NO tolerance widening.

## Constraints

- Correctness first: the ratio path must agree with the DIRECT SACR-C
  path (fiducial == candidate must be an exact identity; perturbed
  candidates gated against the direct evaluation AND lnlike_bruteforce
  at original tolerances across the five anchors + perturbation
  sweeps).
- Refusal symmetry unswallowed on ratio, direct, and brute paths; a
  candidate whose lens params leave the certified domain refuses
  identically in all three.
- Determinism: bit-identical repeats; the fiducial refresh trigger is
  deterministic.
- No new njit expected (vectorized numpy); F010 applies if any appears.
- In-build tests FAST (seconds-to-minutes; single-eval brute anchors);
  full suite = driver post-build.
- Timing gates structural-first (N_ratio ceiling config-independent;
  public-entry speedup), then warm pinned best-of-5 `lnlike <= 10 ms`
  on the crown config (HARD, step rule only via Professor obstruction
  analysis).

## Acceptance (build-level)

- Ratio-vs-direct identity and gated agreement; all existing suites
  green at original tolerances; the 10 ms gate met (or the step-rule
  case honestly documented with the measured floor and the surrogate
  named); SPEC + fragments updated so the commit lands hook-clean.
- Post-build (driver): full suite minus XODE trio detached; timing
  series updated in the journal.

## Environment facts

- Interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server nereid; SSE 8323 via .env). HEAD b2d80a0: 236 passed +
  1 designed xfail in 52 s (-n4, minus XODE trio).
- The paper tex + prototype: `.claude/spec/lensing_paper/`. The
  research report: `.claude/handoff/lensing/envelope_research.md`.
- numba 0.58.1, mpmath 1.3.0, pytest-xdist 3.8.0.
