# Build 4 — Sampled lens coordinates, folding, and the sampling-ready posterior

## Mission

Make the microlensed likelihood SAMPLEABLE: implement the sampled lens
coordinate system, its prior, and folding support so a `Posterior` over
CBC + lens parameters can run under cogwheel's standard samplers, with
injection-recovery validation as the post-build driver step. The
likelihood layer is DONE (9.8 ms/eval warm at unchanged accuracy,
commit ec8a276) — this build adds the prior/coordinates layer per the
locked design decisions and the Professor's binding constraints. This
is the program's third deliverable (todo.d microlensed-PE item 3).

## Binding constraints (Professor, recorded at BUILD 2 COMPLETE — violations are build-killers)

1. Sample `d_app = d_L / sqrt(mu_macro)` — NEVER kappa. kappa is
   exactly degenerate (mass-sheet); it is eliminated, not sampled, and
   must not appear as a sampled or standard-side free parameter beyond
   its fixed role in the reduced parametrization.
2. NO unswallowed exceptions under sampler proposals: either bound the
   prior support to strict positive parity (`1 - kappa > |gamma|`
   respected by construction in the reduced coordinates) or map
   `geometry.LensDomainError` / `operator.CancellationError` to
   `lnL = -inf` AT THE POSTERIOR BOUNDARY ONLY (the likelihood/engine
   contract — named refusals, never silent values — stays untouched;
   the mapping lives where the sampler meets the posterior).
3. The constant-lens-phase ~ orbital-phase degeneracy is 22-ONLY:
   folding must NOT assume it for IMRPhenomXPHM higher modes. Any
   phase-fold is conditional on the waveform's mode content.

## Locked design decisions (decision 5 of the program; do not re-litigate)

- kappa NEVER sampled (exact mass-sheet identity); beta NEVER sampled
  (circular point mass); sample the REDUCED lens coordinates:
  gamma' (shear in the reduced convention) and the source position in
  the SHEAR FRAME; overall amplitude via the EXISTING apparent-distance
  machinery (d_app through `LookupTable`/distance marginalization
  idioms where applicable); the minimum-delay convention ties the lens
  time reference to `t_c`.
- Lens-mass/redshift enter through `w = xi(M_L, z_L) f`; the natural
  sampled combination (e.g. ln of the redshifted lens mass) is the
  Architect's call with Professor consult — cite the paper's
  parametrization (`.claude/spec/lensing_paper/` tex).
- Astroid quadrant symmetries provide the folding candidates: folding
  over the shear-frame source-angle quadrant (and any additional exact
  symmetry the Professor certifies) per cogwheel's standard folding
  machinery (`Posterior`/`prior.py` fold support).

## New design questions (plan must answer, Professor-consulted, code-pins verified)

- Fiducial-cache semantics under sampler parallelism: `_fid_cache` is
  in-memory per-instance state. Under multiprocess samplers
  (fork/pickle of the posterior), each worker gets its own cache —
  confirm JSONMixin/pickle round-trips exclude it cleanly (WP1 of 3g
  checked serialization; re-verify under fork) and that per-worker
  rebuilds are acceptable (~one direct eval per lattice cell per
  worker) or design a prewarm hook. Determinism per worker must hold.
- Prior classes: which existing `gw_prior` subprior combinations are
  reused vs new lens subpriors (registered in `prior_registry`,
  composable per cogwheel's Prior architecture); ranges with provenance
  (certified engine domain: `w <= 500`, `w*sqrt(s) <= 60`, positive
  parity with margin).
- Reference-waveform/relative-binning setup: `par_dic_0` for the lensed
  likelihood under `Posterior.from_event`-style construction (the
  unlensed reference idiom is already the heterodyne's design — wire,
  don't reinvent).

## Scope fences

IN: new lens subprior module(s) under `cogwheel/gw_prior/` (or
`cogwheel/lensing/` — Architect's call, mirror the package's own
conventions), registration in `prior_registry`, the
LensDomainError/CancellationError -> -inf mapping at the posterior
boundary, folding candidates, `cogwheel/lensing/likelihood.py` ONLY for
constructor/serialization plumbing the sampling layer needs, tests via
`domain_test_descriptions`.

OUT: the engine and SACR-C/ratio machinery (consumed as-is; no
channels/operator/_gauge/geometry edits); every refusal threshold; the
stall-ringdown/template builders; sampler internals (`sampling.py`
beyond what prior registration requires); NO tolerance widening.

## In-build gates (FAST — minutes; heavy validation is post-build)

- Prior transform round-trips: sampled -> standard -> sampled identity
  to 1e-12 across a seeded sweep of the support, including boundary
  neighborhoods; Jacobian consistency (finite-difference vs analytic
  where analytic is claimed).
- Positive-parity safety: a dense seeded sweep of the SAMPLED support
  maps to standard lens params that never violate `1-kappa > |gamma|`;
  OR (if the -inf route is taken) proposals engineered to refuse return
  exactly `-inf` with no exception escaping, and the SAME configs raise
  named refusals through the raw likelihood (contract preserved).
- d_app: lnL invariance along the mass-sheet direction at fixed
  sampled coordinates (the degeneracy is exactly absorbed — single-eval
  checks, seeded configs).
- Folding: unfold-sum consistency on a small seeded set (folded lnL
  equals the log-sum over reflected images), and the 22-only phase-fold
  is exercised with XPHM mode content to confirm it is NOT applied.
- Fiducial cache under fork/pickle: posterior round-trips
  (JSON + pickle) exclude the cache; a forked evaluation reproduces the
  parent's lnL bit-identically after cache warm-up.
- One end-to-end smoke: `Posterior` construction on the crown-config
  synthetic event + a few hundred prior draws evaluated (seconds at
  ~10 ms/eval) — finite lnL or exact -inf everywhere, no exceptions.

## Acceptance (build-level)

- All in-build gates green + the ENTIRE existing suite at original
  tolerances; commit hook-clean (SPEC row for the prior layer +
  fragments; the program todo fragment is retired only when the
  post-build validation is done — note it, don't retire it yet).
- Post-build (driver, detached, hour-scale): injection-recovery on a
  small injection set (the `cogwheel/validation/` pipeline) — sampler
  runs, PP-plot/coverage sanity; full suite minus XODE trio.

## Environment facts

- Interpreter: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
  (server nereid; SSE 8323 via .env). HEAD ec8a276: 254 passed +
  1 designed xfail in 58 s (-n4, minus XODE trio). lnlike ~9.8 ms warm.
- Paper tex + prototype: `.claude/spec/lensing_paper/`. Professor topic
  memory: `professor/priors_and_coordinates` (coordinate recipe +
  folding) and `professor/microlensing_chang_refsdal`.
- Samplers available: dynesty, nautilus, zeus (pymultinest optional).
