# Build Brief: Step 9 — Train the Full Surrogate (Once, in Final Coordinates)

## Mission

Train the full lens amplification surrogate in the final coordinates
established by steps 1-8 and 1e-*. This is a TRAINING RUN, not a code
change — it produces the shipped `.npz` artifact.

Prerequisites (all met):
- All coordinate axes finalized (arc-length theta, sqrt-edge lobe, log-reach
  gamma, sqrt-eta, log-w) ✅
- Curvature-relative tube shell (f_max=0.40) ✅
- Annulus retired, caustic-relative far zone ✅
- Ghost decay gate ✅
- Born residual wiring in place ✅
- Full test suite green (1139 passed) ✅
- Part 0 mechanical test in place ✅

## What this build does

This is NOT a normal code build. It:
1. Runs `from_engine` across both parities to build all chart types
   (tube, far-field, lobe-interior)
2. Trains the Born residual chart in the transition zone
3. Saves the combined surrogate artifact (.npz)
4. Runs the census to verify coverage and eps
5. Reports: coverage fraction, per-chart eps, any gaps

## Cost estimate (required by AGENTS.md before any engine run)

- Tube charts: ~20 bands × ~4 arcs × 64 engine calls each ≈ 5,120 calls
- Far-field charts: ~20 bands × ~4 tiles × 64 calls ≈ 5,120 calls
- Lobe charts: ~10 bands × 3 lobes × 64 calls ≈ 1,920 calls
- Born residual: ~10 gamma × ~10 rho × 200 w-points ≈ 20,000 calls
- Per call: ~0.5s (engine evaluation)
- Total: ~32,000 calls × 0.5s ≈ 16,000s ≈ ~4.5 hours
- With 8-core numba parallelism per call: effectively ~1-2 hours

NOTE: This estimate is rough. The actual cost depends on the number of
stable bands, arcs per band, and grid sizes. The build should compute the
exact count BEFORE launching and report it.

## In scope

- Write and run a training script (`scripts/train_surrogate.py`)
- The script calls `LensAmplificationSurrogate.from_engine` with the
  production TrainingConfig
- Also trains the BornResidualChart in the transition zone
- Saves artifacts to the package data directory
- Runs the census on the trained surrogate
- Reports coverage statistics

## Out of scope

- Changing any code (all code changes are done)
- The census lnL tiers (needs a likelihood instance — separate validation)

## Acceptance

- Full-suite gate green with the trained artifact
- Census: served fraction > 95% (the remainder is gamma-guard + dropped slivers)
- No `out-of-box` fallthrough exceeding 1% of draws
- Per-chart max eps < tube_eps_max (0.05)

## Constraints

- This is a LONG RUN (hours). Use the slow-tier infrastructure.
- Follow AGENTS.md cost-estimate discipline.
- The trained artifact is a SHIPPED file — commit it to the repo.
