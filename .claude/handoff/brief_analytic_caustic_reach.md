# Build brief — the caustic reach is a formula, not a 720-point scan (F054)

## Mission

`ppgo_map.caustic_geometry` finds the maximum source-plane caustic radius by
sweeping 720 polar angles on both square-root branches — **1440
`geometry.critical_point` calls, on every likelihood evaluation.** It is 90% of
the surrogate's serve cost, and it is computing a closed-form expression the
long way. Replace it with the algebra.

This is F054 in `.claude/spec/FINDINGS.md` and
`todo.d/lensing_serve_path_caustic_reach.md`. Read F054 first. It runs BEFORE
the 1e-farfield collocation work, because the far-field `rho` axis is DEFINED
by this reach (`surrogate.py:367`, `:405`) — choosing nodes in a coordinate
that is about to move is the mistake 1e already exists to prevent.

## Measured facts (driver, 2026-07-30)

Per surrogate-served lensed `lnlike`:

| | | share |
|---|---|---|
| served `lnlike` | 31.25 ms | 100% |
| `_surrogate_coefficients` | 27.88 ms | 89% |
| `ppgo_map.caustic_geometry` | ~27.5 ms | **90% of the serve** |
| `critical_point` | **1440 calls per evaluation** | |
| `_contract_tensor_spline` (THE SPLINE) | | **1.7%** |

5M evaluations ≈ 43 core-hours at 31 ms; ≈ 4 at the 3 ms the fast path
already targets.

## The derivation — implement THIS, do not re-derive from scratch

With `lam = 1 - kappa`, `e = gamma/lam`, `t = cos 2theta`, and
`u = e t + b sqrt(1 - e^2 (1 - t^2))`:

    u^2 - 2 e t u = 1 - e^2          =>      e t = (u^2 - 1 + e^2) / (2u)

so `t` eliminates in favour of `u`. Writing `W = lam(1-u)`, the two caustic
components are `A = W - gamma`, `B = W + gamma`, and

    A^2 cos^2(theta) + B^2 sin^2(theta) = W^2 + gamma^2 - 2 gamma W t

Substituting `e t` and simplifying collapses the whole radius to a function of
`u` ALONE:

    |y|^2 = lam * [ (1-u)^2 (1+2u) + e^2 (2u-1) ] / u^2

Extremising (`d/du [P/u^2] = 0`, i.e. `P'u - 2P = 0`) gives

    u^3 - e^2 u + e^2 - 1 = 0        which FACTORS as
    (u - 1)(u^2 + u + 1 - e^2) = 0

    =>  u = 1,   u = (-1 +- sqrt(4 e^2 - 3)) / 2      (real iff e >= sqrt(3)/2)

The reach is `|y|` evaluated at the ATTAINABLE stationary points together with
the domain endpoints — the axis cusps `u = 1 -+ e`, and for a macro saddle
(`e > 1`) the wedge turnaround where the discriminant vanishes,
`u = sqrt(e^2 - 1)`. Five candidate `u` values, pure arithmetic.

SANITY CHECK the implementation must reproduce: at `u = 1 - e` this reduces to
`2 gamma / sqrt(1 - gamma)` for `kappa = 0`, i.e. 5.692100 at `gamma = 0.9` —
the number `SPEC.md` already records for the cusp radius.

DIRECTION: `caustic_geometry` also returns the unit direction of the farthest
point, used by `ppgo_map:881` and `test_lensing_ghost:577`. Recover `theta`
from the winning `u` via `cos 2theta = (u^2 - 1 + e^2) / (2 e u)`; the
direction must remain correct, not just the magnitude.

## THE ORACLE IS A REFINED SCAN, NOT THE INCUMBENT

The incumbent 720-point scan is WRONG wherever the extremum is off-grid. Do
NOT assert agreement with it, and do NOT preserve its numbers.

Measured convergence at `gamma = 1.05`, `kappa = 0`:

    n_theta =   720   scan 3.0072072   rel err 1.10e-04
    n_theta =  2880   scan 3.0075293   rel err 2.59e-06
    n_theta = 11520   scan 3.0075370   rel err 1.08e-08
    closed form       3.0075371

The scan CONVERGES TO THE FORMULA. Where the extremum happens to sit on an
axis (every astroid, and every saddle with `e >~ 1.2`) the incumbent agrees to
~1e-16 — but only because 720 is divisible by 4, so `pi/2` is a grid node.
That is accidental alignment, not design, and it is why this went unnoticed.

So the replacement is FASTER AND STRICTLY MORE ACCURATE. Validate against a
high-resolution scan (`n_theta >= 11520`) or against the formula's own
stationarity, never against `n_theta = 720`.

`gamma` just above the parity wall is the sharp case (`gamma = 1.001` gives a
reach of 22.3 and converges raggedly) — put the tightest scrutiny there, not
in the comfortable middle.

## Scope

IN:
* `cogwheel/lensing/ppgo_map.py::caustic_geometry` — the closed form, both
  return values.
* `cogwheel/lensing/surrogate.py::_caustic_reach` and its callers
  (`_to_caustic_fixed`, `_from_caustic_fixed`, `surrogate.py:860`).
* Test updates per the ownership map below.

SINGLE-SOURCING IS ALREADY AN INVARIANT — preserve it.
`test_lensing_ppgo_map.py:13` demands EXACT equality between
`_scalar_caustic_reach` and `caustic_geometry(gamma, 0)[0]`, and
`surrogate_training.py:3723` states the reach is defined "in exactly ONE
place". The closed form inherits that; do not introduce a second copy.

OUT — do not touch:
* the far-field `rho`/`theta_c` COLLOCATION (1e-farfield, the next build) —
  this build changes what the reach IS, not where nodes sit in it;
* `surrogate_training._caustic_reach` (a DIFFERENT, branch-and-interval
  function of the same name) and `_scalar_caustic_reach`'s call sites beyond
  re-pointing;
* the tube chart, the lobe charts, any chart schema. No training.

## Test-suite ownership — DISJOINT, one author per file

~40 references across seven files read reach-derived values; any carrying a
number frozen from the 720-grid shifts by ~1e-4 near the wall. Assigned by
which suite owns the predicate:

* `cogwheel/tests/test_lensing_ppgo_map.py` — the REACH ITSELF and the
  single-source equality. The closed form vs a converged scan, both parities,
  including `e < sqrt(3)/2` (no real interior root) and `e` just above 1.
* `cogwheel/tests/test_lensing_exterior_admission.py` — the ADMISSION
  decisions that consume the reach, including the `gamma = 1.0`
  `LensDomainError` parity-wall refusal (`:1058`), which must survive.
* `cogwheel/tests/test_lensing_surrogate.py` — SERVED VALUES and the cost
  claim.

Leave `test_lensing_exterior_windows.py`, `test_lensing_ppgo_bandsplit.py` and
`test_lensing_ghost.py` alone unless a frozen literal genuinely moves; if one
does, re-point it in the suite that owns it and say so.

## Acceptance

State the measured number for each.

1. `critical_point` calls per served `lnlike` drop from **1440** to O(10) —
   ideally zero on the reach path.
2. `_contract_tensor_spline` becomes a MAJORITY of `_surrogate_coefficients`,
   not 1.7%. Report the before/after split and the served `lnlike` cost
   against the fast path's ~3 ms target.
3. Reach agrees with a CONVERGED scan (`n_theta >= 11520`) to <= 1e-9
   relative across both parities, `kappa != 0`, and the near-wall band
   `e in (1, 1.2)`. NOT against `n_theta = 720`.
4. The returned DIRECTION still points at the farthest caustic point.
5. Served `|F|` and phase unchanged to the F016 envelope bar — this is a cost
   and accuracy change, not a physics change. Where a served value moves
   because the incumbent was wrong, say so and quantify.
6. The `gamma = 1` parity-wall named refusal still raises.
7. Suites you touched run green. Full suite and slow tiers are post-build
   driver steps.

## Constraints

- Assert VALUES against tolerances, not code paths. ONE canonical pin per
  decision, in the file that owns it.
- Never preserve an incumbent number by construction — here the incumbent is
  measurably wrong, so matching it is a FAILURE, not a pass.
- No `git show HEAD:` oracles (F043/F045); freeze literals instead.
- Slow tests never run in-build; `COGWHEEL_BRUTE_ACCURACY` /
  `COGWHEEL_TRAIN_TIER` / `COGWHEEL_STRICT_TIMING` stay empty.
- `SDK_CONDA_ENV` from `.env`; interpreter
  `$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`, never `conda run -n`.
- Prose you change must be true when done: `caustic_geometry`'s docstring
  describes a sweep, and `SPEC.md` mentions the directional caustic radius —
  check both.
