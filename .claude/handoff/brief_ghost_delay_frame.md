# Build 8h-b7: ghost complex-saddle delay-frame repair

## Mission

The decaying complex-saddle ("ghost") term is built in a DIFFERENT delay frame
from the real-image kernels it is subtracted alongside. Repair the frame so the
mid-band far-field label `FARFIELD_KERNEL_SUM_MINUS_GHOST` subtracts a
correctly-phased ghost, and pin the repair with fast regression tests.

This is a correctness defect with a large measured payoff: it is currently the
binding constraint on how far DOWN in dimensionless frequency the ppGO rung can
reach, and therefore on how wide a `w` band the surrogate charts must cover.

## Measured facts (driver-measured 2026-07-27; agents need not re-derive)

Probes: positive parity, `beta = kappa = 0`, source at
`|y| = r_caustic(gamma, theta) + offset`, `w` grid linear on `[10, 59]`.

1. `partition.delays` are MIN-SUBTRACTED (they always contain an exact `0`).
   Example at `gamma=0.30, theta=15deg, offset=0.25`: raw per-image delays from
   `geometry.delay` are `[-0.33608, 1.13981]`; `partition.delays` are
   `[1.47589, 0, ...]`. The subtracted shift is `tau_shift = -min(raw)`.
2. `geometry.ghost_kernel` returns `delay` (`tau_c`) in the RAW frame.
3. Let `R = F - sum(real kernels)` (i.e. `FARFIELD_KERNEL_SUM`) and
   `G = kernel * exp(1j*w*tau_c)`. Measured: `|R|/|G| = 1.00` across the band
   (first/median/last within 1-7%), and the decay rates agree to 4 digits
   (`dlog|R|/dw = -0.08235` vs `dlog|G|/dw = -0.08373`; `-0.10327` vs
   `-0.10342`). Same mode, same envelope.
4. `arg(R/G)` is LINEAR in `w` (fit rms 0.0024-0.045 rad) with intercept
   `-6.18`/`-6.28` rad, i.e. ZERO mod 2*pi -- no constant convention factor.
5. The fitted slope equals `tau_shift` to 5 decimal places:
   `(0.30,45deg)` slope `0.13864` vs `0.13863`; `(0.50,45deg)` `0.58985` vs
   `0.58993`; `(0.30,15deg)` `0.33381` vs `0.33608`; `(0.50,15deg)` `1.00357`
   vs `1.00690`.
6. Applying the correction `G_fixed = G * exp(1j*w*tau_shift)` collapses the
   mid-band residual `|R - G|/|F|`:
   `gamma=0.90,theta=45,off=0.60`: `4.07e-2 -> 3.54e-4` (115x)
   `gamma=0.50,theta=45,off=0.60`: `6.05e-2 -> 1.70e-3` (36x)
   `gamma=0.90,theta=75,off=0.60`: `1.39e-1 -> 3.32e-3` (42x)
7. `w_trust` (smallest `w` above which `|R-G|/|F| < 1e-2` holds) drops:
   `gamma=0.50,theta=45,off=0.25`: `42.7 -> 7.2`;
   `gamma=0.90,theta=45,off=0.25`: `42.7 -> 7.2`;
   `gamma=0.90,theta=45,off=0.60`: `10.0 -> 2.5`.
8. NOT rescued by the fix (expected -- the uniform Airy/Pearcey arm's job, NOT
   this build's): near the principal axes (`theta=15deg`) and `theta=75deg` at
   `offset=0.25`, where `w_trust` stays `19`-`inf`.
9. The exact oracle REFUSES above `W_CEILING_SCHWINGER = 60.0`, so every
   falsifiable comparison must live below it.
10. Today `farfield_envelope_from_partition(..., FARFIELD_KERNEL_SUM_MINUS_GHOST)`
    raises `GhostDomainError` on any `w` grid reaching down to `w=0.5`, because
    the gate keys on `w_min` of the WHOLE grid. Probes must therefore use a
    grid whose minimum clears the gate, or call the primitive directly.

## In scope

- Locate and repair the delay-frame mismatch at its source. Decide explicitly
  whether the correction belongs in `geometry.ghost_kernel` (changing the
  primitive's documented frame) or in `channels.farfield_ghost_term` (keeping
  the primitive raw and normalising at composition). JUSTIFY the choice in the
  docstring; whichever is chosen, `ghost_kernel`'s documented contract and ALL
  its callers must end consistent.
- `find_referencing_symbols` + a `search_for_pattern` grep for every caller of
  `ghost_kernel` / `_ghost_delay` / `farfield_ghost_term` before changing any
  signature or semantics. Empty LSP results REQUIRE the grep cross-check.
- Fast regression tests that would have caught this:
  (a) a FRAME test: the ghost carrier's delay is expressed in the same frame as
      `partition.delays` (assert directly, not via a magnitude proxy);
  (b) a COLLAPSE test: `|R - G|/|F|` falls below a stated bar at the
      representative probes in fact 6, with the pre-fix values as the
      reachable-red witness;
  (c) a SELF-FALSIFICATION test: re-introducing the raw-frame carrier makes
      (a) and (b) fail, so neither is vacuous.

## Out of scope (do NOT touch)

- The `w_min` ghost gate (fact 10). Whether the ghost should be gated per-`w`
  rather than on the grid minimum is a PHYSICS decision that follows this
  repair; it is a separate build. Do not widen, relax, or re-key the gate.
- The uniform Airy/Pearcey arm and the cases in fact 8.
- Surrogate chart geometry, tiling, admission, and `w`-range selection.
- Any campaign, sweep, pilot, or engine run.
- `cogwheel/tests/test_lensing_exterior_admission.py` (just repaired
  independently; leave it alone).

## Acceptance

- New fast tests green, and each demonstrably reachable-red.
- `python -m pytest cogwheel/tests/test_lensing_chang_refsdal*.py -q` green.
- No change to the real-image path: `find_images`, `image_kernel`, `delay`,
  `morse_index` outputs bit-identical on a fixed probe set (assert it).
- Full suite green: driver-verified POST-BUILD, not in-build.

## Constraints

- HARD test-tier ceiling: any single test < 60 s, any test FILE < 5 min, fast
  tier. No slow-tier gates, no bulk sweeps, no hour-scale regressions in-build.
  Analytic or few-eval oracles only. Remember the gate runs your tests AGAIN.
- Every probe must sit BELOW `w = 60` (fact 9).
- Numerical accuracy dominates; units and conventions per AGENTS.md.
- numba compatibility preserved on accelerated paths.
- Spec/TODO workflow applies (behavior change in `cogwheel/`): `todo.d`
  fragment, `completed.d` on completion, `changelog.d` fragment, and a
  `spec_changelog.d` fragment if `SPEC.md` changes. Run
  `python scripts/render_fragments.py` after writing fragments.
- Branch `claude-dev` only. Never commit on `main`.
