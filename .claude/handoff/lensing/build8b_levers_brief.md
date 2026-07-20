# Build 8b-levers — Two certified engine micro-levers (Newton caustic shortcut; contraction fusion)

## Mission

Two small, independently-certifiable engine accelerations, both shelved
since Builds 3d-3g with their obligations already recorded
(`likelihood_envelope-surrogate.md` item 2), now promoted because the
surrogate made them the serving-path bottleneck:

1. **Nearest-caustic Newton shortcut** (`geometry.py`,
   `nearest_caustic_point`): replace the current dense/coarse caustic
   search with a Newton polish from a cheap seed. Measured cost today
   ~1.9 ms per call (the dominant term of `geometry_partition`'s
   5.6 ms, which is itself the floor of the surrogate serving path);
   target ~0.3 ms. OBLIGATIONS (recorded, binding): VALUE-PRESERVING
   (the returned nearest point/distance must match the current
   implementation to tight tolerance on a certification sweep — the
   caustic distance feeds domain gates and switch logic) and
   BRANCH-INVARIANT (identical behavior across the astroid and BOTH
   deltoid lobes, including the branch argument and wedge handling;
   Newton must not jump lobes — seed per lobe and take the min).
2. **Weight-vector contraction fusion** (`operator.py`): fuse the
   per-order weight-vector contraction loops (~2 -> ~1 ms on the
   crown). OBLIGATION: every refusal-relevant quantity
   (cancellation_ratio, estimated_tail, orders) BYTE-UNCHANGED —
   F005-style re-certification; the F_op/F_op_grid outputs must stay
   bit-identical on the certified sweep (the 7a/7b bit-freeze pins
   must pass untouched).

## Context (measured 2026-07-20)

Surrogate-served lensed lnlike = 9.72 ms on generic proposals = 6.2x
the 1.57 ms unlensed floor (owner target 2-4x). Budget: 1.6 unlensed
work + 5.6 geometry_partition + 0.4 spline + ~2 lens contraction.
These two levers project ~9.7 -> ~7 ms (~4.5x). The engine modules
are otherwise FROZEN certified code — these are the first sanctioned
engine edits since Build 7a, and the certification bar is
correspondingly high.

## Hard fences

- ONLY `geometry.py` (`nearest_caustic_point` + its private helpers)
  and `operator.py` (the contraction internals). NO changes to
  `_schwinger.py`, `_hyp1f1.py`, `_gauge.py`, `_dd.py`, `channels.py`,
  or any likelihood/prior/surrogate file.
- NO threshold, refusal constant, or API change anywhere.
- The 7a/7b/8a bit-freeze pins (FROZEN_FOP_PINS, HEAD critical-point
  pins in test_lensing_saddle_geometry, crown byte-identity in
  test_lensing_surrogate) MUST pass unmodified — they are the
  certification instrument, not reconcilable collateral.

## Acceptance

1. Newton shortcut: a certification sweep test (both parities, all
   lobes/branches, on- and off-wedge, near-cusp seeds) pinning
   value-preservation vs the CURRENT implementation (run the old path
   side-by-side via the established importlib-HEAD idiom or a frozen
   reference table) at tolerance <=1e-10 on distance and theta; a
   timing probe showing ~0.3 ms class; the existing
   HEAD_NEAREST_CAUSTIC_PINS exact-equality tests still pass (if
   exact equality is impossible under the new iteration, STOP and
   report — do not weaken a pin).
2. Contraction fusion: bit-identical F_op/F_op_grid outputs and
   refusal quantities on the certified sweep; the F010 py_func-chain
   falsifications still red-capable; crown timing improves measurably.
3. Full lensing-suite regression green (driver-verified post-build);
   serving-path re-measure (driver): geometry_partition and floor
   ratio reported before/after.
