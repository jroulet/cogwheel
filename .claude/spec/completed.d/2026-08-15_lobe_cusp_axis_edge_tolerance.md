---
date: 2026-08-15
section: Lensing training
---

**`_lobe_cusp_axis_map` cusp-vs-edge guards relaxed to admit
cusp-coincident tile edges (F082).**

Reactive fix from a build brief (`.claude/handoff/
lobe_cusp_axis_edge_tolerance.md`), not a pre-existing backlog item: the
7a smoke run — the first end-to-end reach of the lobe-exterior training
path now that F081 unblocks its tiler — crashed in
`_lobe_cusp_axis_map` (`cogwheel/lensing/surrogate.py`) via
`LensAmplificationSurrogate.from_lobe_exterior_engine`. A lobe-exterior
tile's upper theta edge landed exactly on the theta = 0 cusp ray; the
guard compared `cusp_angle`/`theta_hi` with a strict float inequality,
and the two values differed only by 2.8e-17 of representation noise on a
mutual machine-zero, so the guard raised on a degenerate-but-valid
boundary tile rather than a genuine straddle.

Fix: a new dimensionless tolerance, `_CUSP_EDGE_COINCIDENCE_ULPS = 8`
(ULPs, not a tuned physical fudge — sized to comfortably cover
representation round-off while staying far below any real angular tile
width), lets a cusp within that many ULPs of the side-appropriate edge
be treated AS the cusp: `d` at that edge clamps to exactly `0.0` and the
`u = d**(2/3)` map anchors there. A cusp genuinely interior to the tile
beyond the tolerance still raises `ValueError` — the Professor confirmed
this remains unreachable in the current tiler, so the raise is a guard,
not a live path.

Both call sites (`from_lobe_engine`, `from_lobe_exterior_engine`) and the
subdivision path (`_lobe_child_boxes` in `surrogate_training.py`) share
the one guard, so the fix applies uniformly. Certified by
`test_lensing_surrogate_lobe.py`: `LobeCuspAxisMapEdgeCoincidenceTestCase`
(both-side edge-coincidence keep-map pins, the 7a machine-precision
sliver regression, a boundary trichotomy sweep) and
`LobeChildBoxesCoincidentEdgeTestCase` (caller-path pin,
straddle-propagation teeth), plus
`LobeCuspAxisMapEdgeCoincidenceSelfFalsificationTestCase` narrowing the
ULP band to confirm a genuine straddle just past the tolerance still
raises.

Implementation-level guard tolerance; no SPEC.md or DATA_CONTRACTS.yaml
change — neither surface described the old guard's exact-inequality
criterion (nothing to go stale), and `_CUSP_EDGE_COINCIDENCE_ULPS` is an
in-memory training-time constant, not a disk-artifact field. No todo.d
fragment existed for this build (it was reactive, spawned directly from
the crash), so there is nothing to close here beyond this record.

DRIVER FOLLOW-UP (post-build, not measured in-build): the 7a smoke run
that hit this crash needs to be re-run past this point to confirm the
lobe-exterior training path completes end to end.
