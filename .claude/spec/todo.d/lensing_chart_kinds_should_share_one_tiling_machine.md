---
section: Backlog
---

- **EVERY CHART KIND REIMPLEMENTS TILING, SUBDIVISION AND PROBING — make it
  ONE parameterised machine** `[→ spec]` — owner-directed 2026-08-07. Two
  symptoms of one cause.

  (a) **Per-kind subdividers.** There were TWO near-duplicate single-level
  subdividers (`_subdivide_farfield_tile`, `_subdivide_wedge_tile`); the
  2026-08-07 build unified them into one generic `_subdivide_tile`
  parameterised by `(split_children, build_child, gate_kind, eps_bar,
  admit_child)`, with the two names kept as thin wrappers. That was the right
  move and it is HALF DONE: `LobeInteriorChart` still has NO subdivider at all
  (see [[lensing_saddle_forensics]] item b), and `TubeChart` has none either.
  A gated lobe or tube tile becomes a ladder-served gap with no recourse.

  Adding lobe support must NOT mean a third copy — it is now a
  splitter/builder/gate triple. The general shape wants OOP: a chart kind
  declares its coordinate map, its tiler, its splitter, its admission
  predicate and its eps bar; the tiling/subdivision/gating engine is written
  ONCE against that interface. Today those five things are scattered across
  free functions keyed by string region names (`'wedge_interior'`,
  `farfield`, `lobe`), which is why each new kind re-derives the machinery.

  (b) **Probes are not the production path.** Every measurement in this
  program has been made by a hand-rolled scratchpad probe that re-creates
  what the trainer does, because the training path cannot be invoked for one
  region ([[lensing_training_path_cannot_be_run_per_region]]). MEASURED COST:
  a probe that reimplemented the subdivider agreed with a misreading of the
  code rather than the code; a probe that transcribed tile bounds rounded to
  4 decimals overshot `pi/2` and silently produced complex output; and every
  schema change re-invalidates probe measurements
  ([[lensing_wedge_probe_charts_need_retraining_under_v3]]). A probe must be a
  THIN CALLER of the production tiler/subdivider/gate, never a parallel
  implementation — the same DRY rule the codebase already enforces for
  `r_deltoid` (one authoritative `_lobe_boundary_radius`) and for the delay
  frame (single-sourced via `_frame_delays` after it had drifted at four
  sites).

  ACCEPTANCE: one subdivision/tiling implementation, exercised by ALL chart
  kinds including lobe and tube; a region-scoped training entry point that
  probes call directly; and a test asserting that a probe-built chart and a
  trainer-built chart for the same tile are byte-identical. That last one is
  the falsifiable part — without it "the probe uses production code" is a
  claim, not a property.
