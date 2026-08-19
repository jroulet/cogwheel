---
date: 2026-08-18
section: Backlog
---

- **Tube trainer resolvable-sub-arc trim SHIPPED — the F083 knee-scan
  is production, parity-gated, DRY** `[→ spec]` — build
  tube_trainer_subarc_trim: `_trim_tube_arc` in surrogate_training
  (constants 0.6/0.20/0.05 + the 80-point binding-corner scan carried
  VERBATIM from the F083 fixture, which now imports the production
  helper — fixture copy retired), wired after eta sizing in
  `_train_band_charts`'s per-arc loop; PARITY-ONLY gate (trim iff
  parity==+1; saddle arcs byte-identical — parity is a topological
  invariant and the saddle's disjoint deltoids admit no coherent
  global knee scan). The binding-corner bracket's monotone-nesting
  assumption keeps its unconditional falsifier: refused==0 over EVERY
  build node. DRIVER SPOT-CHECK (completion measurement, production
  density gamma=0.4, n_theta=7, f_max=0.40, f_floor=0.12): trim
  [0.1370,1.4780] -> [0.5376,1.0214], refused=0,
  eps_band=0.1084 <= 0.15 — matching the trimmed sweep's 0.1084 at
  this config to 4 decimals (the promoted helper reproduces the
  fixture-derived trim exactly). Part0: the three trim constants
  allowlisted with F083 provenance. Astroid campaign legs UNBLOCKED;
  the campaign fragment's dependency list is now EMPTY.
