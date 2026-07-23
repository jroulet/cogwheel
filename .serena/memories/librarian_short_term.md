## Librarian run — 2026-07-23, ppGO map truncation-on-refusal (commit dc984c1)

Scope: WP "ppGO map: truncation-on-refusal + per-cell w_ceiling schema"
+ "ppGO map consumers: cell-ceiling band-split guard + ceiling-aware
strata trim". Files touched: cogwheel/lensing/{ppgo_map,likelihood,
surrogate_training}.py, DATA_CONTRACTS.yaml (schema 0.1.0 -> 0.1.1
already bumped in-commit), TODO.md fragment (surrogate_component-
representation-8hb.md, new, "In progress").

What was stale:
- `contracts_changelog.d/2026-07-23_feat__ppgo_map_truncation_on_refusal__pe.md`
  was a **placeholder stub** left by a commit-preflight hook ("Auto-
  generated ... Librarian should refine this entry from the commit
  diff") — the coder/commit hook staged DATA_CONTRACTS.yaml schema
  bump + description edit but only auto-stubbed the changelog fragment
  instead of writing real prose. This is a NEW pattern worth watching:
  check every `contracts_changelog.d/`/`spec_changelog.d/` fragment
  touched in the diff for this stub marker, not just fragment
  *presence* — a fragment can exist and render (canonical file
  "updated" cleanly) while still being a content stub. Refined it from
  the DATA_CONTRACTS.yaml diff + coder change reports.
- DATA_CONTRACTS.yaml `certified_ppgo_map` consumers list was missing
  two real consumers: `cogwheel/lensing/likelihood.py::
  LensedRelativeBinningLikelihood._ppgo_band_split` (and the sibling
  `_ppgo_cell_ceiling`) and `cogwheel/lensing/surrogate_training.py::
  train` (via `_stratum_ppgo_boundary`/`_stratum_ppgo_ceiling`) — both
  call `get_certified_ppgo_map()` directly and read `.w_trust`/
  `.w_ceiling` per cell. Only `ppgo_map.py::use_certified_ppgo_map` was
  on record. This gap PRE-DATES this build (band-split existed since
  8h-a) — not something this specific commit introduced, but squarely
  in the "consumers match actual code" checklist so I fixed it anyway.
  Added both, wrote a `bump: minor` fragment (new consumer entries =
  MINOR per the schema versioning key in the yaml header), re-rendered
  -> schema_version 0.1.1 -> 0.2.0 (minor bump resets patch digit, not
  a bug).

What was NOT stale (verified, not just assumed):
- SPEC.md prose: zero mentions of ppgo/certified_ppgo_map/band-split/
  w_cert anywhere — this whole mechanism lives only in DATA_CONTRACTS.
  yaml, never promoted to SPEC.md narrative. Consistent with the
  existing "low-level serving detail, don't manufacture a blurb"
  pattern from prior runs — extends it to internal *dispatch* logic,
  not just perf numbers.
- docs/source/*: zero mentions of ppgo/lensing internals in
  crash_course.rst or api.rst; overview.rst's lensing paragraphs only
  cover the public ChangRefsdalChannels/LensedWaveformGenerator/
  LensedRelativeBinningLikelihood surface, untouched by this build.
  No docs/source edits -> no Sphinx rebuild needed this run.
- data_registry.yaml: no `scripts/data_registry.py` import found in
  cogwheel/ — stays skeleton per the standing rule.
- TODO.md/COMPLETED.md: the new todo.d fragment (8h-b) is correctly
  "In progress" — it's a multi-part program (per-cell ceilings [done
  this build], fold-gated ghost-pair subtraction [not done], caustic-
  fixed interiors [not done]); nothing to move to completed.d yet.
  Confirms the existing "multi-part program stays open" heuristic.

Tooling notes:
- `scripts/sync_derived_docs.py` and `scripts/regenerate_consumer_graph.py`
  exist in this project (I'd missed sync_derived_docs.py in a prior
  run's memory — it's real, use it as Step 0). Needs `jedi` (present in
  conda env `cogwheel-newlal`, absent from the default uv-managed
  `python` on PATH) and `rg`/ripgrep (absent from this machine
  entirely, both on PATH and next to the cogwheel-newlal interpreter)
  for the consumer-graph regeneration step. Without `rg`,
  `regenerate_consumer_graph.py` hard-fails; `sync_derived_docs.py`
  still runs against the STALE cached CONSUMER_GRAPH.json and is
  useful for everything except catching brand-new call sites. Use
  `/home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python` for both
  scripts on this machine, not the bare `python` on PATH.
- sync_derived_docs.py flagged 4 test-only consumers of
  lens_amplification_surrogate (a DIFFERENT artifact, pre-existing,
  unrelated to this build) as "not in DATA_CONTRACTS.yaml — confirm
  transient". Left untouched: matches the project's standing
  convention of not listing test-file consumers (e.g. posterior_samples
  consumers list is production-only too), and it's out of scope for
  this build's diff. Flag for whoever next touches that artifact's
  contract entry, don't fix opportunistically.

Fragile cross-reference to watch: the contracts_changelog.d stub
pattern (commit-preflight auto-stub) — if this recurs on future
builds, check the ACTUAL BODY of every changed changelog fragment, not
just its existence, before assuming render_fragments.py output is
final prose.
