# Librarian Short-Term Observations

## 2026-08-20 (low-w near-fold / wall-band chart-serve doc sync, INS-1-003)

- Scope: in-DAG sync for the `LowWDiffractiveChart` build (4-D reduced-coordinate
  low-w diffractive residual chart; offline-Schwinger trainer; likelihood serve
  auto-attach + chart-first Rung-P consult; 12th census SERVE_ROUTE). Working
  tree was UNCOMMITTED (HEAD = build brief 3a56e97) — driver commits.
- Deferred INS-1-003 fixed on both surfaces: DATA_CONTRACTS.yaml gained the
  `low_w_diffractive_chart` artifact entry (producer
  `scripts/train_low_w_diffractive_chart.py::main`; consumers
  `LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve` +
  `serve_route_census.classify_draw`; npz fields incl. scalar `derate` +
  per-cell `declined_mask`; schema `low_w_diffractive_v1`; content-hash
  provenance). schema_version 3.3.1 -> 3.4.0 (minor, new artifact).
  SPEC.md LOW-W DIFFRACTIVE RUNGS: "near-fold shell DECLINED" sentence flipped
  to the chart-serve statement (shell + wall band, union-band `covers`, de-rate
  sole margin, per-cell DECLINED -> exact-engine fall-through). spec_version
  0.49.1 -> 0.49.2 (patch). todo.d `lensing_low_w_near_fold_serve` moved to
  completed.d `2026-08-20_low_w_near_fold_chart_serve.md`; changelog.d entry
  added.
- NEW STALENESS CLASS (census route list): SPEC's ENGINE-FREE SERVE-ROUTE
  DEMAND CENSUS paragraph still said "exactly one of EIGHT MECE serve routes"
  with an 8-item `SERVE_ROUTES` list — stale since the born-carrier-only and
  diffractive-route builds (2026-08-18), not just this build's 12th route.
  Fixed to TWELVE with the full list + production-rung-ordered waterfall.
  Lesson: a census paragraph that enumerates route counts/lists must be
  re-checked after EVERY build that adds a route, not only when the census
  module itself changes.
- WALL-BAND RULING REVERSAL recorded: the todo fragment's interim owner ruling
  ("wall band SEPARATE, resolved via Schwinger") was superseded — 050d4cf
  reverted the gamma-domain fence, so the chart covers the UNION (shell OR
  wall). The completion record states this so future readers don't conflate the
  interim ruling with the shipped behavior.
- Repointed a PRIOR build's dangling wiki-link while here:
  `todo.d/lensing_training_campaign.md -> [[lensing_born_farfield_completion]]`
  (todo stem moved to `completed.d/2026-08-18_born_farfield_completion.md` a
  prior build left dangling) — trivial repoint, dropped the dangling set from 6
  to the 5 known permanent `[[FINDINGS F0xx]]` false dangles. Also edited
  `completed.d/2026-08-20_diffractive_certificate_interior.md`'s historical
  "`[[lensing_low_w_near_fold_serve]]`, still open" link to repoint to the new
  completed stem (cross-reference pointer, not a measured number — historical
  convention preserved).
- Sphinx: NO docs/source/ edits made — the chart is an internal trained-artifact
  loader like born_residual_chart (api.rst uses `:recursive:` autosummary over
  `cogwheel`; overview.rst is architecture-level and never enumerated the
  trained charts; `cogwheel/lensing/__init__.py` `__all__` unchanged).
- data_registry.yaml SKIPPED (correct): code does not import
  `scripts/data_registry.py` (`get_path`); trained charts load via
  importlib.resources hardcoded paths, and `born_residual_chart` is likewise
  absent from the registry — adding only the new chart would follow no pattern.
- sync_derived_docs.py "5 checks run, all OK" both before and after the
  DATA_CONTRACTS entry — its data_contracts check only validates DECLARED
  refs (module+function token), which all resolve.
- CONSUMER_GRAPH.json is stale (2026-08-17, only 5 loaders) — it has no
  loader for low_w_diffractive_chart yet, so consumer_graph advisory is inert
  this run. On regeneration the graph will likely flag
  `LensedRelativeBinningLikelihood.__init__` as an actual consumer of the
  chart (auto-attach; `__init__` is not private by the checker's dunder rule)
  — same latent pattern as born_residual_chart/ppgo_map auto-attach callers;
  watch for that noise, don't register a non-test consumer entry for it.
- Renderer dangling-set check: after my fragment edits the only remaining
  dangling `[[...]]` links are the 5 permanent `[[FINDINGS F0xx]]` false
  dangles (checker never taught that convention) — expected, untouched.
