2026-08-15 doc-sync for build `saddle_tube_fundamental_training` (work
package: "Saddle tube fundamental-arc trim + per-arc lobe-edge shell
(F081)", uncommitted diff spanning `cogwheel/lensing/surrogate_training.py`
+ `tiling_census.py` + mirrored scripts/tests, 19 files):

- SPEC.md: fixed the stale TUBE D2 GAUGE-IMAGE FOLD sub-paragraph (in the
  giant single-line "Microlensing engine" table row) that said saddle
  training "remains `arcs[:max_tube_arcs]` (fundamental-set trim owed
  pre-campaign)" — the config field `TrainingConfig.max_tube_arcs` is now
  fully REMOVED from the dataclass, and the saddle branch of
  `_tube_training_arcs` derives a D2-orbit partition from the fold law
  (midpoint-angle clustering, tolerance `max(1e-3, 0.25*min_width)`,
  typically 6->3 representative arcs) matching the astroid's existing
  4->1 pattern. Added a new sentence describing the F081 fix riding
  along in the same build (`min_eta_max` now sizes saddle lobe
  admissions + deltoid far-field inner edge, replacing the band-wide
  `max_eta_max` that starved them; `max_eta_max` still correctly sizes
  the tube w-grid cap + astroid interior-skip/wedge extent). Verified via
  `search_for_pattern` that exactly one `max_tube_arcs` mention survives
  in SPEC.md, correctly stating the knob is retired.
- FINDINGS.md: F081 was the LAST finding in the file (ends at EOF, line
  4452) with no RESOLVED marker — appended one confirming BOTH named
  defects (config: heterogeneous r_min from training all 6 arcs; wiring:
  isotropic `max()`) are fixed, referencing the new completed.d fragment.
- Fragment chain for a build that closes a `[→ spec]`-tagged todo.d item:
  created `spec_changelog.d/2026-08-15_saddle_tube_fundamental_training.md`
  (bump: minor, matching the two 2026-08-14 sibling fragments
  tube_d2_fold/tiling_census which were also minor), `git rm`'d
  `todo.d/lensing_saddle_tube_fundamental_training.md`, created
  `completed.d/2026-08-15_lensing_saddle_tube_fundamental_training.md`
  (section: "Lensing serving", matching the 2026-08-14_lensing_tube_d2_fold
  sibling). Checked `search_for_pattern` across `.claude/spec` for the
  todo stem before deleting it — zero other fragments referenced it
  (no `depends_on:` or `[[...]]` backlinks to repoint).
- DATA_CONTRACTS.yaml: confirmed zero matches for
  `max_tube_arcs|TrainingConfig|_tube_training_arcs` — this build changes
  in-memory training/selection logic only, produces no new disk artifact,
  needs no contract entry.
- docs/source/*.rst: overview.rst's "Microlensing engine" paragraph
  (lines ~85-90) is architecture/API-level only, no TrainingConfig/tube-arc
  detail — confirmed via read, no changes, no Sphinx rebuild needed (docs/
  source/ untouched this run).
- `python scripts/render_fragments.py` bumped SPEC.md to 0.44.0, updated
  SPEC_CHANGELOG.md/COMPLETED.md/TODO.md cleanly. Same stray-diff side
  effect as prior sessions hit `.claude/tidy_advisory.json` (NOT
  `foreman_lite.json` this time) — reverted via `git checkout --`. The
  5 pre-existing dangling `[[FINDINGS Fxxx]]` wiki-links the render script
  warns about (F070, F069, F072, F071) are the same known pre-existing
  noise from 2026-08-13-dated fragments, unrelated to this batch — left
  untouched, matches prior-session assessment.
- PATTERN CONFIRMED (extends the "SPEC ENTRIES THAT CITE A FUNCTION BY
  NAME GO STALE SILENTLY" family from long-term memory): a SPEC sentence
  citing a specific CONFIG FIELD NAME (`max_tube_arcs`) goes stale the
  moment a later build retires that field from the dataclass entirely,
  not just when its value/behavior changes. Same detection method applies
  (grep the literal identifier across SPEC.md) — worth generalizing to
  "any identifier SPEC.md names — function, class, or config field — needs
  a grep check whenever the corresponding code file is in the diff."
- Left untouched (not mine, concurrent in-build agent state): all other
  agents' `.claude/agent_state/*.json` and `.serena/memories/
  {architect,coder,inspector,professor,test_dev}_short_term.md`, plus the
  untracked `.claude/handoff/saddle_tube_fundamental_training.md` build
  brief.
