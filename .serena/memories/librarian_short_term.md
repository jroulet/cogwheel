# Librarian Short-Term Observations

## 2026-08-17 (serve_route_census doc sync)

Scope: audited docs after build `serve_route_census` shipped
`cogwheel/lensing/serve_route_census.py` (+ CLI `scripts/serve_route_census.py`
+ test suite) — order-7a step 1 of `todo.d/lensing_training_campaign`.
Inspector PASS, no new findings (INS-1-001 label reused generically across
builds — this instance was an already-resolved dead-code flag, not doc
staleness; confirmed INS-N-00N labels are NOT persistent cross-build IDs,
unlike F0xx findings).

What was stale: SPEC.md's giant single-line Microlensing-engine table row
(~line 53) had NOT been updated for the new module, matching the established
pattern where every prior engine-free census/build module (tiling_census,
tube_d2_fold, saddle_lobe_edge_shell, low_w_extrapolation) gets an inline
dated/build-tagged paragraph in that same row. `sync_derived_docs.py`
reported "5 checks, all OK" with zero diff — it only checks mechanical/
structural completeness (module lists, findings sync), NOT narrative
paragraph-per-module depth, so a clean sync-script run does NOT mean SPEC.md
narrative is current. This is now confirmed on a THIRD build (tiling_census
2026-08-14, saddle_tube_fundamental_training 2026-08-15, serve_route_census
2026-08-17) — treat "new lensing census/training module shipped" as a
standing trigger to check the big engine row regardless of tooling output.

Fixed: inserted a new dated paragraph into SPEC.md's engine row (between
"...training-arc selection)." and "LOW-W FLAT EXTRAPOLATION:"), grounded
entirely in the module's own docstring/code/tests (7 SERVE_ROUTES waterfall,
ROUTE_KINDS D2-invariance, residual_demand's F073 caustic_rho gauge guard,
engine-free-by-construction import discipline, HEAD demand report numbers).
Created `spec_changelog.d/2026-08-17_serve_route_census.md` (bump: minor,
following the `2026-08-14_tiling_census.md` precedent exactly — new
production module + pipeline step = minor, not patch). Created
`completed.d/2026-08-17_serve_route_census.md` worded narrowly to document
only the module/doc-sync as done, explicitly stating steps 2-4 of
`lensing_training_campaign` and all of `lensing_no_engine_census` (order-7b)
remain open — did NOT touch either todo.d fragment since the broader
campaign is unfinished.

Skipped (with reason):
- DATA_CONTRACTS.yaml — no entry needed. Confirmed via precedent: the sibling
  `tiling_census.py` (same diagnostic-report-CLI shape, writes to user-chosen
  `--out`, no fixed pipeline consumer) has no contract entry either. Census/
  report CLI tools are not registered data-contract artifacts in this repo.
- docs/source/api.rst — no edit. `:recursive:` autosummary on the single
  `cogwheel` top-level entry auto-discovers all submodules including new ones
  under existing subpackages (`cogwheel.lensing`); api.rst only needs a new
  entry when a NEW top-level package is added, not a new submodule of an
  existing one.
- docs/source/overview.rst — no edit. Its "Microlensing engine" section is
  pitched at the public-API/architecture level (LensedWaveformGenerator,
  LensedRelativeBinningLikelihood, ChangRefsdalChannels) and deliberately
  omits internal training/diagnostic infrastructure — confirmed neither
  tiling_census nor surrogate_census are mentioned there either, so
  serve_route_census (also internal training support, not public API)
  follows the same established precedent of exclusion.
- todo.d/lensing_training_campaign.md, todo.d/lensing_no_engine_census.md —
  left open, not closed/deleted. Only step 1 of 4 is done.

render_fragments.py ran clean: SPEC_CHANGELOG.md and COMPLETED.md updated,
spec_version bumped 0.44.0 -> 0.45.0. The 5 dangling-wiki-link warnings it
printed are the KNOWN pre-existing false-positive family (check_wiki_links
never learned the `[[FINDINGS F0xx]]` convention — see prior memory entry
"THIRD OCCURRENCE GETS A FRAGMENT") — none of the 5 listed fragments are
mine; do not re-fix, a todo.d fragment already exists for the tooling gap.
git status after render showed no stray tidy_advisory.json/foreman_lite.json
diff this time (that side-effect is intermittent, not universal — still
worth checking every run).

Session note: working directory for this session is the git worktree
`/home/tejaswi/Work/cogwheel-claude-dev`, not `/home/tejaswi/Work/cogwheel`
— `git -C <wrong path>` fails with "not a work tree". Use
`mcp__serena__execute_shell_command` (implicit correct cwd) for git, not
Bash with an explicit wrong `-C`/`cd` path.
