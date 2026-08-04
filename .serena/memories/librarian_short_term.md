# Librarian Short-Term Observations

## Run: 2026-08-04 — census_dry_run.py addition (commit 97f7fc0)

**Scope:** Post-commit sync for commit `97f7fc0` (scripts: census_dry_run.py —
structural coverage = 100%).

**Changed file:** `scripts/census_dry_run.py` — new script, 366 lines.

**Triage result:**

The script stays entirely in `scripts/`, writes no disk artifacts (stdout only),
and introduces no new `cogwheel/` public API. The SCRIPTS/ REWRITE NO-OP RULE applies
for SPEC.md, DATA_CONTRACTS.yaml, and docs/source/.

The one stale surface found: `todo.d/lensing_coverage_map.md` Section D, which
contained the action directive "RUN A CHEAP DISCOVERY CENSUS EARLY". That census
has now been run — `census_dry_run.py` is the tool and the commit message reports
the result (100% structural coverage, breakdown by serve path). Section D was
updated to record:
- Census run on 2026-08-04, commit 97f7fc0, tool `scripts/census_dry_run.py`
- Result: 100% structural coverage, no unnamed regions
- Breakdown: Born exterior 71%, tube/far-field 15%, interior wedge 7%, lobe interior 7%, ppGO fold 0.1%
- Full-box campaign still stays last

`render_fragments.py` was run; `TODO.md` was updated.

**Surfaces confirmed NOT stale:**
- SPEC.md — no reference to discovery census or structural coverage
- DATA_CONTRACTS.yaml — no new disk artifacts in the script
- docs/source/ — scripts/ is not a cogwheel module; no API page needed
- FINDINGS.md — no new finding, no old finding invalidated

**Stale pattern this commit reveals:**
- Coverage map Section D's "RUN X EARLY" imperatives are action calls; when
  the action lands, check Section D and record the outcome. Specifically:
  the pattern "RUN A CHEAP X EARLY ... should not wait for it" signals a
  pending discovery-style step that will go stale on completion.

**Fragile cross-references to watch:**
- lensing_coverage_map.md Section D now cites commit 97f7fc0 and
  `scripts/census_dry_run.py` as the discovery census record. If that script
  is renamed or removed, update Section D.
