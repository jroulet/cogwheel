## Last session: 2026-08-06 post-commit sync (--post-commit 2a23e14, range 842e8ad..2a23e14, ~16 commits)

Scope: unusual backlog — a build stranded at its tree gate meant its in-DAG
Librarian never ran, AND the driver's own spec-only commits (`git add -A`)
inadvertently carried the stranded build's production code. Verified
independently via `git diff 842e8ad..2a23e14 --stat -- cogwheel/ scripts/`:
real code changes were `cogwheel/lensing/surrogate.py` (+186/-),
`surrogate_training.py` (+361/-), three test files, and
`scripts/render_fragments.py` (+52, new dangling-`[[link]]` check). Commit
messages in this range are all `spec:`/`test:`/`handoff:` — none hinted at
the code payload; only reading the actual diff surfaced it. This is a new
variant of the "don't trust commit messages" pattern already in
librarian_knowledge — here the mislabeling was structural (tree-gate stranding
+ `git add -A`), not a hand-fix hiding in a feature commit.

### What was stale
1. SPEC.md's `InteriorWedgeChart` paragraph (inside the giant single-line
   "Microlensing engine" table cell, line 55) had an "ARC-LENGTH ANGULAR AXIS"
   subsection describing `s_fine` = cumulative trapezoid of
   `geometry.caustic_speed`. Now wrong: the wedge angular spline axis is
   `u = d**(2/3)` (`d` = angular distance to the NEAR astroid cusp), split at
   the caustic waist `theta_waist = argmin_theta r_caustic` (new
   `_wedge_theta_waist`), per-tile `axis_origin` verified against the engine's
   own classification. Renamed subsection to "CUSP-ADAPTED ANGULAR AXIS" and
   rewrote it; `_WEDGE_AXIS_SCHEMA` bumped `v1 -> v2` in the same sentence.
   `spec_changelog.d/2026-08-06_wedge-cusp-adapted-angular-axis.md` (bump
   patch). Confirmed via targeted grep that NO other SPEC.md location
   mentions `InteriorWedgeChart`/`theta_wedge`/wedge arc length — one edit
   site, not several.
2. DATA_CONTRACTS.yaml `lens_amplification_surrogate`'s `InteriorWedgeChart`
   sentence: same v1->v2 tag bump plus the `theta_to_s`/`s_grid` field
   description rewritten from arc-length to cusp-adapted `u`. Field NAMES on
   disk are unchanged (`theta_to_s`, `s_grid`, shape `(2, 2001)`) — only what
   they encode changed; a stale v1 artifact hard-refuses (no migration).
   Driver explicitly said do NOT upgrade the separate interior-coverage claim
   ("currently UNSERVED") in the same entry — left that sentence's substance
   alone, only touched the schema-tag paragraph.
   `contracts_changelog.d/2026-08-06_wedge-axis-schema-v2-cusp-adapted.md`
   (bump patch).
3. Caught and fixed my OWN stray reference while writing fragment 2: a plain-
   text (non-`[[link]]`) pointer inside DATA_CONTRACTS.yaml's `FarFieldChart`
   sentence — "see FINDINGS/todo.d lensing_wedge_charts_fail_the_eps_bar" —
   pointed at a fragment file DELETED earlier in this same commit range
   (consolidated by `2d9c52b` into `lensing_wedge_angular_axis_is_cusp_
   singular.md`, confirmed via `git show 2d9c52b --stat`). Repointed to the
   surviving fragment. This class of dangling reference is INVISIBLE to
   `render_fragments.py`'s new dangling-`[[link]]` check, which only scans
   `[[wiki-link]]` syntax inside `todo.d`/`completed.d` fragments — a plain-
   English fragment-name mention inside DATA_CONTRACTS.yaml or a changelog
   fragment is NOT covered. Also fixed the same stale name in both of my OWN
   new changelog fragments before rendering (would have shipped a second
   dangling mention otherwise) — re-read fragment bodies for cross-references
   to recently-deleted files, not just for typos, before running
   render_fragments.py.

### Cross-reference / coherence check (the driver's specific ask #4)
`render_fragments.py`'s dangling-`[[link]]` check ran clean (0 dangling) both
before and after my edits — the four links flagged as dangling in commit
`2d9c52b`'s own message were fixed two commits later in `42328c9` ("repoint
four dangling wiki-links; make render_fragments catch them"), IN RANGE. Read
the full bodies of `lensing_wedge_u_map_stored_in_arclength_fields.md`,
`lensing_wedge_angular_axis_is_cusp_singular.md`,
`lensing_r_caustic_should_root_find_not_scan.md`, and
`lensing_coordinate_program_spine.md`: all four are internally coherent — the
program-spine fragment correctly still says step 1 is "IN FLIGHT" (not DONE),
the root-find fragment correctly says it "MUST NOT land" until step 1 lands,
and the arclength-field-naming fragment correctly frames itself as knowingly-
incurred debt to retire alongside the root-find. Did NOT retire any todo.d
fragment (driver was explicit: implementation landed but production-scale
accuracy is not yet demonstrated).

### Process notes
- `sync_derived_docs.py` reported the same 4 test-only-caller consumer-graph
  warnings for `lens_amplification_surrogate` as at least three prior
  sessions (per my own librarian_knowledge entry) — FOURTH session seeing
  this exact unchanged warning set with zero diff produced. Still haven't
  escalated to the contract owner as my own prior notes recommended; flagging
  a fourth time. At some point this stops being "worth re-noting" and should
  just get fixed or explicitly silenced.
- No stray `.claude/tidy_advisory.json` diff survived to the final
  `render_fragments.py` pass this time (reverted the first occurrence with
  `git checkout --`, and the second full run after my cross-reference fixes
  produced no stray diff at all) — reconfirms this is a sometimes-flaky
  render_fragments.py side effect, not a fixed rule of every invocation.
- `docs/source/` has ZERO mentions of "wedge" anywhere — confirmed via grep
  before deciding to skip the Sphinx surface entirely; this internal lensing-
  dev coordinate program has never touched Sphinx docs and this session is no
  exception.
- SPEC.md and DATA_CONTRACTS.yaml both store this subsystem's prose as ONE
  giant single-line table-cell / YAML-string value each; `search_for_pattern`
  with a broad multi-alternation regex against either file reliably blows the
  answer-size limit even for a handful of real matches (same failure mode
  noted in librarian_knowledge for SPEC.md rows, now also confirmed for
  DATA_CONTRACTS.yaml's `lens_amplification_surrogate` description) — go
  straight to a narrow, specific substring (e.g. the exact schema-tag string)
  instead of iterating broader patterns.
