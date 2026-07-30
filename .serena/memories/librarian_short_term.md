## 2026-07-30 post-commit sync — build 1c, analytic cusp vertex + y''' (commit b9c3ed6)

Scope: sync SPEC.md to b9c3ed6. Diff: `geometry.py` (+ new public
`caustic_third_derivative`; private `_caustic_cascade` refactor;
`caustic_derivatives` 2-tuple byte-identical), `_pearcey_cusp.py`
(`_cusp_vertex` now analytic-root via brentq on `y'.y''`, no FD scan),
`test_lensing_airy_fold.py`/`test_lensing_caustic_derivatives.py`/
`test_lensing_caustic_cusps.py`, `FINDINGS.md` (+F042 resolution, +F043).

FIXED (item 1, NEW-PUBLIC-NAME gate):
- SPEC.md row 53 (microlensing engine), geometry public-name list: added
  `caustic_third_derivative` right after `caustic_derivatives`, before
  `caustic_speed`/`caustic_curvature_radius`/`fold_opening_direction`.
  Verified against `geometry.caustic_third_derivative`'s actual signature
  `(gamma, theta, *, kappa=0.0, branch=1)` and docstring (extends the
  cascade one order via `_caustic_cascade`; `y_triple_prime` shape
  `(2,)`/`(2,N)`). Left the "Certified by test_lensing_caustic_cusps.py"
  clause's own `caustic_derivatives`/... list untouched — task scoped the
  edit to the "geometry public-name list" only, not the certified-by list
  (also confirmed `caustic_third_derivative` is in fact tested there:
  `test_lensing_caustic_derivatives.py` STAGE 2, 40-dps mpmath oracle, mock
  self-falsification — but did not add it to the certified-by prose since
  not asked and that list already reads as a group label, not exhaustive).
- New fragment `spec_changelog.d/2026-07-30_caustic_third_derivative.md`
  (bump: minor — new public API) -> rendered `spec_version 0.29.0`.
  `sync_derived_docs.py --check` re-ran: only pre-existing unrelated
  `consumer_graph` warnings on `lens_amplification_surrogate` test
  consumers (present before my edit too, not caused by it, out of scope);
  nothing flagged for `geometry.py`/`caustic_third_derivative`.

VERIFIED, NOT touched:
- Item 2 (Pearcey-arm prose, old FD cusp scan): grepped SPEC.md for
  `finite.difference|speed scan|golden.section|cusp vertex|_cusp_vertex` —
  zero matches anywhere in the file. SPEC.md's `_pearcey_cusp.py` mention
  never described the internal cusp-finding method (FD or analytic) at
  all, so per the task's own instruction ("if it does not describe the
  internal method, leave it") no edit was needed. Read `_cusp_vertex`'s
  actual body to confirm the analytic-root behavior for future reference:
  brentq on `slope(phase) = y'.y''` in the `phase = theta - beta` frame,
  parity-gated bracketing (astroid exact at {0, pi/2, pi, 3pi/2}; saddle
  wedge-tip served, wedge-edge refused to None).
- Item 3 (FINDINGS F042/F043): both headers present and correctly tagged
  (`## F042 — ... RESOLVED, re-based (2026-07-29)`, `## F043 — ...
  (2026-07-30)`); F042's cross-ref `[[lensing_collocation_from_local_scales]]`
  resolves to a real, still-present todo.d fragment. Did not rewrite either
  (hand-maintained, per scope).
- `.claude/spec/todo.d/lensing_analytic_derivatives.md`: noticed item 1
  (`_cusp_vertex` serving path) and the "extend cascade to y'''" paragraph
  are now DONE by 1c, but items 2-5 (surrogate_training.py consumers:
  `_branch_speed_profile`, `_find_cusps`, `_probe_arc_side`, `_tube_normal`)
  are not — this is a multi-item backlog fragment, correctly stays in
  todo.d until every item lands. Left untouched: TODO fragment authorship
  is Architect/Coder's, not Librarian's, and the task briefing didn't ask
  for it.

Post-commit mode: trigger `.claude/sync_issues.json` (untracked, not in
`git ls-files`) present at start listing 15 pending commits back through
b9c3ed6; committed doc fixes as `docs: post-commit sync (1c)` (staged only
the 3 files I changed — SPEC.md, SPEC_CHANGELOG.md, the new spec_changelog.d
fragment), then deleted the trigger file. Left
`.claude/sdk/_retry_until_launch.sh` (pre-existing unstaged modification,
unrelated to doc sync, a code file) untouched and unstaged — not mine to
touch or commit.

Mechanics: `render_fragments.py` again touched `.claude/tidy_advisory.json`
(commit-hash/timestamp churn) — reverted via `git checkout --`, not
committed, same as every prior session (see `librarian_knowledge.md`).
`search_for_pattern` on SPEC.md returns the ENTIRE table row as one string
(rows are single un-wrapped lines) — a multi-alternation pattern with hits
on the same line duplicates the same full-line context per match; harmless
but don't be surprised by "2 identical results" for 1 line.
