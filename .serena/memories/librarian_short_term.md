# Librarian Short-Term Observations

## 2026-07-30 run — WP1 (_WEDGE_EPS deletion, analytic _tube_normal, docstring fixes)

Scope: commit 145cec3, work package "Delete _WEDGE_EPS, make _tube_normal
analytic, fix false docstrings" (build 1d of the caustic-relative-coordinates
program). Files touched were cogwheel/lensing/{surrogate_training.py,
chang_refsdal/geometry.py} + 3 test files + agent-state/memory noise.

What was stale: TWO todo.d fragments each carry their own numbered checklist
for the SAME piece of work (lensing_caustic_relative_coordinates.md step 1d,
and lensing_analytic_derivatives.md's item 5) — both needed a DONE marker
pointing at the same commit. This is a recurring shape in this repo: a step
is cross-referenced between an "ordering" fragment and an "inventory"
fragment ([[link]] syntax), and completing it means editing both, not one.
Check for `[[fragment_name]]` backlinks whenever marking a step done — the
inventory fragment is easy to miss since it doesn't have "1d" in its own
numbering (its numbering is 1-5 by target, not by build letter).

Also confirmed (did NOT touch, verified only): SPEC.md's microlensing-engine
row already described "estimators surrogate_training.py retired in favor of
the analytic geometry cascade" generically from an earlier build (pre-dates
1d) — no edit needed since it never named _tube_normal specifically.
DATA_CONTRACTS.yaml's surrogate_training.py entry unaffected (no new/changed
disk artifact). docs/source/ has zero hits for _WEDGE_EPS / tube_normal /
wedge / surrogate_training — overview.rst's lensing paragraph is pitched at
architecture level and doesn't reach into private training-path helpers.

Surprise: grepping the sibling target items (_branch_speed_profile,
_find_cusps, _CUSP_SPEED_REL_FRAC) showed they STILL EXIST in
surrogate_training.py despite looking like retired targets 2/3 in
lensing_analytic_derivatives.md. Do not mark those DONE — the fragment's own
"Carried forward from build 1b" section explains they were deliberately
repurposed (cusp-window sizing, not cusp-location estimation) and left open
pending the F040 cusp-window schema build. Read the "carried forward" /
caveat subsections before trusting a numbered list item looks satisfied by
grep alone.

Pre-existing gap noticed but left alone (out of this build's scope): todo.d's
lensing_caustic_relative_coordinates.md steps 1b and 1c are already shipped
(commits 00bf8ae, b9c3ed6) but carry no "DONE" annotation and have no
completed.d fragments — only 1a does. A prior librarian post-commit sync
(b1f1887) explicitly checked SPEC.md needed no edit for that work but didn't
address the todo.d markers. Flagging for whoever next touches this fragment;
not fixed here since it predates the audited diff.

No changelog.d directory exists in this repo's .claude/spec/ — CHANGELOG.md
generation is apparently unused for these internal lensing dev builds (only
COMPLETED.md/TODO.md via completed.d/todo.d are used). Don't go looking for
a changelog.d to write into.

render_fragments.py again left .claude/agent_state/foreman_lite.json dirty
as a side effect this run (in addition to the already-known
tidy_advisory.json) — revert both with `git checkout --` before finishing,
they are not real doc changes.
