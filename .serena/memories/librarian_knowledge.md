# Librarian Long-Term Knowledge

- Before trusting the caller's framing of what changed, verify
  independently: `git show --stat --name-only <hash>` per commit and
  `git diff --name-only <first>~1 <last>` should match the claimed
  changed-files union. Catches drift between task description and
  reality.
- Plain `git diff`/`git show` work directly via Bash even when project
  convention nudges toward Serena-only tools — the "USE SERENA"
  redirect targets non-exempted shell commands, not git itself.
- `search_for_pattern`'s `paths_include_glob` takes exactly ONE glob —
  a comma-separated list silently matches nothing.
- If `docs/source/api.rst` uses `:recursive:` autosummary over the bare
  package name, new subpackages need no manual entry — verify this
  still holds before adding one by hand.
- A TODO fragment framing work as a multi-part program stays open until
  every listed part finishes.
- Record no-op sync runs as a commit rather than skipping silently —
  preserves the audit trail.
- Don't touch other agents' concurrent in-flight uncommitted changes or
  memory files outside the explicit commit range you were scoped to.
- `SPEC_CHANGELOG.md` version numbers can read out of order:
  `render_fragments.py` assigns bumps by alphabetical filename order
  within `spec_changelog.d/`, not by fragment date or build sequence —
  a known rendering quirk; flag, don't "fix".
- When SPEC gains low-level perf/implementation detail (numba, grid
  sizes, timings) but overview.rst is pitched at architecture/API level
  with no per-eval perf claim, there is usually nothing to propagate —
  don't manufacture a performance blurb. While there, verify the
  FINDINGS IDs cited by SPEC exist and remain consistent.
