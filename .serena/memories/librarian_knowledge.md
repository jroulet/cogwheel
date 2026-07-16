# Librarian Long-Term Knowledge

- Before trusting the caller's framing of what changed, verify
  independently: `git show --stat --name-only <hash>` for each commit
  should reproduce the given changed-files list, and
  `git diff --name-only <first>~1 <last>` should match the union across
  all commits in scope. Catches drift between the task description and
  reality.
- Plain `git diff`/`git show` work directly via Bash even when project
  convention nudges toward Serena-only tools for source files — the
  "USE SERENA" redirect targets non-exempted shell commands (e.g. python
  scripts), not git itself.
- `search_for_pattern`'s `paths_include_glob` takes exactly ONE glob — a
  comma-separated list silently matches nothing. Use a single glob, or
  omit it and filter via `relative_path` / exclude globs instead.
- If `docs/source/api.rst` uses `:recursive:` autosummary over the bare
  package name, new subpackages need no manual entry — verify this still
  holds before adding one by hand.
- A TODO fragment that frames work as a multi-part program should not be
  marked complete when only one part lands — leave it open until every
  listed part finishes.
- Record no-op sync runs as a commit rather than skipping silently —
  preserves the audit trail even when nothing needed changing.
- Don't touch other agents' concurrent in-flight uncommitted changes or
  memory files outside the explicit commit range you were scoped to sync.
