# Librarian Short-Term Observations

## 2026-08-17 (post-commit sync, 228debf..69c79b8, beat-free tube representation)

- MISPLACED-DIRECTORY BUG FOUND AND FIXED: `changelog.d/` (the surface
  `render_fragments.py` actually reads for CHANGELOG.md, per
  `SURFACES["changelog"]["frag_dir"] = "changelog.d"`) lives at the REPO
  ROOT, sibling to `cogwheel/`/`scripts/` — NOT under `.claude/spec/`
  alongside `todo.d`/`completed.d`/`spec_changelog.d`/`contracts_changelog.d`.
  Two prior agent commits (a258f08 "find_cusps_wrap_fix",
  5990a8d "census audit") both wrote their changelog fragments to
  `.claude/spec/changelog.d/` — a directory `render_fragments.py` never
  reads — so both fragments sat committed but silently un-rendered;
  CHANGELOG.md was stuck at 2026-08-13 while changelog.d fragments existed
  through 2026-08-17. `git mv`'d both back to the correct top-level
  `changelog.d/`, added the missing beat-free-tube fragment there too, and
  re-ran `render_fragments.py` — CHANGELOG.md picked up all three in one
  render. THE WRONG PATH IS AN EASY MISTAKE: every other fragment surface
  DOES live under `.claude/spec/`, so `changelog.d` is the one exception.
  Check `git ls-files changelog.d | wc -l` vs
  `git ls-files .claude/spec/changelog.d` if CHANGELOG.md ever looks stale
  despite "all surfaces up to date" — the script only says that about the
  surfaces it actually scans; a fragment in the wrong directory is
  invisible to it, not flagged.
- The beat-free tube representation build (69c79b8) is a genuine
  CHANGELOG-worthy breaking change (stale-artifact hard-refusal via
  `envelope_definition`/`TUBE_BEAT_FREE_AIRY` tag) — SPEC.md/DATA_
  CONTRACTS.yaml/completed.d already carried it (verified: SPEC.md
  0.47.0/2026-08-17, F083 resolves in FINDINGS.md), only the user-facing
  CHANGELOG.md entry was missing. Added `changelog.d/2026-08-17_tube_
  beat_free_representation.md` citing commit `69c79b8`, F083 measured
  numbers (n_theta 48->10, eps=4.2652e-03 vs 0.0237 bar), and told
  readers to regenerate cached tube-chart `.npz` artifacts.
- Reconfirmed the known `render_fragments.py` side effect: running it
  touched `.claude/tidy_advisory.json` (commit-hash/timestamp bookkeeping
  bump) with zero real content change — reverted with `git checkout --`
  before committing, per standing memory rule.
- Reconfirmed the 5 permanent dangling `[[FINDINGS F0xx]]` wiki-link
  warnings are the already-filed, already-known checker gap
  (`todo.d/check_wiki_links_findings_convention_gap.md` exists and is
  open) — not new, no action.
- Rest of the 11-commit post-commit backlog (228debf, 64c6434, 35ba7bd,
  21a1abd, 5fd58b6, 49ba729, 7a23e3f, 8e48294, c59b6d6, 202d412) is
  SDK/agent-state/session-state bookkeeping under `.claude/sdk/`,
  `.claude/agent_state/`, `.claude/handoff/`, or spec fragments already
  self-consistently rendered in the same commit — genuine no-ops, skipped
  per the agent-only/notebook-test-only triage rules.
