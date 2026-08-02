# Librarian Long-Term Knowledge

- Before trusting the caller's framing of what changed, verify
  independently: `git show --stat --name-only <hash>` per commit and
  `git diff --name-only <first>~1 <last>` should match the claimed
  changed-files union. A Dreamer/Inspector-flagged "unconsumed" SPEC gap
  may already have been HAND-FIXED in the feature commit itself — check
  before editing; verify-only is the correct outcome, don't re-touch.
- Plain `git diff`/`git show` work directly via Bash even when project
  convention nudges toward Serena-only tools — the "USE SERENA"
  redirect targets non-exempted shell commands, not git itself.
- `search_for_pattern`'s `paths_include_glob` takes exactly ONE glob —
  a comma-separated list silently matches nothing. It also returns the
  ENTIRE table row as one string for SPEC.md (rows are single un-wrapped
  lines) — a multi-alternation regex with several hits on the same line
  duplicates the same full-line context per match; harmless, don't be
  surprised by "2 identical results" for 1 line.
- If `docs/source/api.rst` uses `:recursive:` autosummary over the bare
  package name, new subpackages need no manual entry — verify this
  still holds before adding one by hand (reconfirmed across many builds).
- A TODO fragment framing work as a multi-part program stays open until
  every listed part finishes.
- Record no-op sync runs as a commit rather than skipping silently —
  preserves the audit trail.
- Don't touch other agents' concurrent in-flight uncommitted changes or
  memory files outside the explicit commit range you were scoped to.
- `SPEC_CHANGELOG.md` version numbers can read out of order:
  `render_fragments.py` assigns bumps by alphabetical filename order
  within `spec_changelog.d/`, not by fragment date or build sequence —
  a known rendering quirk; flag, don't "fix". Within a SINGLE date,
  alphabetical filename order determines which fragment gets the lower
  version suffix (e.g. `2026-08-01_c6_...` gets .1 and `2026-08-01_lobe_...`
  gets .2 even if lobe landed first chronologically).
- SPEC ENTRIES THAT CITE A FUNCTION BY NAME GO STALE SILENTLY: a LATER
  commit in the SAME backlog can refactor that function's role (e.g.
  `_frame_t_min` demoted to a thin accessor once `_frame_delays` became
  the authoritative construction) while touching no doc files. Check the
  CURRENT docstring of every function SPEC.md names against what SPEC.md
  claims about it — citing it once does not keep it accurate three
  commits later.
- WHEN A COMMIT DECLARES A MODULE "the single authoritative statement"
  about a convention (e.g. `d_luminosity` is PHYSICAL, not apparent),
  grep SPEC.md for the older contradicting sentence yourself: such
  commits routinely omit SPEC.md from changed_files, so the flip never
  propagates. Check FINDINGS.md too, but leave it alone if the finding
  is about different physics.
- If SPEC.md never described a mechanism's specific criterion in the
  first place, a build that CHANGES that criterion creates no staleness —
  don't manufacture a sentence that wasn't there. Implementation-level
  detail (gate formulas, constants) belongs in the module docstring;
  SPEC.md carries conventions and architecture. A SPEC "public-name list"
  row is often a representative group label, not exhaustive (e.g. omits
  established names like critical_point/macro_matrix/r_caustic too) —
  add a newly-shipped public name for discoverability, but don't treat a
  pre-existing omission of an older name as staleness.
- When SPEC gains low-level perf/implementation detail but overview.rst
  is pitched at architecture/API level, there is usually nothing to
  propagate — don't manufacture a performance blurb. While there, verify
  the FINDINGS IDs cited by SPEC exist and remain consistent.
- LAYERED capability claims: a doc sentence about the PUBLIC entry point
  can stay TRUE while a lower layer already supports the new regime —
  re-read the actual PUBLIC-layer refusal CODE before editing, not SPEC's
  engine-row prose. Flip the sentence only once the public raise is gone.
- A todo fragment's itemized list can imply more distinct mechanisms than
  actually landed — ground-truth against `git log`/code before hunting for
  a claimed Nth separate change.
- If a symbol's own docstring already carries thorough rationale, treat
  SPEC.md's paragraph as a compressed echo of it — check the docstring
  first for staleness before re-deriving from the functions.
- If the Serena MCP connection drops mid-task, just retry — the project's
  own hooks block Bash/Read fallbacks on project files/commands.
- Commit-preflight hooks can auto-stub `contracts_changelog.d/`/
  `spec_changelog.d/` fragments with placeholder text ("... Librarian
  should refine this entry"); check the ACTUAL BODY of every changelog
  fragment touched in a diff for this stub marker.
- DATA_CONTRACTS.yaml is for DISK artifacts: an in-memory dataclass that
  gained new fields needs no entry unless it round-trips through disk —
  confirm via its docstring before conflating "gained fields" with
  "needs a contract". When adding/repairing a consumer list, grep for ALL
  direct callers of the accessor (gaps often predate the build); consumer
  lists are production-only by convention — test-file-only callers flagged
  by sync_derived_docs.py stay off and get flagged to the contract owner.
- On this machine, `scripts/sync_derived_docs.py` and
  `regenerate_consumer_graph.py` need `jedi` (present in conda env
  cogwheel-newlal, absent from the default PATH python) and ripgrep `rg`
  (absent entirely) — use
  `/home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python` for both.
  Without `rg`, `regenerate_consumer_graph.py` hard-fails; `sync_derived_
  docs.py` still runs against the stale cached CONSUMER_GRAPH.json.
- Running `scripts/render_fragments.py` can leave a stray unrelated diff
  in BOTH `.claude/tidy_advisory.json` AND
  `.claude/agent_state/foreman_lite.json` as a side effect — revert both
  with `git checkout --` and don't commit them; "All surfaces up to date"
  with zero real diff is a legitimate clean-backlog outcome, not a missed
  check.
- When a changelog/TODO fragment reports "N new tests", SUM the literal
  per-report test counts from each contributing agent's own change report
  rather than eyeballing a round total — an un-summed guess is an easy
  off-by-a-few arithmetic slip caught only by re-adding the numbers.
- Post-commit doc-sync is triggered by an untracked `.claude/sync_
  issues.json` file listing pending commits back through a given hash;
  after completing the sync (committing only the doc files you changed),
  delete the trigger file.
- todo.d fragments can cross-reference a SAME step between an "ordering"
  fragment (numbered by build letter) and an "inventory" fragment
  (numbered by target, not build letter) via `[[fragment_name]]`
  backlinks — marking a step done means adding a DONE marker in BOTH,
  not just the one that names the build letter; grep for backlinks
  before considering doc-sync of a step complete.
- Before marking a numbered todo/inventory item DONE from a grep hit
  alone, read any "carried forward"/caveat subsection in that fragment —
  a target symbol can still exist because it was deliberately repurposed
  for a different later step, not because the original step is unfinished.
- No `changelog.d` directory exists in this repo's `.claude/spec/` —
  these internal lensing dev builds use only `completed.d`/`todo.d`
  (COMPLETED.md/TODO.md), not CHANGELOG.md generation.
- SPEC.md replacements involving `\|` pipe-escape characters must be done
  via Python's `str.replace` called through the shell — Serena regex mode
  double-escapes backslashes on return, causing "No matches" or corrupt
  text joins. Always verify SPEC.md edits by checking raw bytes (Python
  snippet) rather than Serena's `read_file` view (which un-escapes
  backslashes in the display).
- Constant names cited in SPEC.md and DATA_CONTRACTS.yaml become fragile
  cross-references: if `_LOBE_AXIS_SCHEMA*` or any schema constant is
  renamed in code, BOTH doc surfaces need updating simultaneously.
- `sync_derived_docs.py` reporting "some issues auto-fixed" with no actual
  git diff is likely an internal state flush (a no-op) — trust `git diff`
  as the source of truth, not the script's exit message.
- SPEC STATUS SENTENCES STALE SILENTLY: a sentence like "X is STILL NOT
  wired/done" goes stale the moment X gets wired/done in code, with no
  automatic doc update. On every doc sync pass, scan for status-
  description sentences (patterns: "not yet", "STILL NOT", "remains
  unwired", "currently disabled") and verify each against the actual code
  before deciding whether to flip or preserve it.
