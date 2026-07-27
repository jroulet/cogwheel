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
  a comma-separated list silently matches nothing.
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
  a known rendering quirk; flag, don't "fix".
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
  SPEC.md carries conventions and architecture.
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
