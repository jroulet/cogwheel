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
- When SPEC gains low-level perf/implementation detail (numba, grid
  sizes, timings) but overview.rst is pitched at architecture/API level
  with no per-eval perf claim, there is usually nothing to propagate —
  don't manufacture a performance blurb. While there, verify the
  FINDINGS IDs cited by SPEC exist and remain consistent.
- LAYERED capability claims: a doc sentence about the PUBLIC entry point
  (e.g. overview.rst "ChangRefsdalChannels is positive-parity only") can
  stay TRUE while a lower layer (geometry/operator) already supports the
  new regime — before editing, re-read the actual PUBLIC-layer refusal
  CODE (does channels.py still `raise LensDomainError`?), not SPEC's
  engine-row prose (which may describe the lower layer). Flip the sentence
  only once the public layer's raise is gone (channels saddle guard was
  lifted at Build 7b — the positive-parity-only claim is now due to flip).
- A todo fragment's itemized list can imply more distinct mechanisms than
  actually landed — ground-truth against `git log`/code before hunting for
  a claimed Nth separate change; N items can resolve as fewer mechanisms.
- If a symbol's own docstring already carries thorough rationale (e.g. a
  config dataclass docstring), treat SPEC.md's paragraph as a compressed
  echo of it — check the docstring first for staleness before re-deriving
  from the functions.
- If the Serena MCP connection drops mid-task, just retry — the project's
  own hooks block Bash/Read fallbacks on project files/commands, so there
  is no usable fallback anyway.
- Commit-preflight hooks can auto-stub `contracts_changelog.d/`/
  `spec_changelog.d/` fragments with placeholder text ("Auto-generated ...
  Librarian should refine this entry"); check the ACTUAL BODY of every
  changelog fragment touched in a diff for this stub marker before
  trusting render_fragments.py output — a fragment can exist and render
  cleanly while still being a content stub.
- When adding/repairing a DATA_CONTRACTS.yaml consumer list, grep for ALL
  direct callers of the artifact's accessor function across the codebase
  (not just the most obvious one) — consumer gaps often predate the
  current build.
- DATA_CONTRACTS.yaml consumer lists are production-only by convention;
  test-file-only callers (flagged by sync_derived_docs.py) should be left
  off and flagged for the artifact's own contract owner, not fixed
  opportunistically outside the current build's scope.
- On this machine, `scripts/sync_derived_docs.py` and
  `regenerate_consumer_graph.py` need `jedi` (present in conda env
  cogwheel-newlal, absent from the default PATH python) and ripgrep `rg`
  (absent entirely) — use
  `/home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python` for both.
  Without `rg`, `regenerate_consumer_graph.py` hard-fails; `sync_derived_
  docs.py` still runs against the stale cached CONSUMER_GRAPH.json (misses
  brand-new call sites only).
