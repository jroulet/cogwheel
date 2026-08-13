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
- SPEC.md backtick-in-bash trap (2026-08-08): a `MAX_SUBDIVISION_DEPTH`
  backtick inside a double-quoted bash `python -c "..."` was eaten by
  command substitution — repair via a heredoc temp script; verify by raw-
  bytes read (already known, bit me again).
- Constant names cited in SPEC.md and DATA_CONTRACTS.yaml become fragile
  cross-references: if `_LOBE_AXIS_SCHEMA*` or any schema constant is
  renamed in code, BOTH doc surfaces need updating simultaneously.
  (2026-08-08: `_EXTERIOR_POLAR_AXIS_SCHEMA` is a THIRD instance cited in
  both surfaces — same rule.)
- REQUIRED-vs-OPTIONAL SCHEMA CONTRACT VARIES PER CHART KIND (2026-08-08,
  exterior-polar cusp-adapted u build): the exterior-polar `theta_to_u`
  is OPTIONAL (parity==-1 charts are raw-theta, written conditionally,
  loaded as None) — NOT REQUIRED like wedge v3 / lobe v1. When syncing a
  cusp-adapted contract sentence, verify required-vs-optional per chart
  kind before writing it: a mechanical copy of the wedge/lobe "REQUIRED,
  read unconditionally, KeyError on absence" phrasing would be WRONG for
  exterior-polar. The retained sentence "No arc-length map is needed" is
  now paired with the cusp-adapted u map in both surfaces — a future build
  adding an arc-length or s-map to exterior-polar breaks that sentence.
- PENDING (2026-08-08, lobe cusp-adapted build 98c4e7f): the data-product
  contract still describes the OLD lobe axis schemas (raw-theta V1 and
  sqrt-edge); the production code now ships the SINGLE tag
  `lobe_caustic_relative_v1` (`theta_to_u` required, old tags hard-refuse).
  Deferred to Librarian as INS-4-002 / F050 — DATA_CONTRACTS.yaml needs the
  lobe axis-schema rows updated to the single v1 tag. See
  `mem:lobe_interior_chart` for the schema contract.
- ENUMERATED-KIND-LIST CROSS-REF (2026-08-08): a SPEC sentence that names
  the kinds that have a capability (e.g. "subdivided recursively where the
  kind has a subdivider (far-field, wedge, lobe)") becomes a fragile pair
  with the TODO item listing the missing kind ("TubeChart still has none") —
  if TubeChart ever gains a subdivider, BOTH the SPEC list and that TODO
  item need touching (same rename-preserved-staleness family as the polar
  re-chart case).
- `sync_derived_docs.py` reporting "some issues auto-fixed" with no actual
  git diff is likely an internal state flush (a no-op) — trust `git diff`
  as the source of truth, not the script's exit message.
- SPEC STATUS SENTENCES STALE SILENTLY: a sentence like "X is STILL NOT
  wired/done" goes stale the moment X gets wired/done in code, with no
  automatic doc update. On every doc sync pass, scan for status-
  description sentences (patterns: "not yet", "STILL NOT", "remains
  unwired", "currently disabled") and verify each against the actual code
  before deciding whether to flip or preserve it.
- SCRIPTS/ REWRITE NO-OP RULE: a complete rewrite of a `scripts/` file
  (even +100s of lines) is a legitimate librarian no-op when it stays
  within `scripts/`, introduces no new serialization artifacts, and makes
  no changes to the `cogwheel/` public API. Doc surfaces (SPEC, DATA_CONTRACTS,
  overview.rst) are only affected when scripts introduce disk-persisted
  formats or new public symbols.
- "DO NOT CLOSE BY LOWERING" ADVISORY UPDATE RULE: when a TODO fragment
  contains an explicit advisory such as "Do NOT close this by lowering
  `min_gamma_band`", and a build deliberately does that lowering, the
  advisory wording must be updated (not just the value). The TODO stays
  OPEN if other conditions remain (e.g. mass measurement, treatment
  decision). Update the advisory from "do not close by lowering" to reflect
  the new state, and note why the TODO remains open.
- TREE-GATE-STRANDING + `git add -A` MISLABELING VARIANT: when a commit
  message describes only the intended file(s) but the actual commit
  (via `git add -A`) swept in unrelated stray diffs left by a tree-gated
  tool run, `git show --stat` will show extra paths the message never
  mentions — treat this as a distinct recurring variant of the "verify
  the caller's framing" check (see the `git show --stat` bullet above),
  not a one-off; flag/split it rather than accepting the commit message
  at face value.
- PLAIN-TEXT FRAGMENT-NAME REFERENCES ARE INVISIBLE TO THE DANGLING-LINK
  CHECKER: `render_fragments.py`'s dangling-link check only inspects
  `[[fragment_name]]`-bracketed backlinks; a fragment that references
  another fragment's name in plain prose (no brackets) is silently
  unchecked — such a reference can go stale (renamed/retired target)
  with zero tooling signal. When auditing cross-references for coherence,
  grep for bare fragment-name substrings too, not just `[[...]]` syntax.
- ESCALATE (not just re-note) a repeatedly-unfixed warning: the
  `lens_amplification_surrogate` test-only-caller consumer-graph warning
  from `sync_derived_docs.py` has now recurred identically across FOUR+
  librarian sessions with zero diff/fix each time. Re-noting it a fifth
  time is no longer proportionate — the next librarian session that sees
  this exact warning should escalate to the contract owner directly
  (e.g. via a dedicated TODO fragment) rather than adding a fifth
  passive mention to this memory. STATUS 2026-08-08: the escalation fragment
  `todo.d/surrogate_contract_test_consumer_warning.md` now EXISTS (still
  open) — do NOT create a duplicate; re-verify it is open and move on.
- SPEC STATUS SENTENCE STALE SILENTLY — RENAME-PRESERVED VARIANT (polar
  re-chart, 337ac15): a doc sync that RENAMES a symbol (FarFieldChart →
  ExteriorPolarChart) can preserve the stale sentence BESIDE the new name
  verbatim — 337ac15 updated the class name but kept the old parity
  restriction ("exterior positive-parity only") in BOTH SPEC.md and
  DATA_CONTRACTS.yaml even though the code charts both parities. When a
  sync commit edits a NAME, re-read the REST of that sentence against the
  code, not just the renamed token.
- SPEC_CHANGELOG EMPTY-DATE BUCKET: a patch-bump spec fragment WITHOUT a
  `date:` field renders into the empty-date bucket (e.g. a new fragment
  rendering at 0.11.7 while the top version stays 0.34.0) — same family
  as the alphabetical-ordering quirk; harmless, don't "fix".
- SUPERSEDED FRAGMENT STAYS OPEN AS MEASUREMENT RECORD: a completed.d
  fragment that says it "supersedes the open direction in [[X]]" does
  NOT close X — X remains open as the measurement record until its OWN
  acceptance criteria are met (lensing_farfield_sd_coordinate_degenerates
  stays open pending the sd-coordinate measurement; the rename-deferral
  fragment lensing_farfield_name_spans_three_regimes stays open too).
  Do not mark a superseded fragment DONE on the strength of its
  superseder's closure.
- BULK-TRAINING ACCEPTANCE ITEMS ARE DRIVER POST-BUILD, NOT IN-BUILD
  (2026-08-08): when a completed TODO fragment's stated acceptance is a
  bulk-training sweep (e.g. "4x4x4 probe ~70 charts not 500", "cusp-
  vertex tile clears eps bar"), it is never measured in-build (AGENTS.md);
  close the TODO but record the training-scale items as driver post-build
  verification in the completed.d fragment. Don't let a closed fragment
  imply in-build proof of a training-scale number.
- `depends_on` REPOINTING IS MANDATORY ON COMPLETION (2026-08-09): a
  dependent open fragment's `depends_on: [<old todo stem>]` dangles the
  moment the fragment moves to `completed.d` under a date-prefixed name —
  the renderer's validator warns. Repoint `depends_on` to the NEW
  date-prefixed completed stem (convention confirmed against
  2026-08-07_polar_rechart, 2026-08-07_subdivision-recursion-wedge-v3-r-
  caustic). I missed this at first; the dangling-dep warning caught it.
- `delete_lines` EMPTIES A FILE BUT DOES NOT DELETE IT (2026-08-09):
  `rm` the now-empty todo fragment afterwards.
- SPEC CARRIES MECHANISM, NOT PROVISIONAL VALUES (2026-08-09): when a
  completed fragment records a constant as PROVISIONAL (post-build driver
  measurement owed), write the SPEC sentence for the MECHANISM/gate only —
  a later tightening of the constant then keeps the SPEC valid and only
  the completed record ages. New fragile cross-ref family: SPEC naming
  `_R_PPGO_ERROR_CONST`/`_W_PPGO_FLOOR`/`_PPGO_BAR_DIVISOR` + the phrase
  "returns before any table or quadrature lookup" breaks if a future build
  moves the rung after the table consult.
- FOLD-CARRIER SCHEMA CROSS-REF CLUSTER STILL STALE (2026-08-10,
  INS-1-002/003, carried from the 2D fold-carrier build): SPEC.md ~line 63
  and DATA_CONTRACTS.yaml ~line 199 still describe
  `exterior_polar_rho_log_carrier_v1` as "the ONLY known tag" with a 1D
  `(n_rho,)` rho_carrier — stale since V5 `exterior_polar_rho_u_carrier_v2`
  (2D `(n_rho, n_theta_c)`) shipped and the old tag became known-but-not-
  written. Both surfaces cite `_EXTERIOR_POLAR_AXIS_SCHEMA_V4` + `_V5`,
  both literal tags, and the "Old 1-D rho_carrier artifacts load by
  broadcasting to 2-D" sentence — if V4 or the broadcast is ever dropped,
  all of these go stale together. SPEC.md also cites `_compute_rho_u_carrier`.
  Inspector verified the code correct (OVERRIDE -> doc-sync); the update is
  still pending.
- POST-COMMIT SYNC NO-OP RULE (2026-08-10, commit 992c500): a post-commit
  sync triggered by `.claude/sync_issues.json` whose changed file is
  TEST-ONLY (or a notebook) is a NO-OP — skip entirely, no doc surface
  stale, no sync script needed. Don't manufacture work for test-only
  commits.
- GATE-CRITERION CHANGE TO A SPEC-DESCRIBED RUNG IS STALENESS even when
  the code comment/constant docstring explains the new condition
  (2026-08-11, ppGO resolution-gate build; spec bumped 0.37.7 -> 0.37.8
  patch): SPEC.md enumerates firing conditions, so adding a gate condition
  (fold-pair-existence OR w*delta_min >= _PPGO_RESOLUTION_GATE=4.0)
  requires the SPEC sentence update. `_PPGO_RESOLUTION_GATE` is now a
  FOURTH constant in the `_PPGO_BAR_DIVISOR`/`_R_PPGO_ERROR_CONST`/
  `_W_PPGO_FLOOR` fragile cluster — and it MIRRORS operator.RHO_END, so if
  RHO_END changes the mirror note breaks. Verified the new gate computes
  `geometry.delay` per image (no table consult), so the SPEC phrase
  "returns before any table or quadrature lookup" stays TRUE.
- RETIRING A MULTI-ITEM PROGRAM FRAGMENT (2026-08-11): mark the last
  STILL RED section RESOLVED inline (matching the file's existing inline
  RESOLVED pattern), add `date:` to frontmatter, then `mv` to
  `completed.d/<date>_<slug>.md`. Its prose name appears in FINDINGS.md and
  other completed fragments — plain-text refs to a retired todo path must
  be swept manually (the dangling-link checker only sees `[[...]]`).
- `replace_content` literal-mode no-match (2026-08-11): a NEWLINE sat
  between two words of the needle ("sitting behind\nthis one") — when a
  literal needle reports no match, verify the raw bytes for line wraps
  before assuming a unicode issue.
- DON'T DOCUMENT A PROVABLY-UNREACHABLE CENSUS CATEGORY (2026-08-13): a
  new served-category label can be code-reachable in isolation yet
  provably never populate in production because an upstream dispatch
  branch (e.g. a coarser rho>1 routing rule) always intercepts first —
  documenting it in SPEC's category breakdown would misrepresent the
  breakdown as richer than it behaves. Say the disjointness/ordering fact
  instead (which IS true and durable); add the category sentence only
  once a later build makes the routing order-independent or the category
  is observed to actually fire.
