# Librarian Short-Term Observations — 2026-07-28

Scope: post-commit backlog clear, 20 pending commits (d84a2d5..66a0100), driver
flagged 66a0100 ("frame-invariant far-field label, cusp nodes, carrier guard
remetric") for two specific checks. Outcome: nothing stale, no doc edits made.

What I verified and why each was a non-issue:
- `channels.reconstruct_farfield` gained a REQUIRED positional `t_min`
  (docstring at channels.py:1089 spells out the frame-invariance rationale and
  the hard-fail-if-stale design). Grepped every call site repo-wide
  (`reconstruct_farfield\(` excluding channels.py itself): all are in
  `cogwheel/lensing/likelihood.py` and `cogwheel/tests/test_lensing_*.py`,
  already passing `t_min`. Zero hits in `docs/source/`, `.claude/spec/`, or
  any docstring example outside the function's own module. No doc surface
  ever called it, so the required-arg change created no doc staleness.
- F022 (far-field carrier guard measures `arg` not re/im) and two new TODO
  fragments (`lensing_frame_invariant_roundtrip_precision.md` tagged
  `[→ spec]` but describing an UNRESOLVED precision tradeoff — not something
  to propagate yet since no decision/code change has landed; and
  `tests_reachable_red_on_symptoms.md`, `[housekeeping]`) — both render
  cleanly. `python scripts/render_fragments.py` reported "All surfaces up to
  date" with zero diff (one stray unrelated change to
  `.claude/tidy_advisory.json` appeared as a side effect of running the
  script — not mine, reverted with `git checkout --`, do not commit it).
  All F0xx cross-references across SPEC.md/TODO.md/COMPLETED.md resolve
  against FINDINGS.md headers (F001-F022 all present, F015 appears
  out-of-chronological-order in the file but exists).
- `ppgo_map.annulus_rho` (new helper, WP1/D2 extraction): not referenced in
  any doc surface; correctly not added — it's a function inside an existing
  module (`ppgo_map.py`), not a new top-level module, so enforcement rule 6
  (API coverage) doesn't apply. `:recursive:` autosummary over bare
  `cogwheel` in `docs/source/api.rst` still covers it automatically —
  reconfirmed again this run.
- Ghost gate re-key (decay -> saddle separation, 87643d7) and the new
  far-field carrier guard (F022, 66a0100): grepped SPEC.md for "ghost gate",
  "saddle separation", "carrier guard", "cusp node" — zero hits. SPEC.md
  never described either mechanism's specific criterion, only the
  higher-level delay-frame convention (lines 99-114) and Born-rung status
  (lines 89-98), both still accurate. Per standing rule: a criterion SPEC
  never stated in the first place creates no staleness when it changes.
- No new disk serialization in the 66a0100 diff (`surrogate_training.py`,
  `surrogate.py`, `surrogate_census.py`, `ppgo_map.py` all checked for
  `np.savez`/`.save(`/`to_file`/`.npz`/`json.dump`/`pickle.dump` — none
  added), so DATA_CONTRACTS.yaml needs no new entry.
- Earlier commits in the 20-commit backlog (delay-frame authority, distance
  convention, Born-rung-dormant) were already synced by the prior Librarian
  commit c3dbc29 (itself sitting inside this same pending-commit list —
  the hook flags a librarian's own doc commit too, that's expected, not a
  sign of an incomplete prior run).

Surprise: none really — this was a clean backlog, confirming the "check
before touching" discipline pays off (would have been easy to manufacture an
api.rst or overview.rst edit for `annulus_rho` that wasn't warranted).

No memory update needed to `librarian_knowledge.md` — every pattern applied
here was already recorded from prior runs.
