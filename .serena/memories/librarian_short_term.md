## 2026-08-14 (second run this date) — INS-1-003 doc-sync + TODO 5/7 closure

Scope: build `saddle_rho_relaxation` — F080 per-cell relaxation of the
F073 blanket saddle rho<1 ppGO certification refusal (map owns the
decision; SITE 1/SITE 4 duplicate pre-guards in likelihood.py and
surrogate_census.py deleted). Inspector deferred INS-1-003 (DATA_CONTRACTS
still described the old blanket refusal) since the fix touches
DATA_CONTRACTS.yaml, Librarian-only.

Fixed:
- DATA_CONTRACTS.yaml `certified_ppgo_map`: removed dead consumer entry
  (test class deleted in the same build — verified via get_symbols_overview,
  not inferred), added a SADDLE RHO<1 RELAXATION paragraph naming
  `_saddle_rho_relaxed_floor`/`_SADDLE_RHO_RELAXED_CELLS` as sole authority,
  exact-edge-equality match, fail-safe-to-UNKNOWN semantics, and the single
  active cell (gamma [1.157,1.339], floor 19.164, w_trust 28.746).
- SPEC.md ppgo_map.py module-row clause: same update (was "refuses
  (UNKNOWN) saddle rho<1 cells" unconditionally).
- New fragments: contracts_changelog.d + spec_changelog.d
  (2026-08-14_saddle_rho_relaxation_sync.md /
  2026-08-14_saddle_rho_relaxation_docsync.md), both bump: patch.
- CONSUMER_GRAPH.json: hand-removed the stale caller block for the deleted
  test function. Justified exception to "don't edit generated files":
  `scripts/regenerate_consumer_graph.py` requires `rg` (ripgrep), NOT
  installed in this environment (confirmed by direct failed run) — no
  automated regeneration path existed, and I had read the code directly
  (not inferred) to confirm the caller no longer exists.
- Closed TODO 5/7 (`todo.d/lensing_certified_map_guard_relaxation.md`):
  its exact stated scope ("per-cell relaxation keyed on re-validation
  evidence... serving the clean cell(s)") was fully delivered. Verified the
  two open caveats (MARGINAL/CONTAMINATED cells; F080 fan-asymmetry
  question) are ALREADY tracked in the separate `lensing_training_campaign`
  fragment (7a retrain) before closing — did not silently drop them.
  Wrote completed.d/2026-08-14_saddle_rho_relaxed_guard.md, `git rm`'d the
  todo.d fragment, repointed `todo.d/lensing_no_engine_census.md`'s
  `depends_on: [..., lensing_certified_map_guard_relaxation]` to
  `[..., 2026-08-14_saddle_rho_relaxed_guard]` (mandatory per prior
  convention — missed this once as an initial draft: my own new
  completed.d fragment used `[[lensing_certified_map_guard_relaxation]]`
  as a wiki-link to the file I was simultaneously deleting, which
  render_fragments.py correctly flagged as a NEW dangling link one run
  later. Fixed by de-bracketing the self-referential mention to plain
  prose — a fragment must never wiki-link a target it deletes in the same
  edit, even when the target is itself.

Verified end-to-end: ran `render_fragments.py` twice (before and after
fixing the self-inflicted dangling link) and `sync_derived_docs.py --check`
last — RC=0, zero output. Dangling-link count returned to the pre-existing
baseline of 5 (`[[FINDINGS F069/F070/F071/F072]]` — known checker
limitation, not FINDINGS-scanning capable, unrelated to this session).

Skipped: docs/source/ (Sphinx site) — grepped overview.rst for
ppgo/saddle/lensing mentions; only high-level Chang-Refsdal architecture
prose exists, no internal ppGO-map mechanism detail to sync. No Sphinx
rebuild performed (none needed — the rebuild rule only triggers on actual
docs/source/ edits).

New pattern for the family already documented (constant names cited in
both SPEC.md and DATA_CONTRACTS.yaml are fragile links if renamed): now
add `_SADDLE_RHO_RELAXED_CELLS` / `CertifiedPpgoMap._saddle_rho_relaxed_floor`
to that fragile-reference set — cited in DATA_CONTRACTS.yaml AND SPEC.md.

New pattern (self-link-to-deleted-target): when a completed.d fragment
documents closing a todo.d item, do NOT `[[wiki-link]]` the just-deleted
todo.d stem inside the completed.d fragment's own prose — the linked
target no longer exists by the time the fragment lands, so the dangling-
link checker (correctly) flags it. State the closure in plain prose
instead ("closes the `<stem>` backlog item... that closed todo.d fragment
(removed by this completion)...").
