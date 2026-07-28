# Librarian Short-Term Observations — 2026-07-28 (post-commit --post-commit 31ee133, 7 queued)

Scope: .claude/sync_issues.json listed 7 commits (079db9c, 2f7efe5, 2fc3f30,
5cc9429, 22fd569, f8a88d2, 31ee133) as pending — backlog had been bypassed
twice. Confirmed 079db9c/2f7efe5 were already synced by my own prior session
(overview.rst parity fix + lobe-serve changelog); 2fc3f30 (test suite off the
API surface, Sphinx warnings fixed) and 5cc9429/22fd569/f8a88d2 (FINDINGS
F023-F026, TODO restructure — spec-only, told not to rewrite) were self-
contained and needed no further action. The only commit with real downstream
doc-sync work was 31ee133 (Born carrier + band split + 'born' census + SDK
revision-loop fix).

What I fixed:
- SPEC.md: rewrote the "Born rung (DORMANT)" Conventions bullet — it claimed
  `b1` was "an unpinned placeholder" disagreeing with `operator.F_op` by
  ~13%, which is now FALSE (F023 derived closed-form `b1`/`a0`; the module's
  own docstring/WHY was fully rewritten in the same commit). New bullet:
  "Born rung (carrier machinery landed; serve slot still unwired)" — carrier
  + band-split + exterior fence described, wiring blocker now correctly
  named as the missing TRAIN_TIER residual chart, not a missing derivation.
  Also fixed the CENSUS row's "5-way MECE... (gamma-guard / cusp-window /
  refusal-ball / out-of-box / dropped-sliver)" -> 6-way with 'born' added,
  matching `_FALLTHROUGH_CATEGORIES`' actual tuple order. Bump: minor (fragment
  `spec_changelog.d/2026-07-28_born_carrier_bandsplit.md`), landed at
  rendered 0.22.0 (the alphabetically-later `saddle_lobe_serve` fragment from
  the PRIOR session claimed 0.23.0 — reconfirms the standing memory note:
  render_fragments.py assigns bumps by filename alphabetical order within
  spec_changelog.d/, not by content date; flag, don't fix).
- `todo.d/lensing_born_b1_derivation.md`: this fragment was STALE in a
  meaningful way — it itemized "the coefficients", "the ladder", "guard A
  re-derivation", "correct the docstring", and "the 'born' category ...
  absent from the tree" as OWED work, but 31ee133 landed every one of those
  except the final wiring-into-likelihood step. Left the item OPEN (did not
  delete/move to completed.d — the multi-part-program rule: it stays open
  until every listed part finishes, and wiring genuinely isn't done), but
  rewrote the body so it no longer asserts finished sub-items as pending.
  This is the same class of staleness as the "SPEC entries that cite a
  function by name go stale silently" note, just in TODO.d instead of
  SPEC.md — a fragment describing a plan goes stale when a LATER commit does
  most but not all of the plan and nobody edited the fragment.
- changelog.d: added 2026-07-28_born_carrier_bandsplit.md (prose synthesized
  from the commit message + module docstrings, following the standing
  precedent that every lensing-surrogate feature build gets one) and
  2026-07-28_sdk_revision_loop_fix.md (.claude/-only agent-infra fix;
  precedent for documenting these is `2026-07-27_sdk-agent-infra-hardening.md`,
  which also chose to record despite being excluded from main-sync).

What I verified and left alone:
- FINDINGS F023-F026 and F009 all exist with matching headers; not touched
  (driver said already written/rendered, and Inspector owns spec accuracy).
- DATA_CONTRACTS.yaml / docs/source: grepped for the fall-through category
  strings and "Born" — zero hits in either. The census categories are not
  enumerated on any Sphinx page, and no new disk artifact exists yet (the
  residual chart is TRAIN_TIER and unbuilt), so neither surface needed edits.
- api.rst: `:recursive:` autosummary over bare `cogwheel` still holds — new
  public functions in EXISTING modules (`_born.py`, `channels.py`,
  `surrogate_census.py`) need no manual entry.
- overview.rst: not touched, and confirmed NOT reverted (this session's
  driver context explicitly warned about this; the both-parities correction
  from 2f7efe5 is still in place).
- Old changelog.d/2026-07-27_born_rung.md (the original DORMANT entry) is
  now factually superseded by the new entry's content but was correctly left
  UNEDITED — CHANGELOG is append-only history, not a living doc.
- Test counts: SPEC's "27 tests" for `test_lensing_surrogate_census.py` is
  still accurate post-31ee133 (counted `def test_` — still 27; the 'born'
  category is exercised inside existing tests, no new test method added)
  — did not need updating despite the file's diff stat showing +/-25 lines.
- Sphinx rebuilt clean (`python -m sphinx -b html docs/source docs/build`,
  zero warnings) even though I touched no file under docs/source/ this pass
  — ran it anyway as due diligence given the backlog had been bypassed
  twice; confirms 2fc3f30's "fix every in-repo Sphinx warning" claim still
  holds.
- `sync_derived_docs.py` run 1: zero-diff auto-fix (no stray
  tidy_advisory.json this time, unlike prior sessions); flagged the same 4
  test-only `lens_amplification_surrogate` consumers as before — correctly
  left off DATA_CONTRACTS per the production-only consumer-list convention.

Mechanical/process note: the task prompt said "seven commits queued" and gave
driver context per-commit, but did NOT itself enumerate the commit hashes —
cross-checked against `.claude/sync_issues.json` (authoritative) rather than
inferring from `git log`, per the standing practice of trusting the trigger
file over any paraphrase.
