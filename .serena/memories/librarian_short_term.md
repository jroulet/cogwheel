## Last session: 2026-08-06 post-commit sync (--post-commit 1749eed)

Scope: commits since last audit (c83c67f) through 1749eed — 6c6d6df, aac4d16,
034fcf7 (the real code change), 1749eed. Only 034fcf7 touched `cogwheel/`.

### What was stale
1. DATA_CONTRACTS.yaml `lens_amplification_surrogate` description: the
   FarFieldChart-record sentence said "(exterior far-field and interior
   SACR-C alike)" — true before 034fcf7, false after (034fcf7 retired the
   `ffin` path: FarFieldChart no longer ever carries INTERIOR_SACR_C, only
   InteriorWedgeChart/LobeInteriorChart do now). Fixed the parenthetical;
   contracts_changelog.d/2026-08-06_farfield-interior-retired.md (bump patch).
2. `todo.d/lensing_interior_wedge_chart_unwired.md` — the fragment 034fcf7
   completes — was left open by the build (driver flagged this explicitly).
   Deleted it, added completed.d/2026-08-06_interior-wedge-chart-wired.md.
   Tagged `[→ spec]` but SPEC.md never actually carried the "never trained"
   staleness claim in the first place (it already described InteriorWedgeChart
   generically without commenting on wiring status) — confirmed via targeted
   grep before concluding no SPEC.md edit was owed. Don't assume a `[→ spec]`
   tag means a required edit; verify the claim exists first.
3. `todo.d/lensing_farfield_name_spans_three_regimes.md` (added in aac4d16,
   the SAME audit that produced the fragment above) explicitly referenced
   `[[lensing_interior_wedge_chart_unwired]]` and said "DO NOT start this
   before the interior-wedge wiring lands" — both now stale since the
   fragment above is retired. Updated title "three regimes" -> "two regimes"
   (the interior tiles it complained about no longer exist under
   FarFieldChart), rewrote the backlink paragraph as past-tense history, and
   flipped the "DO NOT start" gate to "can now be started" (still deliberately
   DEFERRED per its own opening line, not blocked).

### New pattern confirmed
- A TODO audit that spawns TWO sibling fragments in the same commit
  (aac4d16: lensing_interior_wedge_chart_unwired + lensing_farfield_name_
  spans_three_regimes) can have one reference the other by name/backlink.
  When the referenced fragment completes and is deleted, grep BOTH for
  `[[fragment_name]]` backlinks AND for prose that assumes its still-open
  status ("DO NOT start until X lands") — the backlink check alone would
  have missed the "three regimes" title going stale, since the title itself
  doesn't contain the backlink syntax.
- DATA_CONTRACTS_CHANGELOG.md version numbers land out of chronological
  order exactly like SPEC_CHANGELOG.md does (already documented for SPEC in
  librarian_knowledge): my 2026-08-06 fragment rendered as `0.2.5` inserted
  ABOVE an existing `0.2.4` (2026-08-04) entry, while the top of the file
  is already at `0.4.1` (2026-08-01) from an unrelated earlier chain. Same
  quirk, same file family (render_fragments.py), not something to fix here
  — flagged, not touched.
- No docs/source/*.rst edits this session (no new module, no signature
  change, no toctree change) — confirmed via targeted grep for FarFieldChart/
  InteriorWedgeChart/astroid/ffin before concluding skip; Sphinx rebuild
  skipped accordingly (Step 4 only applies when docs/source/ changed).
- Consumer-graph drift warnings (4 test_lensing_surrogate.py Serialization*
  methods calling LensAmplificationSurrogate.load, flagged again by
  sync_derived_docs.py) reconfirmed as the standing test-only-caller
  convention — left off DATA_CONTRACTS.yaml consumer list, no fragment.
  This is the second session this exact warning set has fired unchanged;
  if a THIRD session sees it, worth asking the contract owner to silence it
  rather than re-triaging from scratch each time.

### Technique reminder
- SPEC.md/DATA_CONTRACTS.yaml table rows are giant single lines;
  `search_for_pattern` explodes in size when the keyword sits inside one of
  these (context lines re-duplicate the whole row per match). Route around
  it with a direct python read+`str.find`/slice via
  `mcp__serena__execute_shell_command` instead of tuning context params.
