# Foreman-Lite Short-Term Observations

- INS-8-001 and INS-8-002 (this pass, 2026-07-28): NINTH distinct
  finding-ID batch (INS-1..8 pairs, plus INS-5-DOC-1 x7) showing the
  identical mis-route — both findings explicitly say "Flag to Librarian"
  / "Inspector does not edit canonical surfaces" in their own text, yet
  land in the Foreman-Lite queue again. Re-verified read-only via
  search_for_pattern: SPEC.md line 55 still says "5-way MECE" fall-through
  breakdown (code's _FALLTHROUGH_CATEGORIES is six-way, 'born' category
  present) and the "Born rung (DORMANT)" paragraph (lines 88-91) still
  cites the low-w far-zone / sqrt(mu_macro)-expansion premise superseded
  per F025. Declined both edits, touched nothing — SPEC.md is
  Librarian-owned per the ownership split and my hard requirement not to
  write SPEC.md. The orchestrator routing bug is now confirmed persistent
  across 9 separate passes — strongly recommend filtering "-> Librarian"
  tagged findings out of the Foreman-Lite queue at the source
  (orchestrator level) rather than relying on each pass to catch and
  decline it individually.