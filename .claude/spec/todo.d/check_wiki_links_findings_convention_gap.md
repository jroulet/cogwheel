---
section: Backlog
---

- **`check_wiki_links` NEVER LEARNED THE double-bracketed "FINDINGS F0xx"
  CONVENTION — 5 PERMANENT FALSE DANGLES ON EVERY `render_fragments.py`
  RUN** `[housekeeping]` — the dangling-link checker in
  `scripts/render_fragments.py` only resolves double-bracketed wiki-link
  targets against `todo.d`/`completed.d` stems; it has never been taught
  that a double-bracketed "FINDINGS F0xx" reference is a valid
  cross-reference into `FINDINGS.md`'s own `## F0xx` headers. Five
  completed.d/todo.d fragments legitimately reference findings this way
  (`lensing_low_mass_exterior_training_registers_zero_charts.md` -> F070,
  `lensing_slow_tier_fixtures_left_their_served_domains.md` -> F069/F070,
  `2026-08-13_cusp_tie_guard_watches_the_wrong_side.md` -> F072,
  `2026-08-13_schwinger_certified_band_is_narrower_than_150.md` -> F071)
  and all five print as DANGLING on every render, permanently — this has
  now recurred across many Librarian post-commit-sync sessions with zero
  actionable fix each time (see `mem:librarian_knowledge`, "THIRD
  OCCURRENCE GETS A FRAGMENT"). Not a doc-content bug: every one of the
  five targets exists and is correct prose; only the checker's resolver
  is incomplete. Fix belongs in `scripts/render_fragments.py`'s
  `check_wiki_links` (or whatever renamed successor): also scan
  `FINDINGS.md` for finding-level headers as valid link targets before
  flagging a double-bracketed "FINDINGS F0xx" reference as dangling.
  Outside Librarian scope (`scripts/` tooling, not `cogwheel/` or doc
  prose) — filed here so future sync sessions can grep this fragment
  instead of re-noting the same warning in short-term memory a further
  time. (NOTE: this fragment intentionally avoids literal double-bracket
  syntax in its own body — using it here would add itself to the
  dangling-link count.)
