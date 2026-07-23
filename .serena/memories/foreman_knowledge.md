# Foreman-Lite Long-Term Knowledge

- When fixing a narrowly-scoped finding (e.g. a stale docstring
  reference), touch only the exact spots the finding names — don't
  rewrite surrounding narrative that may need a broader rebase, even if
  it now reads slightly inconsistent. Flag the deeper contradiction for
  whoever owns that broader work instead of scope-creeping into it.
- A bare "user doesn't want to take this action" shell-tool denial with
  no reason given is often transient — retry once, then fall back to
  read-only verification (e.g. `read_file`) rather than repeatedly
  retrying.
- A finding whose text says "Librarian-owned" / "-> Librarian:" (doc-sync,
  SPEC row) is NOT Foreman-Lite work — Foreman-Lite must not write SPEC.md.
  Decline immediately; do not re-verify the same no-op every pass (it
  recurred 7x on INS-5-DOC-1). Escalate the mis-route as an orchestrator
  routing bug rather than burning an agent turn repeating the decline.
- `rename_symbol` updates live code references but not docstring mentions
  of the old name written as `module._old_name` text — grep and fix those
  separately.
