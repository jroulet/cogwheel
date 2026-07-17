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
