---
section: Backlog
---

- **A DRIVER DECISION AND AN OWNER DECISION ARE INDISTINGUISHABLE IN THE BUILD
  LOG** `[housekeeping]` — `gates.py` writes `User: accepted remaining
  findings — proceeding past the Inspector gate` for every file-based
  escalation decision, regardless of who created the decision file. The driver
  resolves routine escalations under standing authorization; the owner
  resolves the rare ones. Both render as `User:`.
  MEASURED (2026-08-07, `subdivision_recursion`): the driver accepted
  INS-1-001 — a real finding, shipping a stale `DATA_CONTRACTS.yaml` until the
  driver fixed it post-build — and the build record attributes that call to
  the owner. On a build that shipped a schema change, "who approved this" is
  exactly the question a log is for.
  Same defect class the SDK already fixed once for commit attribution: build
  commits were hardcoded to a model tier no role is ever assigned, writing a
  false authorization trail into git history (`orchestrator.py` now derives
  the trailer from `AGENT_MODELS`). This is the same false trail in the build
  log instead of the commit.
  FIX: the decision file's author is knowable — the driver writes
  `escalation_accept` itself, whereas an owner decision arrives through the
  interactive path. Label them distinctly (`Driver:` / `Owner:`), or record
  the provenance in the file and echo it. Cheap; the value is that an audit
  can tell delegated calls from human ones.
