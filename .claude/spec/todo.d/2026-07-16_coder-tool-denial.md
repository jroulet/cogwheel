---
section: Backlog
---
- **Find what actually denies coder tool calls mid-build** `[housekeeping]` — an
  intermittent `"The user doesn't want to take this action right now. STOP what
  you are doing and wait for the user to tell you how to proceed."` comes back as
  a `tool_result` on a Coder's shell call during a build. A well-behaved Coder
  refuses to route around a denial, so the work package ends `BLOCKED` with zero
  files. It killed several Build-1b launches on 2026-07-16 and remains
  **unexplained**.
  **Ruled out, each by measurement, not argument:**
  * *The sandbox `/tmp` allowlist.* Detached (real build context), N=5 per arm,
    only `ignoreViolations {"file": ["/tmp/**", "/private/tmp/**"]}` varied:
    **0/5 denied with it, 0/5 without**. No effect. (I applied, reverted,
    re-applied and re-reverted this — the first "falsification" was a control
    error: the bisect/replay ran *with* the allowlist applied.)
  * *Positional ("the 2nd write is denied").* Four sequential `/tmp` heredoc
    writes in one session: 4/4 succeeded.
  * *Command content.* The byte-for-byte command denied at 09:11 replays clean
    3/3.
  * *Hooks.* `hook_trace.log` shows only instructive serena redirects, all
    retried successfully. Not a hook decision.
  * *User-scope deny rules.* `~/.claude/settings.json` has none.
  * *Coder memory.* The denial is a `tool_result` from the harness; nothing in
    the model's context can cause it, and the denied coder made zero memory
    calls.
  **Where the evidence points.** A minimal probe NEVER reproduces it (0/10)
  while real build coders hit it repeatedly (~4 in ~6 observations) — not
  chance. The trigger is something the probe lacks:
      probe:       task_context="", max_turns=6, 1 tool call before the write
      build coder: crew prompt + pre-read spec files + full WP text,
                   max_turns=90, 7-10 tool calls before the write
  Suspect **session depth or context size**, not `/tmp` and not the sandbox.
  Consistent with data first dismissed as noise: a write at call ~7 passed, at
  call ~8 denied, at call ~10 denied.
  **Next experiment** (`scratchpad/denial_rate.py` is the harness — extend it):
  hold the context DETACHED and vary ONE thing at a time — (a) a build-sized
  `task_context` and pre-read specs vs empty, (b) `max_turns` 6 vs 90, (c) N
  filler tool calls before the write. Compare denial RATES, N>=5 per arm. Never
  judge this by a single passing run; that error cost most of a day, twice.
  **Known-good workaround if it recurs and blocks progress:** in-workspace
  scratch has never been denied — a Coder writing `_scratch_probe.py` inside the
  repo via `mcp__serena__create_text_file` and running it works (observed
  2026-07-16 09:25, and every driver probe since). Only out-of-workspace shell
  writes have ever been refused. One line in `coder.md` would route around it,
  at the cost of leaving the real cause unknown.
  **Do not "fix" this by telling Coders not to probe.** A Coder probing to check
  a claim in its brief before building on it caught a real error in that brief
  (the prefactor phase cancellation was asymptotic, not exact). Probing is
  correct engineering when the math is subtle; the probe path needs to work.
