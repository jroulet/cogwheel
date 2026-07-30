Run the mechanical style pass, then launch the Tidier ONLY if judgment work
remains.

## 1. Mechanical pass first — deterministic, always

```
python scripts/tidy_mechanical.py
```

With no arguments it reads `.claude/tidy_advisory.json`, skips test files (out
of the Tidier's scope) and entries deleted since the advisory was written, and
applies the purely syntactic rules: whitespace-only lines, runs of 3+ blank
lines, trailing whitespace, final newline. Every edit is checked by an AST
round trip, so it cannot change semantics or produce a syntax error. It also
REPORTS lines over 79 columns without wrapping them.

Present its output. This is usually the whole job: measured 2026-07-30 across
a 32-file advisory, exactly ONE file needed a change and it took seconds.

**Do not launch the agent to redo this.** An agent doing the whitespace rubric
by hand took longer than a full build and was still unfinished, and one such
pass wrote the literal characters `\n` into `operator.py` where newlines
belonged, leaving the package un-importable (FINDINGS F047).

## 2. Launch the Tidier only for what a script cannot decide

Judgment work means: public API ordered before private helpers, imports
grouped in the right LAYER (not merely sorted), imports that are genuinely
unused (a name may be referenced only inside an `njit` body or a docstring
example), or module organisation that no longer matches what the module does.
Long lines the script reported are also judgment — where to break a line is a
readability call.

If none of that is in question, SKIP the agent and say so. "Mechanical pass
clean, nothing needing judgment" is a complete result.

When it is warranted:

1. Read the agent prompt from `.claude/crew/tidy.md`
2. Read the agent state from `.claude/agent_state/tidy.json`
3. Call the Agent tool with:
   - The full agent prompt as the main prompt
   - Append the state JSON under a `## Current State` header
   - Append user arguments under a `## Arguments` header: $ARGUMENTS
   - State plainly that the mechanical rubric is ALREADY APPLIED and must not
     be redone, and name the specific judgment questions you want answered
   - Scope it to a SHORT file list. The agent's cost scales with files read,
     and a 14-file list is what made the measured run outlast a build.
4. Present the agent's complete output without summarizing or filtering

## 3. MANDATORY in every case — even if you skipped the agent

```
python scripts/update_agent_state.py tidy
```

Skipping this leaves `tidy.json` reading `status: "failed"` from the PREVIOUS
run, so a pass that worked is indistinguishable from one that never ran. That
is exactly what happened between 2026-07-19 and 2026-07-28: the role produced
real work on 07-27 (a 153-line reflow of `cogwheel/lensing/**`), the state file
was never updated, and the advisory accumulated for eight days with nobody the
wiser. Run it even when nothing could be committed — a collision with a live
build is a normal outcome, and the state still records that the pass happened.
The post-commit hook prints a loud STALE banner when this is skipped.

Delete `.claude/tidy_advisory.json` when the pass is done.
