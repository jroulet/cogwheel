Print the crew roster. No tools needed — just output this:

## cogwheel — Agent Crew

| Agent | Model | Mode | Role |
|-------|-------|------|------|
| **Architect** | Opus (Sonnet for triage) | Plan (read-only) | Lead planner, stats test specs, DESIGN finding triage |
| **Simplifier** | Sonnet | Plan (read-only) | Complexity auditor — lean/watch/trim |
| **Coder** | Opus | Execute | Implements work packages |
| **Tidier** | Sonnet | Execute | Structural style cleanup |
| **Test Developer** | Sonnet/Opus | Execute | Unit + domain tests |
| **Inspector** | Opus | Execute (read-only) | Code review + verification |
| **Librarian** | Sonnet | Execute | Documentation sync |
| **Dreamer** | Sonnet | Execute | Memory consolidation |
| **Foreman-Lite** | Sonnet | Execute | Trivial changes (fast path) |

### Pipeline
```
Phase 1: Architect + Simplifier → Plan (JSON)
Phase 2: Coder → [Tidier ∥ Test Dev] → Inspector → Librarian
Phase 3: Dreamer
Fast path: Foreman-Lite (skips all coordination)
```

### Memory
Each agent (except Simplifier) has short-term + long-term memories.
The Dreamer consolidates short-term into long-term during Phase 3.
Cross-reads enable learning: Coder reads `inspector_knowledge` to write
cleaner code; Inspector reads `coder_knowledge` to avoid false positives.

### Invoke
- `/build "task"` — SDK pipeline (autonomous, with plan review)
- `/build --fast "task"` — Skip planning, Foreman-Lite handles directly
- `/inspect` — Inspector review
- `/doc-sync` — Librarian audit
- `/dream` — Memory consolidation
- `/tidy` — Style cleanup
