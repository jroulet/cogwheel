---
section: Backlog
---

- **CODER TURN BUDGETS DO NOT SCALE WITH THE WORK** `[housekeeping]` — measured
  2026-07-30 across three builds. `_test_dev_budget` scales with spec count
  (`min(60 + 20*n, 250)`), but a Coder WP's `max_turns` is a free-form Architect
  estimate that does not scale with the WP's declared `where` file count. The
  mechanism exists on one side of the DAG and not the other.

  Logged tool calls vs budget, and where the agent made its FIRST write:

  | agent | budget | tool calls | 1st edit at | outcome |
  |---|---|---|---|---|
  | F054 `coder-2` (1 file) | 75 | 12 | call 8 | ok $3.96 |
  | 1e-ff `coder-4` | 105 | 110 | call 8 (8%) | EXHAUSTED |
  | 1e-ff-port `coder-2` (4 files) | 95 | 156 | call 14 (15%) | EXHAUSTED |
  | 1e-ff `test_dev-6` (11 specs) | 80 | 145 | call **61 (76%)** | EXHAUSTED |
  | 1e-ff `test_dev-7` | 80 | 131 | call 62 (78%) | EXHAUSTED |

  TWO distinct failure modes, and they need different fixes:

  1. **Orientation starvation** (`test_dev-6/7`): 76-78% of the budget spent
     reading before the first write. Root cause was F057 (11 specs counted as
     1); the sharding fix addresses it, but the fixed orientation COST is still
     absent from `60 + 20*n` — the constant 60 was not measured against this
     codebase, where `surrogate.py` alone is 4000+ lines.
  2. **Honest overrun** (`coder-4`, `coder-2`): both started editing inside 15%
     and simply ran out doing real work. A one-file WP got 75 turns; a
     four-file, ~60-site port got 95. A budget that barely moves while the work
     grows fivefold is not a budget.

  FIX: give the Coder the same treatment as the Test Developer — floor
  `max_turns` at a function of `len(wp.where)` (and, if cheap, the total line
  count of those files), keeping the Architect's estimate only when it is
  HIGHER. Log the arming value, per F058.

  ACCEPTANCE: replay the three builds' WPs through the new formula and show it
  would have budgeted above the observed call counts; a single-file WP is
  unchanged.
