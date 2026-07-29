---
section: Backlog
---

- **Consolidate duplicate routing pins in the lensing test suite
  [housekeeping].** Measured 2026-07-29: one branch decision is pinned in many
  files at once, so a single-condition change reds the whole set.

  | decision pinned | test methods | files |
  |---|---|---|
  | `SchwingerCertificationError` identity | 32 | 10 |
  | `select_branch` routing | 16 | 6 |
  | `W_CEILING_SCHWINGER` boundary | 11 | 6 |

  Demonstrated cost: the authoritative-gate build changed two branch
  conditions and re-pointed EIGHT test files over three revision rounds,
  roughly two-thirds of its wall clock. None of those tests had caught either
  defect they nominally guarded (F028, F029) — they asserted which code path
  ran, never what the answer was.

  TARGET: one canonical pin per decision (`test_thresholds_have_one_home` is
  the model — lift it from constants to the predicate), with every other site
  either asserting a VALUE against an oracle or deleted. Do NOT weaken an
  assertion to make it pass, and do not delete one that still encodes a true
  claim.

  METHOD WARNING: an AST classifier was tried and is NOT sufficient to drive
  deletions — `assertEqual` (499 uses) is semantically ambiguous between
  numbers, branch labels, shapes and counts, leaving 58% of methods
  unclassifiable. Any consolidation must be read file by file. Because this
  deletes coverage, it wants a scoped build with an explicit
  no-silent-weakening acceptance, not an ad-hoc sweep.
