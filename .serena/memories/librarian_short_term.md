# Librarian Short-Term Observations

## 2026-08-21 (low-w shell chart doc sync, INS-1-007 / INS-2-003)

- KILLED-MODULE SPEC STALENESS AT THE WHOLE-ARTIFACT SCALE: a build that DELETES
  a chart module + trainer + test file (LowWDiffractiveChart) and replaces the
  artifact leaves SPEC.md's LOW-W DIFFRACTIVE RUNGS paragraph AND
  DATA_CONTRACTS.yaml's `low_w_diffractive_chart` entry stale on EVERY clause —
  representation (quotient r_new with fold_cusp_reference/_NON_VANISHING_MIN_RATIO
  vs macro-lead demodulated-DIFFERENCE R = f_pure - born_lead_carrier), axis
  (w^(2/3)/w23_grid vs log w/log_w_grid), margins (derate/declined_mask vs none),
  coverage (union shell-OR-wall vs shell box only), schema (v1/v2 vs
  low_w_shell_v1), producer/consumer module names, and the census SERVE_ROUTES
  label. Same family as 'SPEC cites a function by name', at whole-artifact scale.
- BORN FLOOR GATE STALENESS: the `rho > 2` text in SPEC.md appeared THREE times
  (row FIRST-CLASS BORN INTERCEPT gate, row BORN EXTERIOR RUNG, and the Born
  rung bullet); a floor change to `_BORN_RHO_FLOOR = 1.4` requires sweeping all
  three plus DATA_CONTRACTS `> 2.0`/`rho <= 2`/grid-node clauses. Also removed a
  stale "(A BornResidualChart.load classmethod is not yet implemented...)"
  parenthetical — the classmethod long shipped (likelihood.py:1184 calls it).
- THE GAUGE-DISJOINTNESS FACT IS THE DURABLE SPEC SENTENCE: the Born floor
  1.4 (scalar-reach ppgo_map.caustic_rho) and the shell RHO_HI 1.4 (directional
  _caustic_rho) are DIFFERENT physical surfaces — scalar <= directional always,
  so a theta-dependent coverage GAP exists between shell and Born at rho ~ 1.4.
  This is a real adjudication (INS-3-001) that the build settled; record the
  disjointness fact, don't claim 'no gap no step'. (Inspector still lists a
  trivial prose-fix finding in test_lensing_low_w_shell_chart.py claiming
  'no gap/no overlap' — that's a TEST file, read-only for the librarian; the
  code-touching role must fix the docstrings.)
- TRAINER RUNTIME CLAIM: the old DATA_CONTRACTS born entry said "Training
  runtime approx 11 s"; the re-trained 7x8x13 grid's runtime is NOT stated
  anywhere in scripts/train_born_residual.py — don't invent a number, drop the
  runtime sentence and keep "trained offline by <script> (<grid>)".
- NEW FRAGILE CROSS-REF CLUSTER (from the shell-chart build): SPEC.md +
  DATA_CONTRACTS.yaml now cite `LowWShellChart`/`low_w_shell_chart` +
  `_SCHEMA = 'low_w_shell_v1'` + RHO_LO = _DIFFRACTIVE_FIT_FENCE_RHO_LO /
  RHO_HI = 1.0 + _DIFFRACTIVE_FIT_FENCE_DELTA (single-sourced from _diffractive)
  + the `w_shell = 1/delta_min` band-split. If the shell rho band or the
  fence-constant pairing ever moves, BOTH surfaces go stale together.
- CENSUS-ROUTE-LIST ENUMERATION STALENESS (recurring, now the rename variant):
  the SERVE_ROUTES 12-member enumeration in SPEC.md must be swept on ANY
  add/remove/RENAME — this build renamed `low_w_diffractive_chart` ->
  `low_w_shell_chart` (route count unchanged at 12), requiring the tuple list
  AND the "split by parity into <route> / diffractive_analytic /
  diffractive_engine_hosted" waterfall sentence AND the "exactly one of TWELVE
  MECE serve routes" count check.
- SPEC.md table-pipe caution reconfirmed: the LOW-W paragraph replacement text
  must escape `|` as `\|` INSIDE the table row (I wrote `measured |r| ~
  0.61-1.6` unescaped first, breaking the column count from 9 to 11; reverted to
  `\|r\|`). Verified table column integrity by comparing unescaped-pipe counts
  (git show HEAD vs worktree).
- RENDER SIDE-EFFECT: render_fragments.py dirtied .claude/tidy_advisory.json
  (pre-existing build diff, NOT librarian-caused — confirmed it was already
  modified in the build's own diff before I ran anything). Left as-is.
