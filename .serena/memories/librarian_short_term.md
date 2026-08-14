# Librarian Short-Term Observations

- 2026-08-14 post-commit sync (5 commits, 3747b47..399cbb0: calibration-
  script import fix, F079 wrap fix + cusp-arm coverage retirement, F080
  finding + 5/7 re-scope, tube_d2_fold build). Step 0
  (`sync_derived_docs.py`) ran clean (5 checks OK) but left a stray diff in
  `.claude/tidy_advisory.json` (bumped its `commit`/`timestamp` fields and
  `touched_files` to reflect HEAD) — same stray-diff family as the
  previously-documented `render_fragments.py` side effect, just triggered
  by `sync_derived_docs.py` this time. Reverted with `git checkout --`,
  did not commit; add this as a second known trigger for that pattern.
- Only ONE real staleness found across the whole diff: `.claude/spec/
  todo.d/lensing_training_campaign.md` item (b) named
  `measure_saddle_cusp_arm_coverage.py` as a live "or" option alongside
  "the retirement path ... decide from the same evidence" — but the very
  build two commits later (`find_cusps_wrap_fix`, F079) DELETED that
  script and retired `_SADDLE_CUSP_ARM_COVERAGE` as measured-INERT,
  answering the "decide from the same evidence" clause the fragment had
  left open. Rewrote (b) to say DONE-via-retirement instead of presenting
  a dead script as a still-open choice. New pattern to watch: an open TODO
  fragment that poses an "either X or Y, decide later" choice can go
  stale the moment a LATER commit in the same backlog actually decides it
  — even though the fragment's prose was accurate when written, re-check
  "decide later" clauses against the FINDINGS entry they point to before
  trusting the fragment's own wording.
- Everything else naming the four deleted scripts (measure_cusp_arm_reach,
  measure_cusp_arm_actual_boundary, measure_saddle_cusp_arm_coverage,
  calibrate_ppgo_rung) is a legitimate HISTORICAL record and was left
  alone: completed.d/2026-08-04_cusp-arm-coverage.md,
  completed.d/2026-08-10_ppgo_rung_gate_calibration.md,
  completed.d/2026-08-10_saddle_exterior_full_treatment.md, FINDINGS.md
  (F0xx body prose), COMPLETED.md, CHANGELOG.md, changelog.d/2026-08-04_
  cusp-arm-coverage.md, and the build brief
  `.claude/handoff/find_cusps_wrap_fix.md` — all describe what WAS done
  with a script that existed at the time. `lensing_coverage_map.md` row 3
  ("DONE 2026-08-04...measured by ... measure_cusp_arm_actual_boundary.py")
  is the same historical-record pattern inside an otherwise-open backlog
  fragment — don't confuse a DONE sub-item's historical script citation
  with a live reference.
- retired_concepts.json already had both new F079 entries
  (`_CUSP_ARM_COVERAGE`, `_SADDLE_CUSP_ARM_COVERAGE`) written by the build
  itself — verified, no action needed.
- DATA_CONTRACTS.yaml, docs/source/*, `data_registry.yaml`: zero diff in
  the 5-commit range and zero references to cusp-arm coverage or the
  deleted scripts — confirmed by grep, not just assumed from the brief.
  SPEC.md tube-fold row left untouched per explicit instruction
  (driver-written, already verified).
- The calibration-script-import-fragility pattern flagged in
  `librarian_knowledge.md` (2026-08-14 entry) was the FIRST of these 5
  commits (0eb9ea0) fixing exactly that fragility — confirms the flagged
  risk was real; no further action, already resolved upstream.
