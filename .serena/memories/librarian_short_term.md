## 2026-07-30 post-commit sync — F041 arc-guard fix + salvaged 1b estimator retirement (commit 00bf8ae)

Scope: sync SPEC.md + COVERAGE_DESIGN.md to 00bf8ae. Diff for that commit:
`surrogate_training.py` (M, six numerical estimators retired -> analytic
geometry cascade), `test_lensing_caustic_cusps.py` (A, ~1182 lines, NEW),
`test_lensing_exterior_admission.py` / `test_lensing_surrogate.py` /
`test_lensing_surrogate_training.py` (M).

FIXED (NEW-MODULE gate — item 1, the only real staleness found):
- SPEC.md row 53 (microlensing engine): appended a "Certified by
  `test_lensing_caustic_cusps.py`" clause to the existing CERTIFIED BY
  sentence, before "The envelope LOO stop is gamma'-keyed" — covers the
  analytic `caustic_derivatives`/`caustic_speed`/`caustic_curvature_radius`/
  `fold_opening_direction` certification.
- SPEC.md row 55 (surrogate/training, TRAINING section): appended the same
  test module after "...chart being trained wrongly." before "FAR-FIELD
  TILING:" — covers analytic cusp-root, closed-form caustic-inradius,
  foot-of-normal curvature value, and fold-orientation-guard certification
  for the retired estimators (verified against the test file's own class
  list: CuspAnalyticRootTestCase, CausticInradiusClosedFormTestCase,
  FootOfNormalCurvatureValueTestCase, InteriorAdmissionMarginRemovalTestCase,
  InwardSignFoldHealthTestCase, SelfFalsificationTestCase, etc.).
- `spec_changelog.d/2026-07-29_caustic_cusps_test_coverage.md` (bump: patch)
  -> rendered `0.26.1`. `scripts/sync_derived_docs.py --check` no longer
  flags the module-list gate for this file (re-ran to confirm).

VERIFIED, NOT touched (task item 2 — SPEC row 55 "stale six-estimator"
claim did NOT hold under inspection, contrary to the driver's framing):
- Grepped row 55's full text for every retired-estimator name/wording
  (`_probe_arc_side`, `_PROBE_ETA`, `_CLOUD_MARGIN_FRAC`, `np.gradient`,
  "numerical", "differenti-", "margin" (only false positives:
  "caustic-margin census", "MARGINALIZED"), "inradius", "winding", "probe",
  "orientation") — zero matches. Row 55 already describes the FOOT-OF-NORMAL
  guard and cusp detection generically (`_min_curvature_radius` mentioned
  by name only, no implementation claim attached) and never named
  `_caustic_inradius`/interior-admission/arc-orientation machinery at all —
  so there was no false "numerical estimator" claim to correct. Confirmed
  the actual code (`_min_curvature_radius`, `_find_cusps`,
  `_branch_speed_profile`, `_caustic_inradius`) now calls
  `geometry.caustic_curvature_radius`/`caustic_speed`/analytic root — matches
  what little SPEC.md does say.
- COVERAGE_DESIGN.md (task item 3): only ONE relevant mention anywhere,
  section C6, "`_min_curvature_radius` already SKIPS a tube chart when
  `eta_max` exceeds half the local curvature radius" — a BEHAVIORAL claim
  (still UNCHECKED/circumstantial per its own STATUS line) that holds
  unchanged under the analytic implementation. No edit needed.
- FINDINGS.md F038-F042 (task item 4): all five headers exist
  (`## F0NN — ...`, double-hash), cross-references within them (F036, F039,
  F041) all resolve to real headers. No F0NN gaps introduced.
- `test_lensing_surrogate_training.py` (modified, not new, in 00bf8ae) and
  the pre-existing consumer_graph warning (`lens_amplification_surrogate`
  serialization-test consumers not in DATA_CONTRACTS.yaml) are OUT OF SCOPE:
  the training test file isn't in ANY SPEC.md certified-by list even before
  this commit (pre-existing gap, not introduced by 00bf8ae, not part of the
  NEW-MODULE gate since it's a modified not added file) and the
  consumer_graph warning predates my edits (present on the very first
  `sync_derived_docs.py --check` run, before any change) — flagged here,
  not fixed, per house convention of not expanding scope beyond the synced
  commit.

Mechanics: `mcp__serena__execute_shell_command` with a heredoc is a SILENT
NO-OP here too (rc 0, zero stdout) — same trap as the global memory notes,
now confirmed for `python3 - <<'PY' ... PY` specifically. Fix: write the
script to the scratchpad dir via `Write`, then `execute_shell_command
python3 <path>`. `render_fragments.py` again left the stray
`.claude/tidy_advisory.json` commit-hash/timestamp/touched_files diff —
reverted via `git checkout --`, not committed (same as every prior
session's note in `librarian_knowledge.md`).
