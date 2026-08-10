# Librarian Short-Term Observations

## 2026-08-10 — Post-commit sync: exterior_2d_fold_carrier (commit 572eaa4)

### Scope
Post-commit mode triggered by `.claude/sync_issues.json` for commit 572eaa46
(feat(lensing): 2D (rho, u) fold-carrier on ExteriorPolarChart).

### Outcome: NO-OP — all surfaces already current
This was a clean no-op sync. The feature commit 572eaa4 was a "full-Librarian" commit
that included all doc sync work inline:
- SPEC.md updated (V4/V5 tags, `_compute_rho_u_carrier`, 2D carrier description)
- DATA_CONTRACTS.yaml updated (both known tags + 2D `rho_u_carrier` + broadcast note)
- All spec/contracts/changelog fragments written and rendered
- `todo.d/lensing_exterior_2d_fold_carrier.md` deleted, `completed.d` fragment added
- Previous post-commit sync (2dd11b9) covered the fold-carrier demodulation build

### sync_derived_docs.py
Ran clean. Only the known recurring `lens_amplification_surrogate` test-consumer
warnings — tracked by open TODO fragment
`todo.d/surrogate_contract_test_consumer_warning.md`. Do NOT create a duplicate.

### Sphinx docs
- `docs/source/api.rst` uses `:recursive:` autosummary — lensing auto-discovered, no
  manual entry needed (confirmed again)
- `docs/source/overview.rst` references lensing at high level (ChangRefsdalChannels,
  LensedWaveformGenerator, LensedRelativeBinningLikelihood) — no surrogate internals,
  no staleness from 2D carrier build
- `BornResidualChart.load` "not yet implemented" sentence in SPEC.md is still
  accurate — confirmed no `def load` in `cogwheel/lensing/born_residual_chart.py`

### Pre-existing stray diff
`tidy_advisory.json` was already modified (M) at session start — hash updated from
c00899ec to 572eaa4 by post-commit hook. NOT caused by this session, not staged.

### Fragile cross-references (still active from previous sync)
- Both SPEC.md and DATA_CONTRACTS.yaml cite `_EXTERIOR_POLAR_AXIS_SCHEMA_V4`,
  `_EXTERIOR_POLAR_AXIS_SCHEMA_V5`, and the two literal tags
  (`exterior_polar_rho_log_carrier_v1`, `exterior_polar_rho_u_carrier_v2`)
- SPEC.md line ~56 cites `_compute_rho_u_carrier` — rename touches both surfaces
- "Old 1-D rho_carrier artifacts load by broadcasting to 2-D" sentence paired with
  V4-retained claim — if V4 is ever dropped or broadcast removed, all three sentences
  go stale together
