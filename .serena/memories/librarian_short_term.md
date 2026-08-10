# Librarian Short-Term Observations

## 2026-08-10 — post-commit sync for bc156f6 (ppGO rung gate TODO + brief)

**Scope**: Commit bc156f6 added a build brief and a TODO fragment for the ppGO rung gate calibration build. No cogwheel source code, no SPEC.md, no Sphinx docs changed.

**Triage outcome**: Pure no-op. Changed files were:
- `.claude/handoff/brief_ppgo_rung_gate_calibration.md` — agent-only path
- `.claude/spec/TODO.md` — generated canonical
- `.claude/spec/todo.d/lensing_ppgo_rung_gate_calibration.md` — new TODO fragment only

**No doc surface stale.** No sync script run needed.

**Still pending from librarian_knowledge** (carried forward, not in scope this pass):
- FOLD-CARRIER SCHEMA CROSS-REF CLUSTER (INS-1-002/003): SPEC.md ~line 63 and DATA_CONTRACTS.yaml ~line 199 describe exterior_polar_rho_log_carrier_v1 as "ONLY known tag" with 1D rho_carrier — stale since V5 2D tag shipped. Inspector verified code correct; doc update still pending (needs a future dedicated pass).
- lensing_farfield_sd_coordinate_degenerates and lensing_farfield_name_spans_three_regimes fragments still open as measurement/deferral records.
- surrogate_contract_test_consumer_warning escalation fragment still open.
- Lobe axis-schema rows in DATA_CONTRACTS.yaml still describe old V1/sqrt-edge tags; production code ships single `lobe_caustic_relative_v1` tag (INS-4-002 / F050, deferred).
