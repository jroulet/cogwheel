# Librarian Short-Term Observations

## 2026-08-10 — post-commit sync for d5da155 + 6d42675

**Scope**: Two commits: ppGO rung gate calibration build (d5da155) and saddle
exterior full treatment TODO + brief (6d42675).

**Triage — 6d42675**: Pure no-op. Added only agent-only paths (.claude/handoff/
brief + .claude/spec/todo.d/lensing_saddle_exterior_full_treatment.md +
generated TODO.md). No doc surface stale.

**Triage — d5da155**: Changed files:
- `.claude/agent_state/architect.json`, `.claude/agent_state/librarian.json`,
  `.serena/memories/architect_short_term.md`, `.serena/memories/
  professor_short_term.md` — agent-only, no-op
- `scripts/calibrate_ppgo_rung.py` — new scripts/ file; per SCRIPTS/ REWRITE
  NO-OP RULE: no new public API, no new disk artifacts — no-op for doc surfaces
- `cogwheel/lensing/chang_refsdal/_pearcey_cusp.py` — changed `_R_PPGO_ERROR_
  CONST` 50.0→3.0 and `_W_PPGO_FLOOR` 50.0→8.0 (constants measured, not
  provisional). SPEC.md already describes the mechanism mechanistically without
  provisional language — no SPEC update needed.

**Actions taken**:
- Closed ppGO rung gate calibration TODO:
  - Created `completed.d/2026-08-10_ppgo_rung_gate_calibration.md`
  - Deleted `todo.d/lensing_ppgo_rung_gate_calibration.md`
  - Ran render_fragments.py (COMPLETED.md and TODO.md updated)
- Reverted stray `.claude/tidy_advisory.json` diff (sync script side-effect)
- Did NOT commit `.serena/memories/professor_short_term.md` (pre-existing stray
  from build, not my change)

**Key finding recorded in completed.d**: ppGO does NOT certify in the immediate
excised cusp-window region (R too small there, R-gate requires R >= 71); the
excised cusp-window draws still fall to Pearcey. Gates measured and conservative
(not a code mistake). The original TODO's full acceptance (excised regions
served by ppGO) was not met for the immediate cusp-window; this is a physics
constraint, not a calibration failure.

**`[→ spec]` tag in closed TODO**: SPEC.md already mechanistically correct; no
spec_changelog.d fragment written.

**Still pending from librarian_knowledge** (carried forward, not in scope):
- FOLD-CARRIER SCHEMA CROSS-REF CLUSTER (INS-1-002/003): SPEC.md ~line 63 and
  DATA_CONTRACTS.yaml ~line 199 still describe exterior_polar_rho_log_carrier_v1
  as "ONLY known tag" with 1D rho_carrier — stale since V5 2D tag shipped.
  Inspector verified code correct; doc update still pending.
- lensing_farfield_sd_coordinate_degenerates and lensing_farfield_name_spans_
  three_regimes fragments still open as measurement/deferral records.
- surrogate_contract_test_consumer_warning escalation fragment still open (per
  memory: do NOT create a duplicate — it exists).
- Lobe axis-schema rows in DATA_CONTRACTS.yaml still describe old V1/sqrt-edge
  tags (INS-4-002 / F050, deferred).
