# Librarian Short-Term Observations

## 2026-08-10 — post-commit sync for d5da155 (ppGO rung gate calibration)

**Scope**: Commit d5da155 calibrated the provisional ppGO rung gate constants in
`cogwheel/lensing/chang_refsdal/_pearcey_cusp.py` (`_W_PPGO_FLOOR` 50.0→8.0,
`_R_PPGO_ERROR_CONST` 50.0→3.0) and added a new calibration script
`scripts/calibrate_ppgo_rung.py` (291 lines).

**Triage outcome**: Pure no-op. Reasoning:
- `_pearcey_cusp.py` changes are constant value + docstring updates only — no
  new or removed public symbols, no API changes, no serialization artifacts.
- `scripts/calibrate_ppgo_rung.py` is in `scripts/` (not `cogwheel/`); introduces
  no disk serialization artifacts (searched for save/load/savez/to_file/.npz/.json/
  open() — only match was a print() warning string).
- SPEC.md has no "provisional" language for these constants (confirmed by empty
  search). SPEC correctly carries mechanism/gate description, not the provisional
  values themselves.
- DATA_CONTRACTS.yaml: no stale entries for `_R_PPGO_ERROR_CONST` or `_W_PPGO_FLOOR`.
- `sync_derived_docs.py` ran cleanly — only the recurring test-consumer warning for
  `lens_amplification_surrogate` (escalation TODO `surrogate_contract_test_consumer_warning`
  already open; do NOT re-create).
- `depends_on: [2026-08-10_exterior_2d_fold_carrier]` in the open TODO fragment
  resolves correctly (completed fragment exists).
- Stale constant values (50.0/50.0) remain in the TODO fragment description, but
  updating TODO fragment context descriptions is outside the Librarian's enforcement
  rules (acceptance criteria are still unambiguous and the fragment is open).

**Pattern observed**: A build that exclusively calibrates constants (no new public
symbols, no new modules, no new disk artifacts) is always a Librarian no-op even if
it's a substantial code change — triage on API/module/serialization surface only.

**Finding from commit message (for awareness)**: ppGO still does NOT serve the
excised cusp-window region (R too small even with new gates; needs R>=71). The
`lensing_ppgo_rung_gate_calibration` TODO remains open with original acceptance
criteria intact.

**Still pending from librarian_knowledge** (carried forward, not in scope):
- FOLD-CARRIER SCHEMA CROSS-REF CLUSTER (INS-1-002/003): SPEC.md ~line 63 and
  DATA_CONTRACTS.yaml ~line 199 still describe `exterior_polar_rho_log_carrier_v1`
  as "ONLY known tag" with 1D rho_carrier — stale since V5 2D tag shipped.
- Lobe axis-schema rows in DATA_CONTRACTS.yaml still describe old V1/sqrt-edge
  tags (INS-4-002 / F050, deferred).
- surrogate_contract_test_consumer_warning escalation fragment still open.
- lensing_farfield_sd_coordinate_degenerates and
  lensing_farfield_name_spans_three_regimes fragments still open as
  measurement/deferral records.
