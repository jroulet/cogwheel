# Inspector Short-Term Observations

## 2026-08-14 FINAL RE-REVIEW — Build tiling_census_node_budget (INS-1-001 CLOSED) — PASS

Scope: re-review of sole open finding INS-1-001 (design). Repo worktree =
/home/tejaswi/Work/cogwheel-claude-dev. Three census files still UNTRACKED
(new): cogwheel/lensing/tiling_census.py, scripts/tiling_census.py,
cogwheel/tests/test_lensing_tiling_census.py. Suite: 26 passed in 57s. Green.

### INS-1-001: RESOLVED (was: design, non-blocking; disclosure was test-only)
Prior pass kept it OPEN because the conservative-upper-bound caveat lived ONLY
in the test-file docstring + a structural pin (PpgoTrimIndependenceTestCase),
NOT on the shipped surface the JSON consumer / human module reader sees. Coder
has now added BOTH pieces the suggested fix asked for, on the shipped surface:
1. Module docstring: dedicated section "CONSERVATIVE UPPER BOUND (no ppGO trim
   modeled)" (tiling_census.py lines ~23-34) naming
   `surrogate_training._apply_ppgo_trim` / `get_certified_ppgo_map()`, stating
   counts are a conservative UPPER BOUND, never an underestimate, and that
   ppGO-served empties are not mis-flagged SILENT_EMPTY.
2. run() Returns docstring (lines ~736-742) repeats the caveat and points at
   the ppgo_trim_modeled key.
3. Real dict key `'ppgo_trim_modeled': False` (line 779) placed immediately
   after `aggregate_call_count` — machine-readable for the 7a cost consumer.
Out-of-scope constraint honored: NO `_apply_ppgo_trim`/`get_certified_ppgo_map`
call added to the counting loops (all grep hits are docstrings). Counting loops
still use `_census_region` -> record['n_nodes'] * _LABELS_PER_NODE.
Non-breaking: no test asserts an exact set(result.keys()), so the additive key
breaks nothing; PpgoTrimIndependenceTestCase intact (test file line 338).

### Everything else: re-confirmed correct across prior passes (unchanged)
Engine-free guarantee, thin-caller tiler fidelity, cost model mirroring
`_self_estimate`, band-verdict two-sided logic, Q1-Q4, schema tag
'tiling_census_v1'. No new findings.

### Carry-forward -> Librarian (doc staleness, NOT this build's code):
- Region vocabulary (lobe_exterior/lobe_interior/wedge_interior, and the
  census region names) absent from SPEC.md / DATA_CONTRACTS.yaml.
- exterior_polar_rho_log_carrier_v1 "ONLY known tag" stale since V5 2D carrier.

### Pattern reinforced
DISCLOSURE-IN-TEST != DISCLOSURE-ON-SHIPPED-SURFACE (from prior pass) — now
CLOSED the right way: the caveat a downstream JSON consumer or human module
reader needs must live in the MODULE docstring AND the output payload, not only
the test file. A structural test is a good regression guard but is not the
disclosure the consumer reads.
