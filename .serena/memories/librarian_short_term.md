# Librarian Short-Term Observations

## 2026-08-10 — Post-commit sync: exterior fold-carrier demodulation (b061103)

### Scope
Triggered by sync_issues.json listing SDK fix commit (`4c16ca4` — no-op, per prior session).
Actual doc work: build `exterior_fold_carrier_demodulation` (b061103) left untracked fragment
files and working-tree canonical file changes that needed committing.

### What changed
Build b061103 added:
- `_needs_fold_carrier` / `_compute_rho_carrier` to `surrogate_training.py` / `surrogate.py`
- `ExteriorPolarChart.rho_carrier` field — 1D `(n_rho,)` fold-carrier delays
- Schema bumped to `exterior_polar_rho_log_carrier_v1` (`_EXTERIOR_POLAR_AXIS_SCHEMA_V4`)
- Ghost-dominated tiles recovered (no longer dropped); ghost-transition zone (~40% of prior box)

### State at commit time
- Fragments untracked (never staged): completed.d, spec_changelog.d, contracts_changelog.d,
  todo.d entries
- Canonical files (SPEC.md, DATA_CONTRACTS.yaml, TODO.md, COMPLETED.md, SPEC_CHANGELOG.md,
  DATA_CONTRACTS_CHANGELOG.md) modified in working tree but not staged (except COMPLETED.md)
- render_fragments.py ran cleanly: "All surfaces up to date"
- overview.rst has no surrogate/exterior content — no Sphinx update needed (implementation detail)

### Committed
Staged and committed: all spec fragments + canonical files for fold-carrier build
(completed.d, spec_changelog.d, contracts_changelog.d, todo.d/lensing_exterior_2d_fold_carrier,
deletion of todo.d/lensing_exterior_fold_carrier_demodulation, SPEC.md, DATA_CONTRACTS.yaml,
SPEC_CHANGELOG.md, DATA_CONTRACTS_CHANGELOG.md, TODO.md, COMPLETED.md,
handoff/brief_exterior_2d_fold_carrier.md, librarian_short_term.md)

### Critical discovery: 2D carrier already in progress (uncommitted)
The working tree surrogate.py already has:
- `_EXTERIOR_POLAR_AXIS_SCHEMA_V5 = 'exterior_polar_rho_u_carrier_v2'` added to known schemas
- `_compute_rho_u_carrier` replacing `_compute_rho_carrier` — shape changed from `(n_rho,)` to
  `(n_rho, n_th)` (2D carrier on rho × theta_c grid)
- test files heavily modified

These are uncommitted — part of the lensing_exterior_2d_fold_carrier build in progress.
DATA_CONTRACTS.yaml currently describes COMMITTED state only (1D `rho_carrier`, V4 schema).
When 2D carrier build commits, DATA_CONTRACTS.yaml needs:
- rho_carrier shape updated from `(n_rho,)` to `(n_rho, n_u)` (or `(n_rho, n_theta_c)`)
- New V5 schema tag added (or V4 replaced)
- SPEC.md fold-carrier section updated to describe 2D mechanism

### BornResidualChart.load status sentence
SPEC.md says "(A BornResidualChart.load classmethod is not yet implemented; construct directly
from the shipped .npz arrays)." — VERIFIED STILL ACCURATE (no def load in _born.py).

### sync_derived_docs.py
Only recurring test-consumer warnings for `lens_amplification_surrogate`. The escalation
fragment `todo.d/surrogate_contract_test_consumer_warning.md` is still open (verified via
prior memory — do NOT create duplicate).

### Fragile cross-references
- SPEC.md + DATA_CONTRACTS.yaml cite `_EXTERIOR_POLAR_AXIS_SCHEMA_V4` and
  `exterior_polar_rho_log_carrier_v1` — when 2D build commits and V5 is the new known schema,
  BOTH surfaces need updating simultaneously
- SPEC.md FOLD-CARRIER section cites `_needs_fold_carrier`, `_compute_rho_carrier` — the
  uncommitted code RENAMES this to `_compute_rho_u_carrier`; when that commits, SPEC.md
  fold-carrier function names go stale
- DATA_CONTRACTS.yaml cites `rho_carrier` shape as `(n_rho,)` — will need update to `(n_rho, n_u)`
