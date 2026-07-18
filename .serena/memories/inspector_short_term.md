# Inspector Short-Term Observations

## 2026-07-18 — Build 3f RE-REVIEW (SACR-C) — VERDICT ISSUES (spec-only)

Second pass over the SAME uncommitted diff at HEAD 5e6bc3e (worktree
/home/tejaswi/Work/cogwheel-claude-dev). git status unchanged since the
first 3f review; diff byte-identical. Re-verified from scratch rather
than trusting memory.

### Correctness GREEN (re-confirmed this session)
- `pytest test_lensing_{channels,gauge,likelihood,fast_path}.py`:
  108 passed, 1 xfailed in 90 s. Imports resolve clean (ran the import
  probe on likelihood + channels + _gauge + the newly-imported privates
  `_channel_switch`/`_physical_kernels`/`reconstruct_from_envelope` and
  `_gauge` publics `switched_analytic_channels`/`channels_from_envelope`/
  `envelope_total`).
- SACR-C algebra re-derived by hand: telescoping is exact for ANY
  weights — unit carriers give carrier*conj(carrier)=1 (~1 ulp),
  sum alpha_j = 1; `envelope_total` recomputes carrier_c via the same
  `_unit_carrier` (F001 mod-2pi reduction) so cancellation is
  machine-precise. `channels_from_envelope`/`reconstruct_from_envelope`
  are the exact inverse; only E is interpolated.
- Caller/callee: only production caller `_get_dh_hh_no_asd_drift` does
  `delays,k0,k1,_ = _amplification_coefficients(...)` — discards the now
  seed-grid `partition`; tests use partition only for geometry/timing/
  falsification, and NearCusp falsification reads reconstructed k0, not
  `partition.kernels`. No external caller passes removed `n_kernel_nodes`.
  `_physical_kernels`/`_channel_switch`(4-arg)/`reconstruct_from_envelope`
  signatures all match their likelihood call sites.
- `_envelope_loo_nodes` termination + edge cases sound: endpoints
  (w_min, w_max) always retained so worst-cancellation w_max is always
  evaluated (CancellationError symmetry preserved); new_w always sorted
  so the size>=2 `searchsorted(grid,new_w)` keep-index is correct;
  empty-flanks and node-cap both break the loop; scale floor guards
  |F|->1. `exact_transition_channels` retained in `_gauge` and still
  tested (test-only now, not a NOW bug).
- DATA_CONTRACTS.yaml unaffected (ChangRefsdalPartition is an in-memory
  dataclass, not a serialized artifact — even with its 7 new fields).

### The ONE finding — INS-6-001 (== old INS-5-003) STILL OPEN
SPEC.md line 55 (Microlensed-PE row, "Fast path (Builds 3/3b)" sentence)
STILL describes the removed machinery: `_DEFAULT_KERNEL_NODES = 100`,
FULL-CLUSTER union, "smooth channel kernels K_a(w) engine-evaluated ...
then cubic-splined to bin sub-samples", F008 full-cluster rule. Code is
now SACR-C: single beat-free envelope E(w) on a LOO-adaptive coarse grid
(seed 8, stop 4e-3, ceiling 48), closed-form kernel reconstruction,
criticality-separation switch |tau_a - tau_c| (supersedes F008 where the
report certifies it). git confirms SPEC.md UNMODIFIED by the diff though
the plan listed it as expected-to-change. Spec->artifact invariant
broken; goal-#4 planned-work-with-no-diff. Inspector flags; Librarian
owns the sync (also wants an F008-superseded addendum in FINDINGS).
Severity: design.

### Carried forward
INS-6-001 open until Librarian syncs SPEC line 55 + FINDINGS F008 addendum.

### Pattern (reinforced)
A build that REMOVES module constants (`_DEFAULT_KERNEL_NODES`) and a
whole node-grid mechanism almost always leaves the SPEC row that named
those constants stale — check the exact paragraph, not just that the row
exists. Byte-identical re-review still worth re-running the suite +
import probe + hand-checking the telescoping identity.
