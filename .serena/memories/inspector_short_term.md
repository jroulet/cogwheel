# Inspector Short-Term Observations

## 2026-07-20 — Build 8c-cont RE-REVIEW #3 (registration + census + INS-3-001 fix)

Scope: uncommitted tree, worktree /home/tejaswi/Work/cogwheel-claude-dev.
`cd` into main tree hook-blocked; use `git -C <worktree>` + Serena shell
with cwd set. Full python: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.

### Prior-finding dispositions
- **INS-3-001 (dropped-sliver bucket always-empty) → RESOLVED.**
  surrogate_training.py now: (a) `_build_provenance(box, config, charts,
  dropped_gamma_slivers=())` stores `provenance['dropped_gamma_slivers'] =
  [list(s) for s in dropped_gamma_slivers]` (line 950); (b) train() collects
  `all_dropped_slivers` across BOTH parities as flat `[float(lo),float(hi)]`
  pairs (line ~1011) and the single call site passes it:
  `_build_provenance(box, config, charts, all_dropped_slivers)` (line 1033).
  Census read side `_dropped_slivers_from` defaults to
  `surrogate.provenance.get('dropped_gamma_slivers')`; `_normalize_slivers`
  handles None/empty/[[lo,hi],...] correctly. Report `'parities':
  parity_reports` each carrying `dropped_gamma_slivers` → the override helper
  `dropped_slivers_from_training_report` reads `report['parities'][label]
  ['dropped_gamma_slivers']` consistently. Shape fix matches the prior
  suggestion (FLAT, not parity-keyed). Import probe green.
- **INS-1-001 (census test suite) → STILL OPEN.** No
  `test_lensing_surrogate_census.py` in cogwheel/tests (dir listing +
  grep for `surrogate_census`/`census.` in test_lensing_surrogate.py both
  empty). surrogate_census.py (767 lines) + CLI ship with ZERO dedicated
  tests. The ten Domain Test Descriptions, the two design falsifiables
  (tube >=3x raw at eps_95; fold-approach flat vs raw ~-1/2 slope) and the
  F010 mutation test remain absent. Route to Test Developer (independent
  authorship, NOT WP-CS coder). NOTE: test_lensing_surrogate.py DOES round-
  trip the `dropped_gamma_slivers` provenance field (lines 1358/1668), so
  the serialization half of TEST 12 is covered; the census-tool behavior
  is not.

### WP-REG (registration) — COMPLETE, verified
- DATA_CONTRACTS schema 0.1.0→0.2.0 (correct MINOR). New artifact
  `lens_amplification_surrogate` producer=scripts/train_lens_surrogate.py::main,
  consumers = LensedRelativeBinningLikelihood + LensedMarginalizedExtrinsic
  Likelihood + surrogate_census.run. Both likelihood ctors confirmed accept
  `amplification_surrogate` (marginalized_likelihood.py:97,175,242 incl.
  get_init_dict pop-when-None + NotImplementedError-when-set). data_registry
  package_data root + entry added. regenerate_consumer_graph LOADERS entry;
  CONSUMER_GRAPH.json regenerated (generated_at 2026-07-20).
- `pipeline_graph.py list` → lens_amplification_surrogate consumers=8
  registry_path=yes; consumers_of resolves 2 declared likelihoods + census
  run [declared+actual] + 4 test callers + _load_or_build [ACTUAL-undeclared].
  `_load_or_build` (training resume path) undeclared is benign (producer-side).

### No new findings this pass
The continuation only added: surrogate_training.py provenance threading
(additive, correct), registration files, test_lensing_surrogate.py growth.
No regression to surrogate.py/channels.py/likelihood.py crown byte-identity
introduced by this delta.

### Verdict
ISSUES — solely because INS-1-001 (census test suite) is undelivered.
resolved_ids: INS-3-001.

### Carry-forward
Census tube cusp-window fall-through, refusal-ball, both design
falsifiables, F010 mutation remain runtime-unexercised until the missing
census test suite lands (INS-1-001).
