# Librarian Short-Term Observations

## 2026-08-10 -- post-commit sync for a25f140 (NO-OP)

Scope: saddle build tidier pass (style: consolidate mid-file import block in
test_lensing_surrogate_training.py).

All three changed files are test-only:
- cogwheel/tests/test_lensing_farfield_envelope.py
- cogwheel/tests/test_lensing_surrogate.py
- cogwheel/tests/test_lensing_surrogate_training.py

No cogwheel/ module changes, no public API changes, no serialization, no new
disk artifacts. POST-COMMIT SYNC NO-OP RULE applies (established 2026-08-10,
commit 992c500). Zero doc surfaces stale. sync_issues.json deleted.

## Previous session carry-forwards (still pending):

- FOLD-CARRIER SCHEMA CROSS-REF CLUSTER (INS-1-002/003): SPEC.md ~line 63
  and DATA_CONTRACTS.yaml ~line 199 still describe
  exterior_polar_rho_log_carrier_v1 as "the ONLY known tag" -- stale since
  V5 2D tag shipped. Both surfaces need updating. Still pending.
- Lobe axis-schema DATA_CONTRACTS.yaml rows (INS-4-002/F050) deferred.
- lensing_farfield_sd_coordinate_degenerates + name_spans_three_regimes open.
- surrogate_contract_test_consumer_warning escalation fragment open; no dup.
- Concurrent Tidier uncommitted changes (surrogate.py M, tidy_advisory.json M)
  from 2026-08-10 session still present in working tree; NOT committed by
  Librarian (out of scope).
