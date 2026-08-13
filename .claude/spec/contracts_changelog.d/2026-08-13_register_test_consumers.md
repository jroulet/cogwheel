---
bump: minor
---

Registered the test-only callers of `LensAmplificationSurrogate.load` (31
call sites, `test_lensing_surrogate.py`, `test_lensing_surrogate_lobe.py`,
`test_lensing_surrogate_training.py`, `test_lensing_farfield_envelope.py`)
and `CertifiedPpgoMap.load` (7 call sites, `test_lensing_saddle_rho_guards.py`,
`test_lensing_ppgo_bandsplit.py`) as consumer entries on `lens_amplification_surrogate`
and `certified_ppgo_map`, each tagged `kind: test`. No suppression flag
existed in `scripts/sync_derived_docs.py`'s `check_consumer_graph` (it only
matches on `module`/`function`, so `kind` is inert to the check but documents
the distinction for readers); adding a filter for `kind: test` there would be
a code change outside the Librarian's remit. These 38 callers were genuine,
recurring `consumer_graph` advisories on every commit (`todo.d/
surrogate_contract_test_consumer_warning.md`, open since 2026-08-0x across
4+ Librarian sessions for the 4-entry subset now folded into this larger
list) — registering them clears the noise and the fragment.
