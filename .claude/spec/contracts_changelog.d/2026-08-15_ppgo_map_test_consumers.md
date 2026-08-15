---
bump: minor
---

Registered four more test-only callers of `CertifiedPpgoMap.load` on
`certified_ppgo_map` (`cogwheel/tests/test_lensing_ppgo_map.py`:
`BandScopedRelaxationTestCase.setUpClass`,
`CensusLikelihoodBandSplitMirrorTestCase.setUpClass`,
`RelaxedCellSelfFalsificationTestCase.setUpClass`,
`ShippedMapSaddleRelaxedCellTestCase.setUpClass`), each tagged `kind: test`
per the 2026-08-13 convention (`contracts_changelog.d/
2026-08-13_register_test_consumers.md`). These were flagged by the
pre-commit consumer-graph advisory during the `certified_map_guard_
relaxation` build; folding them in clears the noise for future commits.
