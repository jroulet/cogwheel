---
section: Backlog
tags: [housekeeping]
---

# Resolve recurring test-only consumer warning for lens_amplification_surrogate

`scripts/sync_derived_docs.py` has flagged the same four test-only callers of
`LensAmplificationSurrogate.load` across four or more librarian sessions without
resolution:

```
[consumer_graph] lens_amplification_surrogate: actual consumer
  'cogwheel/tests/test_lensing_surrogate.py::SerializationMultiChartTestCase.test_round_trip_preserves_every_chart_field'
  (via LensAmplificationSurrogate.load) is not in DATA_CONTRACTS.yaml
```

The four flagged callers are all in `test_lensing_surrogate.py`:
- `SerializationMultiChartTestCase.test_round_trip_preserves_every_chart_field`
- `SerializationMultiChartTestCase.test_round_trip_preserves_full_provenance`
- `SerializationMultiChartTestCase.test_round_trip_served_values_are_bit_identical`
- `SerializationTestCase.test_npz_round_trip_is_bit_identical`

**Current convention**: DATA_CONTRACTS.yaml consumer lists are production-only;
test-only callers are intentionally excluded. The consumer-graph tool flags them
as warnings rather than errors.

**Action required** (contract owner): choose one:
1. Explicitly annotate the `lens_amplification_surrogate` entry in
   DATA_CONTRACTS.yaml with `test_consumers_excluded: true` (if the schema
   supports it) so the tool can suppress the warning, OR
2. Update `sync_derived_docs.py` to filter out test-only callers from the
   consumer-graph diff (the general fix), OR
3. Confirm the warning is intentionally accepted noise and add a suppression
   comment in the consumer-graph cache.

Until resolved, every Librarian post-commit run will re-flag this and re-note it.
