---
bump: patch
---

### Consumer-graph drift layer + contract name corrections

Added jedi+ripgrep consumer-graph drift detection (see
`scripts/regenerate_consumer_graph.py`, cached to `CONSUMER_GRAPH.json`) and
corrected contract function names it caught as not matching the code: the
coherent-score lookup-table producer/consumer is `LookupTable._get_table` (not
`_instantiate_table`), the profiling/tests consumer is `PostProcessor.make_table`
(not the non-existent `EventResultsDataFrame`), and the `event_data_npz`
reference-finder consumer is `ReferenceWaveformFinder.from_event`.
