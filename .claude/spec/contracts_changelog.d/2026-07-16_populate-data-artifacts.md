---
bump: minor
---

### Register cogwheel's disk data artifacts

Populated the previously-empty `artifacts:` registry with 11 disk-mediated data
products, verified against the code's write/read sites: `posterior_samples`
(samples.feather), `sampler_config_json`, `posterior_config_json`,
`postprocessing_tests_json`, `profiling_stats`, `event_data_npz`,
`example_asd_npy`, `events_metadata_csv`, `coherent_score_lookup_tables`,
`injection_set_feather`, and `injection_json`. Each carries its producer and
declared consumers at module + function level so agents can query the data-flow
graph (`scripts/pipeline_graph.py`) instead of re-discovering it. `fields` lists
are intentionally omitted until exact column/attribute names are confirmed.
