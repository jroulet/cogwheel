---
bump: minor
---

Registered `scripts/serve_route_census.py::main` as a consumer entry on
`lens_amplification_surrogate`, tagged `kind: script`. Unlike the
`kind: test` entries registered 2026-08-13/15, this is a genuine
PRODUCTION (non-test) caller: with `--with-artifact PATH`, `main` calls
`LensAmplificationSurrogate.load` and threads the artifact through `run`
so the census's `surrogate` route becomes reachable (the order-7b
acceptance-mode census); the engine-free demand mode (no `--with-artifact`)
never touches this artifact. Flagged by `sync_derived_docs.py`'s
`check_consumer_graph` during post-commit doc sync for the
`serve_route_census` + census band-ladder-fix batch; registering it
clears the advisory.
