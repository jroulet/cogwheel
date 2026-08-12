---
date: 2026-08-12
section: Backlog
---
# Consolidate duplicate routing-path pins — RESOLVED (commit 26d088a)

The recurring "stale fixture" churn (every serving-ladder improvement
re-pointing the same routing assertions across many files) was mostly
self-inflicted. Routing decisions were pinned as code-path assertions
(`route == 'ppgo'`, `dispatched to f_schwinger`, `calibration invoked`)
in duplicate across files, instead of as the served VALUE against the
canonical oracle.

`test_lensing_airy_fold.py`: 42 → 22 route-pin sites across 21 methods.
All load-bearing value assertions preserved (served-value-vs-exact,
envelope bars, byte-identity); one pure-path spy test replaced with a
census anti-vacuity assertion. Other files already held their canonical
pins in the right home. Net −40 lines. Full lensing fast tier green.

Principle reinforced: assert VALUES (served amplification vs oracle), not
which code path produced them. One canonical pin per routing decision, in
the predicate-owning file.
