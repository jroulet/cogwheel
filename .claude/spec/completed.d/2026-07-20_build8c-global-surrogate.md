---
date: 2026-07-20
section: likelihood
---
### Build 8c (+8c-cont) — global surrogate artifact machinery (complete)

Multi-chart surrogate (WP1), training driver (WP3, driver-hand-
finished: band-safe arcs, stable-band bisection, robust probes,
eta_max 0.05, theta-wrap serve seam), census tool (WP-CS + owner-
approved binning-floor line), registration (WP-REG: DATA_CONTRACTS +
data_registry + LOADERS + consumer graph). Tests: multi-chart suite
in test_lensing_surrogate.py (selection determinism, serialization
round-trip, back-compat) + driver-commissioned
test_lensing_surrogate_census.py (27 tests; tiers measured
0.0148/0.0008/0.0163 nats vs bars 0.05/0.1/1.5; falsifiables at F018
bars). FINDINGS F018 (design-claim currency). SDK proving run:
fixes 1-8 exercised; port items 9-11 recorded (coder continuation,
decision-wait monitors, revision-loop test-dev path). Full-box
training deferred post-8e per owner re-sequencing.
