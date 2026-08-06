---
bump: patch
---

### Fast-tier tests are bounded at 900 s

Recorded `cogwheel/tests/conftest.py` under Conventions: a per-test wall-clock
ceiling that applies only when every slow tier is off, yields to an explicit
`--timeout`, and no-ops without `pytest-timeout`.

Convention only — no behavior change in `cogwheel/`. Motivated by F061: four
tests reaching `f_schwinger` above `w = 60` (~85-120 s per call versus ~0.2 s
on the double-double path) held xdist workers until the tree gate burned its
entire 3600 s ceiling and stranded a build, naming no test.
