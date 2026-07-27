---
date: 2026-07-27
bump: minor
---

Record the Born rung under Conventions as present but DORMANT: the analytic
weak-deflection module `chang_refsdal/_born.py` exists, is tested, and is
deliberately NOT wired into the serve path because its O(1) coefficient `b1` is
an unpinned placeholder giving ~13% disagreement with `operator.F_op` inside
its own gate's pass region.

Recorded rather than omitted so the annulus `3.0 < |y| <= 4.2426` is not
mistaken for covered: those draws still reach the exact engine.
