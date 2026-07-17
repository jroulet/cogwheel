---
section: sdk
---
# Brief/plan depth guards in the SDK launcher [housekeeping]

Mechanical enforcement of the CLAUDE.md "SDK Build Briefs" discipline (the
belt-and-suspenders layer beyond driver judgment):

1. DONE (2026-07-17): `launch_build.sh` warns (stderr, non-fatal) on briefs
   >12 KB or referencing META_PLAN.
2. DONE (2026-07-17): plan gate prints `plan_depth_banner` (WP count + plan
   size, WARNING above 3 WPs) — gates.py, tested.
3. REMAINING: Orchestrator scopes context per agent — slice the approved
   plan so each coder receives its own WP + shared preamble, not all WPs +
   all test descriptions (largest single depth reduction available).
   Agent-visible change: prove on the next live build (Build 3) before
   porting.

Rationale: bare-denial rate is transcript-depth-correlated (0/106 in first
two calls, median call 14, issue #74351); gw's 37 shallow builds recorded
zero denials on the identical harness. Prove in cogwheel, then port to
teja-force skill + gw with the rest of the hardening.
