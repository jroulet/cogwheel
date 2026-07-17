---
section: sdk
---
# Brief/plan depth guards in the SDK launcher [housekeeping]

Mechanical enforcement of the CLAUDE.md "SDK Build Briefs" discipline (the
belt-and-suspenders layer beyond driver judgment):

1. `launch_build.sh`: warn (stderr, non-fatal) when the prompt file exceeds
   ~12 KB or references META_PLAN.md.
2. Orchestrator plan gate: surface WP count in the approval banner; flag
   plans with >3 WPs so the reviewer must consciously accept the depth.
3. Orchestrator: scope context per agent — slice the approved plan so each
   coder receives its own WP + shared preamble, not all WPs + all test
   descriptions (largest single depth reduction available).

Rationale: bare-denial rate is transcript-depth-correlated (0/106 in first
two calls, median call 14, issue #74351); gw's 37 shallow builds recorded
zero denials on the identical harness. Prove in cogwheel, then port to
teja-force skill + gw with the rest of the hardening.
