# Live session state (2026-08-17 ~10:40 EDT) — post-compaction re-orientation

RUNNING: build `tube_beat_free_representation` (THIRD launch, recovery).
Log /tmp/tube_beat_free_representation_20260817_103643.log, PID 1259398,
approval dir /tmp/tube_beat_free_representation_approval, monitor armed.
(Launch 2 was killed by the watchdog during a healthy quiet Architect
planning turn; fixed in 21a1abd — gate-wait beat capped at 900s,
planning/skill turns now route through _iter_query_with_timeout with
240s keepalive slices; the watchdog-starvation fragment is CLOSED.)
At plan-ready: review per the brief
`.claude/handoff/tube_beat_free_representation.md` (recovery note at the
bottom; the (nodes, eps) falsification quote is unconditional acceptance;
>24-node escalation tripwire; finish-not-rewrite; parsimony on test
reconciliation).

DRIVER LOOP: file-based gates (touch plan_approved / write plan_rejected
or escalation_fix with diff-verified rulings); on terminal: verify
commit, run driver acceptance, close fragments, PUSH origin/claude-dev,
launch the next build.

QUEUE (each fires on its predecessor; briefs/fragments carry the detail):
1. beat-free build lands -> commit/close -> push.
2. c3 band-split + wave_refused-to-zero build
   (todo.d/lensing_saddle_c3_band_split_serving.md +
   todo.d/lensing_wave_refused_to_zero.md — shared band-split machinery).
3. Cheap serve-route census re-run (scripts/serve_route_census.py)
   -> refreshed demand map (current one:
   .claude/handoff/demand_census_corrected_10k.json).
4. (f_max, f_floor) joint sweep on beat-free charts — runner
   /tmp/f_fraction_sweep.py, priced ~2.6-2.8h, w<=60 cap; F083 killed
   the old constants' provenance.
5. Deltoid far-field redesign
   (todo.d/lensing_deltoid_farfield_coordinate_redesign.md; needs the
   f-constants). MOVED BEFORE tiling+campaign (owner, 2026-08-17): the
   redesign changes the saddle-side charting strategy, so training or
   tiling the saddle sector first would be retrained-on-arrival — the
   same never-train-on-coordinates-about-to-change lesson as the killed
   50h campaign.
6. Demand-sized tiling design under the no-explosion gate, on FINAL
   coordinates both parities
   (todo.d/lensing_training_campaign.md, rewritten demand-first).
7. Residual training campaign (cost estimate FIRST, monitored) — ONE
   campaign, no planned retrain.
8. Certified-map retrain (F080 edge-biased binding + fan-asymmetry
   question) — after the campaign's charts exist.
9. 7b acceptance census (--with-artifact; zero engine-served AND zero
   wave_refused per the owner bar).

PARKED: Tidier backlog (60 files), Librarian queue — close-out batch.
STANDING RULES IN FORCE: cost estimate before engine runs; pairing gate
before oracle claims; diff-verify escalation-fix claims; no bare git
commit during builds; push at milestones; test parsimony at plan gates;
analytics-first / no-explosion / symmetry doctrines (owner).
