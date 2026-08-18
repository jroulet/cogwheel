# Live session state (2026-08-17 ~16:55 EDT)

QUEUE ITEM 4 COMPLETE: f-constants RULED with provenance
(.claude/handoff/f_constants_decision.md, commit 5ceb2b3): f_max=0.40,
f_floor=0.08 both parities; the bar gap at n_theta=7 is a PER-BAND
DENSITY allocation for the tiling design (astroid 0.10-0.40 + saddle
~1.1 flagged). Full-arc astroid sweep numbers were INVALID (untrimmed
arcs); the 6-way sharded re-sweep on F083-knee-trimmed arcs is the
authoritative astroid evidence. NEW CAMPAIGN BLOCKER FILED:
todo.d/lensing_tube_trainer_resolvable_subarc_trim.md (promote the F083
trim into the trainer; sequence with/before tiling design).
RE-SEQUENCED after the lobe-gauge probe + owner low-w directive:
5a. LOW-W DIFFRACTIVE ANALYTIC RUNG build (owner: engine fall-through
    at the band bottom unacceptable). Brief in prep
    (.claude/handoff/low_w_diffractive_rung.md). One rung, three
    closures: astroid F070 bottom, deltoid below-split hosting, every
    band-split's floor. Certificate-gated, no measured constants,
    census re-gated IN-BUILD (mirror-fidelity law).
5b. Census re-run -> remaining deltoid mid-band demand.
5c. Deltoid far-field redesign build sized to the remainder — brief
    drafted (deltoid_farfield_redesign.md) but MUST absorb the probe
    first: real demand is 12.67% of prior (lobe-gauge probe,
    saddle_residual_lobe_gauge_probe.md — F073 quantified: 443/868
    census-'interior' + 824/852 'shell' are genuine far-field), and
    the main job is extending rho_outer (1.25-2.40) to the prior edge
    (|y|=3), cusp rays on tile boundaries.
Then: trainer-trim + tiling design -> ONE campaign -> map retrain ->
2b arm-extension (wave_refused 2.13% -> 0) -> 7b census.

CLOSED TODAY (2): c3_band_split_zero_refusal — commit 6958f0c (code) +
b097ce1 (census mirror re-gate): saddle_c3 0.32%->14.09%,
ppgo_above_ceiling 0->15.87%, wave_refused 12.03%->2.13%,
engine_residual 72.25%->53.30% (re-gated census
demand_census_post_c3_regate_10k.json). saddle_c3 fragment CLOSED;
lensing_wave_refused_to_zero stays OPEN at the ZERO bar — the 2.13%
residual (above-150, 150*min_dt<4 unresolved corner) is the
arm-extension (2b) build, to land before 7b. Hand-finish pattern used
again: tree gate red on stale suites -> triage subagent -> fixture
surgery (15->10 ppgo-ceiling file; refusal fixtures deterministic via
inverse_transform + lnprior premise) -> driver gate re-run ALL GREEN
(2432) -> driver commit.
Brief: .claude/handoff/c3_band_split_zero_refusal.md (verified facts at
6a3f43c; tripwire scoped to the engine-reachable overlap band w<=150).
Acceptance: census wave_refused -> ZERO (or named measure-zero set),
saddle_c3 live, byte-exact null-split identity, full fast suite green.
On terminal: driver acceptance = census re-run (queue item 3) comes
right after.

CLOSED TODAY: tube_beat_free_representation — F083 quote (n_theta=10,
eps=4.2652e-03 vs 0.0237 bar), commit 69c79b8; close-out crew passes
done (Librarian e195e82, Dreamer 2de2b19, Tidier 7243c9b).

# Older re-orientation notes (superseded)

HAND-FINISH IN PROGRESS: build `tube_beat_free_representation` launch 5
ran the full DAG — Inspector PASS (after driver escalation rulings: fix
INS-5-001/002 caller threading, accept INS-3-PARSE with driver-side
verification), Professor PASS — then the TREE GATE went RED (9F+5E, all
legacy tube fixtures in test_lensing_surrogate.py hitting the new F_ref
gate; the build terminated WITHOUT committing; work preserved in tree +
refs/sdk/coder_checkpoint). Driver recovery: fixer subagent repaired the
fixtures (assertions/golden literals byte-identical) and re-pointed the
2 overlap-band precedence tests as structural probes (require_fref=False
tube leg; new fragment
todo.d/lensing_tube_exterior_double_match_dead_branch.md files the
dead-branch question); test_lensing_surrogate.py 128/128 serial AND
xdist. NOW RUNNING: driver full-gate re-run
(/tmp/driver_gate_rerun_beat_free.log) + F083 (n_theta, eps) extraction.
THEN: driver commit of the whole build tree (pathspec-sweep; the
build's pre-staged index entries and baseline-untracked test files must
be included — the orchestrator's own commit would have excluded them),
hand-run /doc-sync + /dream (build stranded pre-commit), quote (n_theta,
eps) in the completion record, close the todo.d fragment, push.
Launch-5 SDK fixes landed for future builds: keepalives, JSON repair,
verification revision feedback, test_dev continuation, Librarian wrap,
--resume-plan (approved plan archived:
.claude/handoff/tube_beat_free_approved_plan.json).
NEXT BUILD'S BRIEF IS READY: .claude/handoff/c3_band_split_zero_refusal.md.
If a future launch needs a plan: review per the brief
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
