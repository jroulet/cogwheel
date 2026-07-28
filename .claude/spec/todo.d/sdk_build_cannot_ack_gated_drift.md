---
section: Backlog
---

- **A build cannot satisfy the gated-drift gate, so any signature change
  strands it at commit** `[housekeeping]` — observed on the saddle lobe-serve
  build (2026-07-28): all WPs implemented, Inspector PASS, Professor PASS,
  tree gate GREEN (858 passed / 126 skipped / 5 xfailed), and then
  `Build failed: Spec/doc discipline hook blocks the commit and preflight
  could not auto-remediate`. HEAD unchanged; the driver had to finish it.

  The blocker was `.claude/hooks/check_gated_test_drift.py`, correctly: WP2
  changed `_evaluate_chart` and `select_chart` signatures and eight
  gated/skipped tests reference them. Those tests cannot report their own
  breakage, which is exactly the hole the hook covers — the suite was green at
  858 while their status was unknown.

  THE GAP: the only two exits are `GATED_DRIFT_ACK` (a per-test statement that
  the committer RAN the test under its tier) and `--no-verify` (blanket). A
  build can do neither honestly: running the gated tiers in-build is forbidden
  (slow tiers never run in-build), and `--no-verify` would also skip the
  correctness gates ahead of it. So a build that changes any signature with
  gated references strands at commit BY CONSTRUCTION, regardless of whether
  the change is sound. Here it was sound: the driver ran all eight classes
  under `COGWHEEL_TRAIN_TIER=1` and got 28 passed / 1 xfailed / 0 skipped /
  0 failed.

  SELF-INFLICTED AMPLIFIER worth recording: the same day, census tests were
  tiered behind `COGWHEEL_TRAIN_TIER` (184s -> 63s fast tier). Three of the
  eight flagged classes are ones that tiering moved OUT of the default run.
  Tiering therefore enlarged the drift hook's blast radius. Both changes are
  individually correct; their interaction strands builds. Expect this coupling
  to grow as more suites are tiered.

  Options, roughly in order of preference:
  (a) Make the SDK's commit step SURFACE the drift output as a driver decision
      rather than a RuntimeError — the build stops, reports the flagged tests,
      and the driver runs the tiers and acks. This is what happened manually;
      mechanising it removes the "failed build" framing from a build whose
      code was fine.
  (b) Let a build record a PENDING-ACK manifest (the flagged test list) that
      the driver's post-build step consumes, so the ack is a checklist rather
      than an archaeology exercise.
  (c) Have the build run ONLY the flagged gated tests (not whole slow tiers)
      when the flagged set is small. Risks re-importing slow work into builds;
      needs a cap.

  Do NOT resolve by weakening the hook. It did its job: nothing else in the
  pipeline would have told anyone those eight tests were at risk, and the
  green 858 actively concealed it.
