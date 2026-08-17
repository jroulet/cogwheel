# Build: beat-free tube representation — demodulate by both fold carriers

## Mission

The tube chart's stored envelope oscillates as `cos(w * Delta_tau)`
because TWO fold-pair carriers are demodulated by ONE. No coordinate
removes a beat — measured: 30 delay-uniformized nodes gave eps 0.145 vs
0.146 at 24 arc-length nodes (the reverted graduation build's own
numbers), while the brute ladder needs 48 nodes for 0.0237. Fix the
REPRESENTATION: demodulate the tube envelope by BOTH fold carriers —
`Delta_tau(theta, eta)` is closed form from the step-1 cascade, the SAME
quantity `_airy_fold` computes for `xi` (DRY: import, never re-derive) —
so the stored residual is beat-free and its structure does not scale
with w. Expected: tube theta nodes collapse to the smooth-variation
scale (~5-10), passing the F083 accuracy bar at a fraction of the brute
baseline. The no-explosion vision realized for the tube; blocks tube
training and the (f_max, f_floor) sweep.

## Facts (measured; the graduation post-mortem is authoritative)

1. F083: envelope 5.3x swing, 8 extrema over 1.34 rad at w ~ 52
   (gamma=0.4 astroid, arcs[0]); nodal-error ladder 6 -> 0.40,
   24 -> 0.146, 48 -> 0.0237 vs the 5e-2 bar. The beat is the fold
   pair's interference; identity-path (the D2 fold is exonerated,
   mapping error 0.0000).
2. The reverted graduation build (diff archived at
   `.claude/handoff/tube_graduation_salvage/working_tree.patch`):
   cherry-pick its KEEPER pieces — tube schema tag + hard-refusal of
   stale artifacts, the DRY delay-equality import + 1e-12 pin, the
   `_heldout_eps` unserved-as-coverage reporting, the engine-free
   census wiring — NOT its s' axis (its TV coordinate is flat at
   Delta_tau's mid-arc extremum: "Tube delay map is not strictly
   increasing" on every real arc; moot here, cautionary only).
3. The gate also caught, in the reverted tree: a lobe-path regression
   (`LobeUCoorDBoundShiftMarginTestCase` eps-stability 0.84 vs 0.01)
   and a part0 absorber-guard hit. This build must leave the lobe path
   BYTE-IDENTICAL and pre-clear any new module constant against the
   part0 guard (allowlist-with-justification or import the production
   constant — the F-sweep DRY precedent).
4. Representation design (Professor decides the exact form in-plan):
   the switched-channel machinery already carries RESOLVED pairs as
   separate carriers with smootherstep weights; the tube's beat appears
   where the pair is NOT separately carried. Candidates: (a) demodulate
   by the symmetric two-carrier factor
   `2 cos(w * Delta_tau) e^{i w tau_bar}` with analytic
   `Delta_tau(theta, eta)` baked into the chart's serve-side
   reconstruction (zero-crossing care: cos vanishes — store against a
   NON-VANISHING analytic pair combination, e.g. the complex pair sum
   with Airy-uniform amplitudes, the `_airy_fold` object itself); (b)
   store the residual against the fold arm's uniform-Airy serve where
   it certifies and against the two-carrier geometric sum where
   resolved. The choice must be NON-VANISHING across the band (no
   division by a zero of the carrier), analytic (no new measured
   constants), and serve-side mirrored exactly (envelope-definition
   tag; the FARFIELD_KERNEL_SUM_MINUS_GHOST precedent for label +
   mirror + contracts fragment).
5. Real-arc buildability is an ACCEPTANCE item: the four bands the
   Professor probed (astroid small/large gamma incl. 0.045, saddle 1.2)
   must build without error.

## Scope

IN: the two-carrier (or arm-residual) demodulation in
`_build_tube_chart` + the serve-side mirror in the tube branch of
`_evaluate_chart`/`_tube_serves` path; envelope-definition tag + schema
bump + contracts fragment; the salvage keepers (fact 2); the F083
falsification RUN (the accuracy half: gamma=0.4 astroid at the new
node count vs the 0.0237 bar); real-arc buildability on the four bands;
fast tests (value pins on the new representation's round-trip, the
non-vanishing-carrier guard, both parities).
OUT: the s'/TV axis (dead); lobe/wedge/exterior paths (byte-identical);
the Nyquist-count machinery beyond a simple safety cap; training
campaigns; the f-sweep (driver, after); serving-ladder changes.

## Acceptance

- gamma=0.4 astroid tube: eps <= 0.0237 at <= ~12 angular nodes
  (report the exact count and eps; the expectation is ~5-10 — if the
  beat-free residual still needs > 24 nodes, STOP and escalate: the
  representation choice is wrong, do not brute-force).
- All four Professor bands build and serve; lobe path byte-identical;
  part0 guard green; DRY delay-equality pin 1e-12; stale-schema
  hard-refusal; full fast suite green.

## Constraints

Branch claude-dev; fragments (closes
`todo.d/lensing_tube_beat_free_representation.md`, `[→ spec]` +
contracts); values-not-paths; no measured constants in the
representation; in-build tests fast; escalate rather than iterate on
any surprise — the last two tube builds both died to premises their own
tests could not see; the Professor's plan-time derivation of the
demodulation factor is the load-bearing step, spend the review there.

## RECOVERY NOTE (2026-08-17, second launch)

The first launch died at test_dev error_max_turns during Inspector
revision 1 (findings: 2 implementation + 1 design, unresolved). The
working tree carries WP1-WP3 substantially complete (coder checkpoint
980f7e3d9229) plus partially-reconciled tests. Survey before planning;
re-scope to finish, not rewrite. The test reconciliation that exhausted
the budget involves the D2-fold and surrogate suites' interaction with
the new representation — apply the parsimony contract (re-point existing
pins, never rewrite suites) and budget the test WP accordingly. The
F083 falsification's RESULT is not on record — running it and quoting
the (nodes, eps) pair in the completion record is unconditionally part
of acceptance, whatever else gets re-scoped. The >24-node escalation
tripwire stands.
