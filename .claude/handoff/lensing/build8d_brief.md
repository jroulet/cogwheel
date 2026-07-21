# Build 8d — Homogenization: Schwinger as THE exact evaluator

## Mission

Dissolve the deliberate Build-7a interim state. Today the engine runs
TWO exact wave evaluators: the legacy dd/1F1 operator-series path
(bit-frozen, serving certified positive-parity configs) and the
Schwinger 1D quadrature (serving saddle parity + the strong-shear
rescue). Owner directive: "make it homogeneous" — Schwinger becomes
the SINGLE production exact evaluator on BOTH parities; the legacy
path is DEMOTED to oracle duty (regression gate, never the hot path);
and the w range is pushed by ROUTING (see fences — the quadrature
ceiling itself is an arithmetic wall, measured below).

Three concerns:

1. **Single-evaluator dispatch.** `F_op`/`F_op_grid`'s positive-parity
   arm routes through the Schwinger evaluator (same
   reduce/rotate/reconstruct as the existing saddle arm with
   `lam = 1 - kappa`); the parity dispatch collapses to one exact
   path + the geometric branch. Refusal vocabulary unchanged; the
   gamma' = 0 (shear-free) positive-parity configs — which Schwinger
   cannot represent (it requires gamma' > 0) — keep the legacy path
   as their named serving route (an explicitly scoped, documented
   exception, NOT a parallel production path; measure its prior-box
   hit fraction and report it).
2. **Oracle demotion with a byte-level overlap gate.** The legacy
   evaluator moves behind an oracle interface used only by tests. An
   OVERLAP-DOMAIN regression harness certifies Schwinger vs legacy on
   the certified overlap (positive parity, w <= 60, gamma' > 0):
   agreement at the owner-set gate (1e-10 relative, the F005/7a
   standard), refusal-decision identity both directions, and the
   fixture pins re-baselined ONLY per the F017 discipline (physics
   quantities keep their tolerances; any pin that moves must carry a
   contract-flip witness).
3. **w-range routing.** Resolved high-w configs (w*delta_min >= 4.0,
   L > 48) are already served by the geometric branch to the engine
   ceiling w <= 500 — audit and, where certification supports it,
   WIDEN the geometric branch's applicability band on both parities
   (its thresholds were calibrated conservatively pre-Schwinger).
   The unresolved-high-w near-caustic corner (w > 60, not
   geometric-eligible) stays a NAMED refusal — the cusp fast-serving
   build (8e) owns its uniform-asymptotics serving (scope fence,
   2026-07-20). Report the measured prior-box fraction of that
   refusal corner so 8e's value is quantified.

## Measured facts (pre-answered — do not re-derive)

- The Schwinger dd ceiling is an ARITHMETIC WALL, not a tuning:
  e^{pi w/4} cancellation costs 0.341 decimal digits per unit w
  against the dd accumulator's ~31.9. Driver-measured (ceiling
  patched): certified at w = 55/60 on saddle AND strong-shear
  configs; config-dependent at w = 64 (the 10-surviving-digit margin
  boundary, exactly as _schwinger.py documents); universal
  SchwingerCertificationError by w = 68; the certified-or-refuse
  contract held throughout. MORE NODES DO NOT HELP. A quad-double
  accumulator (~63 digits) would reach w ~ 155 at ~4x node cost —
  that is an OWNER decision, deliberately OUT of this build's scope
  (pose it in the plan's ESCALATION section if the Architect believes
  it should be scoped in; do not implement it).
- Warm Schwinger per-point at the ceiling: ~300-450 ms on a loaded
  box (certification pair included); earlier measurement 30-125 ms
  mid-band. The surrogate (8c, off by default) is the production
  speed layer — homogenization does NOT need the exact path to be
  fast, it needs it to be SINGLE and certified.
- The 7a cross-parity fallback already runs Schwinger on positive
  parity for rescued strong-shear configs, oracle-certified at 1e-10
  uniformly — the overlap gate has a proven precedent and harness
  idiom (test_lensing_fast_path.py's rescue certification).
- gamma' = 0 is measure-zero in the sampled box (gamma uniform on
  (0, 1.6)) but reachable in unit tests and by users; the legacy
  exception path exists for it. Prior-box hit fraction of the
  UNRESOLVED high-w refusal corner: UNMEASURED — measuring it is
  in-scope (cheap census over prior draws, geometry-only).
- Positive-parity bit-frozen claims: the crown byte-identity pin
  (surrogate suite) and the ratio-layer cache determinism pin against
  the CURRENT positive-parity evaluator. Homogenization CHANGES the
  positive-parity wave values within 1e-10 — those byte-level pins
  WILL flip and must be re-baselined with contract-flip witnesses
  (7b precedent), while physics tolerances (RB-vs-brute, oracle
  agreement) must hold unchanged.

## Out of scope — hard fences

- NO quad-double / precision-substrate work (owner decision; escalate
  in the plan, do not implement).
- NO Airy/Pearcey/uniform-asymptotics serving (8e owns ALL of it —
  scope fence 2026-07-20; the high-w near-caustic corner refuses by
  name in this build).
- NO surrogate retraining, NO full-box training, NO sampling/PP
  (ruling A), NO enable-by-default changes.
- The legacy evaluator's CODE is not deleted — demoted to oracle duty
  behind a test-facing interface (deletion is a later housekeeping
  decision once the oracle role is stable).
- Refusal vocabulary: no new exception classes; every refusal stays
  named and certified-or-refuse everywhere.

## Acceptance (two-tier)

1. In-build (FAST): overlap-domain harness green — Schwinger-vs-legacy
   1e-10 relative on a both-regime sweep (crown, two-image, strong
   shear approaching the ceiling), refusal-decision identity with
   zero flips both directions; single-dispatch proof (AST/grep-level:
   no production call site reaches the legacy evaluator except the
   documented gamma'=0 route); geometric-branch widening certified
   against the existing mpmath oracles at unchanged tolerances;
   byte-pin re-baselines carry contract-flip witnesses; the F010
   mutation idiom reds the new dispatch (a corrupted Schwinger route
   must be catchable); full lensing-suite green at fixture scale.
2. POST-BUILD (driver): parallel full-suite gate; serving-path price
   points re-measured (expect unchanged — the hot path is
   surrogate/geometry, not the exact evaluator); the measured
   unresolved-high-w refusal fraction reported to the owner alongside
   the 8e brief draft; SDK port-checklist consolidation ships in the
   same window.
