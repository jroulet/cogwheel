---
date: 2026-08-13
section: Lensing
---

- **Interior fold-ppGO rung re-gated: exact 4-image predicate + raw-ppGO c3
  certificate** (build ppgo_interior_certificate; completion record written
  by the driver — the build's doc phase was skipped by a tree-gate failure
  unrelated to its work, see F072 test fallout). Leg 1 `rho <= 1`
  (`caustic_rho`, admits caustic-exterior sources at 58.7% of points — F073)
  replaced by `int(geom.real_mask.sum()) == 4`, the exact interior predicate
  (0/2400 disagreements vs the closed-form caustic). Leg 3
  (`_uniform_error_estimate`, a fold-arm heuristic that is NOT a bound)
  replaced by `geometry.ppgo_error_estimate` — the `w**-3` c3 term of raw
  ppGO's own stationary-phase series, ported from the validated reference
  algebra (reproduces shipped C1/C2 at 2.4e-15 / 5.8e-14) — under
  `_PPGO_INTERIOR_SAFETY = 2.0` (measured true/certificate ratio over 1248
  interior oracle points: median 0.587, p99 0.953, MAX 0.980, 0.0%
  optimistic). Leg 2 (`xi_min >= 4`) DROPPED BY MEASUREMENT, per the
  approved decision rule: every 4-image interior config fails it, while the
  certificate at S=2.0 admits 230 evidence-band points with MAX true error
  4.8e-5 against the 1e-4 bar and none over — the leg only suppressed
  certifiable service. No ghost term: four real roots prove the ghost is
  exactly zero (`GhostAbsentError`). Census-mirror staleness deferred to
  [[lensing_census_mirror_regate]]; SPEC gate-description staleness deferred
  to the Librarian (INS-1-001).

  Closes BOTH backlog items on this rung: the gate-inversion item (the
  fold correction never certifies on the interior — measured best 2.15e-3
  vs the 1e-4 bar — so the "invert the gate" alternative was rejected by
  measurement, not preference) and the serves-wrong item (the rung has
  served raw ppGO since 71a5051; its acceptance — "report the served error
  against an oracle valid at the SERVED w" — is met by the c3 certificate's
  derivation-backed extrapolation, stated with its assumptions in the
  retired fragment and FINDINGS).
