---
section: Backlog
---

- **TWO PRE-EXISTING SPEC/CODE DIVERGENCES, found by the Librarian while
  syncing something else** `[→ spec]` — surfaced 2026-08-12 during the
  post-commit sync for `LobeExteriorChart` (INS-5-001). Both predate that
  work and were correctly left unfixed: the Librarian owns SYNC, while
  spec-vs-code ACCURACY is an Inspector-owned invariant, so the fix lives
  upstream in the spec or the code, not in a doc edit.

  1. **`theta_to_u` is documented REQUIRED but read softly.** SPEC.md and
     DATA_CONTRACTS.yaml both state that `LobeInteriorChart`'s `theta_to_u`
     NPZ key is REQUIRED and that a missing map hard-refuses with `KeyError`
     at load. `_chart_from_npz` actually reads it with a soft
     `data.get(...)`. So a stale or truncated artifact loads and serves on a
     silently absent angular map instead of refusing.
     DECIDE WHICH IS RIGHT, then make the other match. The documented
     behaviour is the safer one and matches the wedge convention deliberately
     adopted for `theta_to_u` (hard refusal so a stale artifact cannot
     mis-serve on the wrong angular coordinate); the soft read looks like it
     arrived with a fix for a round-trip bug that never updated the docs.
     Note the NEW `lobe_exterior` kind specifies the hard-KeyError behaviour,
     so leaving the interior soft makes two sibling chart kinds disagree on
     load strictness.

  2. **Multiplicative vs additive, contradicted between the two surfaces.**
     SPEC.md and DATA_CONTRACTS.yaml disagree on whether the astroid exterior
     arm's `_to_caustic_fixed` coordinate is directional-MULTIPLICATIVE
     (`rho = |y| / _caustic_reach`) or a scalar ADDITIVE offset
     (`rho = 1 + |y| - _caustic_reach`). One of them is wrong. This matters
     because the additive form is exactly what made the saddle exterior
     unrepresentable in the corridor (negative rho), and the parity-dependent
     split between the two forms is load-bearing.

  Both are cheap to settle by reading the code once; neither needs a build.
  Found by the sync recorded in
  `.claude/spec/spec_changelog.d/2026-08-12_lobe_exterior_chart.md`.
