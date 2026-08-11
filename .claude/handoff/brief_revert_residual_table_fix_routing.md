# Build Brief: Revert residual-table reformulation; fix cusp-arm routing

## Mission

Revert the over-engineered residual-table reformulation from the zero-quadrature build, and apply the actual minimal fix: correct `_cusp_vertex` routing so interior cusp sources (3 comparable images, inside the caustic near a cusp) are served by the demodulated Pearcey table, with exterior cusp sources served by ppGO. No live quadrature in the hot path.

## Professor verdict (2026-08-11, measured 568 served configs)

- The x=-71 "structural barrier" that motivated the residual table exists ONLY in the expanded box's corners (R~115), never reached by a served source: the 0.07 rad cusp window forces small delta_parallel -> |x| <= 7.95 (demodulated table box is x +-27.6, so it fully covers the served x-range).
- Every served EXTERIOR cusp config has R >= 71.6 > r_ppgo_min = 71.1, so the ppGO rung preempts the table — the table is never consulted for exterior cusp sources.
- The residual format was also numerically unstable near the fold caustic (|P_asymp| ~ 1e9) and was reverted by the Inspector (INS-1).
- The table's REAL role: INTERIOR cusp sources (inside the caustic, 3 comparable images), where R < 71 fails the ppGO gate. These refuse due to a `_cusp_vertex` ROUTING BUG: it seeds via `nearest_caustic_point` (image-theta), which can snap to the wrong cusp or a fold segment, giving wrong (x, y) controls and R too small -> arm refuses -> exact engine.

## Work

1. **Revert residual-table residue**: ensure `_pearcey_table.py` is the demodulated format (the Inspector's INS-1 revert of 0.2.0 restored demodulated; verify the current working tree state is consistent — schema should be the demodulated format, `demod_real`/`demod_imag` keys, `derive_box` or the radius-based box WITHOUT the residual semantics). If any residual-format residue remains, revert it.
2. **Fix `_cusp_vertex` routing** (the core fix): `_cusp_vertex` currently brackets the cusp nearest to the `seed_theta` from `nearest_caustic_point` within +-0.1 rad. The bug: for interior sources near a cusp, the seed (image-theta nearest critical point) can route to the wrong cusp or a fold segment. Fix: probe ALL nearby cusps (the four astroid cusps at phase {0, pi/2, pi, 3pi/2} for positive parity; the six deltoid cusp candidates for the saddle) and select the one whose SOURCE-PLANE distance `|source - vertex.source|` is minimized. This gives the correct (x, y) controls (larger R for interior cusp sources), passing the radius gate, so the demodulated table serves them.
   - Preserve the existing refusal semantics (wedge-edge saddle cusps still refuse; LensDomainError -> None).
3. **Regenerate + ship the deleted `pearcey_table.npz`**: the artifact was deleted (INS-2-001) and blocks tests. Regenerate via `scripts/train_pearcey_table.py` at the demodulated format and commit it so the shipped package has a working default table.
4. **Verify**:
   - Interior cusp sources (inside the caustic, 3 comparable images) serve via the demodulated table (not exact engine) after the routing fix.
   - Exterior cusp sources continue to serve via ppGO (R >= 71).
   - No live quadrature in the hot path (`_consult_pearcey` refuses, never calls live quadrature).
   - The tree gate passes (fix the two tests that errored on the missing table: PpgoGoldenAgreementTestCase, PpgoFinitenessGuardTestCase — they'll pass once the table is shipped).
5. Keep the zero-quadrature intent: the hot path is table + ppGO + spline, no live quadrature.

## Measured facts (re-probe at HEAD before coding)
- Served cusp-arm region: x in [-7.95, 7.95], y in [-233, -71], R >= 71.6 (568 configs, both parities, rho [1.1,5.0], w [10,200])
- ppGO crossover r_ppgo_min = 71.1; radius_min = 7.37
- `_cusp_vertex` at _pearcey_cusp.py ~489: brackets nearest cusp within +-0.1 rad from seed_theta (nearest_caustic_point)
- Astroid cusps at phase {0, pi/2, pi, 3pi/2}; saddle deltoid: finite wedge-tip cusp at lobe centre, diverging wedge-edge cusps (refuse)
- Current table artifact DELETED (D cogwheel/data/pearcey_table.npz); _pearcey_table.py has residual-format changes
- Affected tests: PpgoGoldenAgreementTestCase, PpgoFinitenessGuardTestCase (PearceyTable.load on default path errors)

## Constraints
- Fast tests. Follow AGENTS.md.
- This REVERTS the residual-table over-engineering and applies the minimal routing fix. Keep the demodulated table format.
- No live quadrature in the hot path.
- Refusal-conservative: never serve a wrong value silently.
- Plan-gate requirement: each `domain_test_descriptions` spec names exactly ONE primary `test_*.py`; no spec may reference another spec's primary file.

## Design note from the driver
The user's hypothesis ("maybe all the residual-table things were not needed and it would have worked by just fixing the routing + gates") was confirmed by the Professor. This build does exactly that: revert the residual reformulation, fix `_cusp_vertex` routing, ship the demodulated table. The Professor's evidence (R >= 71.6 for all exterior served, |x| <= 7.95) is the justification.
