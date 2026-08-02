# Professor Short-Term Observations

## Part 0 mechanical invariant tests review (2026-08-02)

- **test_lensing_part0_mechanical.py**: 13/13 pass (0.47s). All four test classes
  (TestNoPriorBoxConstants, TestNoRetiredConceptNames, TestNoNewDiscretizationAbsorbers,
  TestSelfFalsification) green.

### Verification details:

1. **Anti-vacuity**: 22 lensing .py files scanned, 136 module-level numeric constants
   found — well above the thresholds (>10 files, >20 constants).

2. **Prior-box diagonal (≈4.2426)**: No constants with |value − 3√2| < 0.01 found
   in the lensing tree. Correct.

3. **Prior-box half-width by name**: No constants with value=3.0 AND a box-name
   fragment (BOX, RANGE, EXTENT, etc.) found outside the allowlist. 4 allowlisted
   constants (_Y_SCALE_CAP, _SPLIT_BASE, two _SPLINE_DEGREE) correctly exempted.

4. **Retired concepts**: 4 entries in retired_concepts.json (_WEDGE_EPS, _PROBE_ETA,
   _CLOUD_MARGIN_FRAC, _CUSP_SPEED_REL_FRAC). None appear in lensing source.
   Registry well-formed (no duplicates, all required fields present).

5. **Absorber constants**: 5 constants matching `^_[A-Z][A-Z0-9_]*(_EPS|_MARGIN|_FRAC|_STANDOFF|_SAFETY)$`
   found, all in the allowlist. NOTE: The allowlist contains 10 entries but only 5 match
   the pattern — the others (_DEFAULT_FARFIELD_OVERLAP, _INTERLOBE_CORRIDOR_ETA_SCALE,
   CROWN_CAUSTIC_MARGIN, _MARKER_SCALE_FLOOR, _U_MARGIN_CONST) have names that don't
   match the regex. This is dead allowlist code — harmless but slightly over-specified.
   It doesn't compromise test safety (only means those 5 were never needed in the allowlist).

6. **Self-falsification**: All 5 mutation-detection tests pass, proving the detectors
   have teeth against synthetic violations.

### Minor concern:
- The absorber allowlist is broader than the regex catches (5 dead entries out of 10).
  This is spec–implementation drift — the spec may have anticipated a broader suffix set.
  No correctness impact.

- Heavy full-sampling validation is operator-deferred.
