# Session: Part 0 Mechanical test review (test_lensing_part0_mechanical.py)

## Build under review
Two new test additions:
1. `test_no_retired_names_in_live_docs` on `TestNoRetiredConceptNames` — scans 3 live spec docs for retired concept names
2. `TestNoDocstringAbsorberLanguage` — AST-scans surrogate files for forbidden docstring phrases on constants

Plus self-falsification companions: `test_live_doc_detector_fires` and `test_docstring_absorber_detector_fires` in `TestSelfFalsification`.

## Verdict: PASS
- All 18 tests pass in 0.92s (well within budget).
- `test_no_retired_names_in_live_docs`: All 3 live doc files exist (640 lines total); scans against 4 registered retired concepts; 0 violations. Self-falsification `test_live_doc_detector_fires` proves detector fires on synthetic tempfile injection.
- `TestNoDocstringAbsorberLanguage`: Both target files (surrogate_training.py 226KB, surrogate.py 220KB) are parsed. Zero constant-docstrings found (files use `#:` Sphinx comments, not `Assign → Expr(str)` pattern). This is correct — the test is a regression guard for the specific docstring-on-constant convention. Anti-vacuity checks file parsing succeeded. Self-falsification proves the detector fires on synthetic input.
- Minor note: spec's diagnostic recommends adding 'annulus', 'ANNULUS_INNER_RADIUS' to retired_concepts.json — currently not present (registry has 4 entries: _WEDGE_EPS, _PROBE_ETA, _CLOUD_MARGIN_FRAC, _CUSP_SPEED_REL_FRAC). This doesn't affect test correctness but would broaden coverage. Deferred to a future build.
- The test structure is logically sound: the docstring absorber test catches a specific bug-class (constants whose docstrings admit they exist to absorb artifacts), while being vacuously true now — the correct posture for a guard test.
