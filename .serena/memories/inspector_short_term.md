# Inspector Short-Term Observations

## Build 8 — Part 0 Mechanical Test Suite (2026-08-xx)

### Scope
Review of new test file `cogwheel/tests/test_lensing_part0_mechanical.py` — a
pure AST/text scanning suite enforcing three of four Part 0 structural
invariants on `cogwheel/lensing/`:
1. No prior-box-derived constants (diagonal ≈ 4.2426, half-width 3.0 with box-like name)
2. No retired concept names in production symbols/exports/source
3. No new discretization-absorber constants without explicit allowlisting
(4th invariant — "no stepping where closed form exists" — is explicitly
optional per the brief and not implemented.)

### Findings

**One trivial finding (carried forward from previous review, still open):**

**INS-9-001 (trivial):** `_ABSORBER_ALLOWLIST` contains 5 of 10 entries that can
NEVER match `_ABSORBER_PATTERN` (which requires `^_[A-Z][A-Z0-9_]*(_EPS|_MARGIN|_FRAC|_STANDOFF|_SAFETY)$`):
- `_DEFAULT_FARFIELD_OVERLAP` (suffix _OVERLAP not in set)
- `_INTERLOBE_CORRIDOR_ETA_SCALE` (suffix _SCALE not in set)
- `CROWN_CAUSTIC_MARGIN` (no leading underscore)
- `_MARKER_SCALE_FLOOR` (suffix _FLOOR not in set)
- `_U_MARGIN_CONST` (suffix _CONST not in set)
These are dead entries that add maintenance noise without protecting against
anything. Not a correctness issue; they may be defensive against future pattern
widening but this intent is undocumented.

### Production Code Assessment
No production code changed. This is a test-only build. All 13 tests pass in
0.46s. The test correctly:
- Uses only stdlib (ast/json/pathlib/re/unittest) — no circular imports
- Scans only `cogwheel/lensing/` (not tests/)
- Validates retired_concepts.json structure and deduplication
- Includes self-falsification (5 tests proving detectors have teeth)
- Has an anti-vacuity test (confirms >10 files, >20 constants scanned)
- All allowlisted constants verified to exist in the codebase

### Carried Forward (pre-existing)
- INS-8-001: `test_raising_constant_to_two_refuses_an_admit_config` still
  fails (pre-existing from Build 6 C5).
- INS-5-001: SPEC.md lines 53/97-137 old annulus references — Librarian.
- INS-5-003: DATA_CONTRACTS.yaml line 228 'caustic-frame annulus rho' — Librarian.

### Verdict
**PASS** — one trivial finding (dead allowlist entries, INS-9-001 still open),
no bugs or design issues introduced by this build.
