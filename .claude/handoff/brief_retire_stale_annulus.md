# Build Brief: Retire stale annulus/fence references + extend Part 0

## Mission

Step 5 (C8, commit 65eebcb) retired the annulus concept and its fences,
but 61 references to "annulus" survive in tests and production code.
The standing rule says: "Default is DELETE, not re-point." Tests pinning
a retired boundary should be deleted, not updated to pin the new one.

Also: the Part 0 mechanical test (`test_lensing_part0_mechanical.py`)
only scans production code for retired names. It should ALSO flag tests
that pin retired constants or use retired terminology as if it's live.

## Scope

### 1. Add retired names to the registry

Add to `.claude/hooks/retired_concepts.json`:
- `ANNULUS_INNER_RADIUS` (retired by 65eebcb, replaced by caustic-relative rho)
- `GAMMA_FENCE` (retired by 65eebcb, consequence of annulus, not independent physics)
- `_SADDLE_GAMMA_FENCE` / `1.0502342` (same)

### 2. Clean production code

In `cogwheel/lensing/`:
- `_born.py`: "far annulus" in docstrings → "far exterior" or "Born exterior rung"
- `channels.py`: "far annulus" → "far exterior"  
- `likelihood.py`: "Born-annulus" → "Born residual"
- Any remaining references to the 3.0 boundary or the gamma fences

### 3. Clean tests

For each of the 61 "annulus" references in `cogwheel/tests/`:
- If the test pins a value that traces to `ANNULUS_INNER_RADIUS = 3.0`:
  DELETE the test (the boundary doesn't exist)
- If the test uses "annulus" as terminology but tests a legitimate
  concept that survived (e.g. the Born exterior rung): RENAME to
  use the current terminology
- If the test is a fixture comment explaining history: leave but mark
  with `# Historical: the annulus was retired in C8 (65eebcb)`

### 4. Extend Part 0

Add a test class `TestNoStaleTestTerminology` that:
- Scans `cogwheel/tests/` for retired concept names
- WARNS (not fails) on each occurrence so the test is informational
- Lists the offending files/lines for manual triage

This is a WARNING gate, not a hard fail — tests may legitimately
document retirement history.

## Acceptance

- `retired_concepts.json` has the annulus/fence entries
- Zero "annulus" references in production code (`cogwheel/lensing/`)
  that use it as a live concept (docstrings may explain history)
- Tests that pinned `3.0` or `0.75` or `1.0502342` as physics are gone
- Part 0 test still passes

## Constraints

- Do NOT delete tests that test LIVE functionality just because they
  mention a retired name in a comment. Only delete tests whose ASSERTION
  traces to a retired boundary.
- Fast tests only.
- Follow AGENTS.md and the spec/TODO workflow.
