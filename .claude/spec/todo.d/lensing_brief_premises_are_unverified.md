---
section: Backlog
---

- **BUILD-BRIEF "MEASURED FACTS" ENTER AS AXIOMS AND NOTHING RE-MEASURES
  THEM** `[housekeeping]` — briefs carry a "Measured facts (do NOT re-derive;
  each cost real engine time)" section. The instruction is deliberate:
  re-measuring in-build is slow and deepens transcripts. The consequence is
  that a stale premise is unfalsifiable inside the build.
  MEASURED INSTANCE (2026-08-07, `subdivision_recursion`): the brief's fact 2
  asserted `r_caustic(0.9, pi/2) = 5.67376` against an exact 5.69210, a 0.32%
  error, and scoped WP2 as the branch-selection fix to close it. That fix had
  ALREADY LANDED in an earlier build — both the pre-build and post-build trees
  return `5.692099788303084`, bit-for-bit, 0 ULP apart. The error did not
  exist when the build started.
  It survived every layer, and no layer failed at its job: the Architect
  planned a WP around it and wrote the "must become 5.69210 (was 5.67376)"
  acceptance; the Professor reviewed the WP's reasoning and passed; the Test
  Developer wrote a SHARD D test pinning the corrected value, which PASSED
  trivially because the value was already correct; three Inspector rounds
  (~$16 of Opus) raised one finding, about documentation. No role is
  chartered to re-measure a brief premise, and a test pinning only the
  CORRECTED number passes identically whether or not correction was needed.
  The build still delivered a real 10.6x `r_caustic` speedup, so the WP was
  not wasted — but its stated purpose was fiction. The same section's OTHER
  premise (200 calls = 1.85 s) reproduced accurately at 1.788 s, so this is
  not "driver numbers are unreliable"; it is that stale and fresh facts are
  indistinguishable once written down.
  FIX CANDIDATES: (1) every quantitative brief premise carries the SHA it was
  measured at, and the Architect re-runs a one-line probe for any premise
  whose SHA is not an ancestor of the build's HEAD — one shell call per
  stale-suspect fact, not a re-derivation; (2) for any WP justified by a
  defect, acceptance pins the CHANGE (pre-value differs from post-value), not
  only the post-value; (3) driver-side, re-run the probe if the brief sat
  unlaunched or intervening builds touched the same file — this fact was ~1
  day old with two builds in between. (2) catches it in-build, (1) catches it
  at plan time more cheaply; they compose.
  Same root shape as [[lensing_golden_fixture_recomputes_geometry]]: a
  provenance claim that is true when written and silently false later, with
  nothing in the loop that re-checks it.
