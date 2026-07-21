---
section: Backlog
---
- [ ] **Test-suite curation pass (8e window)** `[housekeeping]` —
  OWNER APPROVED (2026-07-21): audit all ~476 tests for obsolescence
  accumulated across Builds 3-8d. RETIREMENT RULE (binding, per the
  8b F010-retirement precedent): a test retires ONLY if its contract
  no longer exists AND its falsification power is PROVABLY covered by
  a NAMED successor test — retire with a pointer comment, never
  silently. Categories to audit:
  1. TRANSITION WITNESSES past their transition (schedule rule:
     old-vs-new witness comparisons retire ONE BUILD after their
     transition commits; the new-value pins stay) — 8d's evaluator-
     swap witnesses become eligible at 8e close; 7b-era pin witnesses
     already eligible.
  2. BACK-COMPAT surfaces with no real users — OWNER DECISION items
     (e.g. the 8a single-box npz load path: no 8a artifact was ever
     shipped; retiring it is an API-promise call, list it in the
     disposition table, do not decide it).
  3. REDUNDANT certifications across builds (positive-parity oracle
     agreement in operator + schwinger + fast_path; multiple
     RB-vs-brute variants) — merge to the strongest single site per
     contract, pointer comments from the retired sites.
  4. INTERIM-STATE relics (7a interim refusals, pre-fusion internals)
     — verify all were re-targeted or retired; sweep for stragglers.
  DELIVERABLE: a disposition table (keep / retire-with-pointer /
  merge / owner-decision) over every test with per-test justification;
  driver adjudicates; owner rules on API-surface rows. MOTIVATION
  beyond hygiene: pin-heavy suites re-price every future build's
  re-baseline work (measured 2026-07-21: the 8d re-baseline consumed
  a full test-dev session across 6 files); pruning dead pins cuts
  that churn at the source. Do NOT run before the 8d commit lands
  (its witnesses must first exist to become schedulable).
