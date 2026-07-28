# Inspector Short-Term Observations

## 2026-07-29 (review #12) — working tree = advisory + memory only; INS-10-001 still open

Scope: "review all uncommitted changes." Working tree is CLEAN of code:
`git status --porcelain` shows only
  M .claude/tidy_advisory.json
  M .serena/memories/foreman_short_term.md
  M .serena/memories/inspector_short_term.md
No .py, no SPEC, no DATA_CONTRACTS changes. `git diff --stat HEAD` on
SPEC.md and DATA_CONTRACTS.yaml is EMPTY. HEAD is still d0aadf7 (same
commit reviewed in #10/#11). The saddle Born build landed in 31ee133 +
d0aadf7; nothing new to certify this pass. This is the THIRD consecutive
pass (10, 11, 12) with no code delta.

### Re-check of carried finding INS-10-001 (SPEC Born-rung stale) — NOT RESOLVED
Re-read SPEC.md lines 88-112 directly and re-read HEAD code:
- SPEC still: "positive parity, exact exterior fence gamma < 3/4",
  born_lead_carrier = "sqrt(mu_macro)*exp(1j*w*phi_geo)" NO a0/b1, NO
  Morse; above-split = "two-real-image geometric-optics sum plus
  farfield_ghost_term where admitted". "serve slot still unwired" (correct).
- Code (_born.py born_lead_carrier, lines 341-398) at HEAD: docstring +
  body serve BOTH parities; det_a=(1-kappa)^2-gamma^2<0 => morse=(-1j)**1,
  else 1.0. Macro-saddle band served with Morse phase.
Conclusion: INS-10-001 OPEN. SPEC paragraph never updated (no diff touches
SPEC). Librarian doc-sync item; code correct & tested (verified review #10).
Carry forward unchanged.

### No new findings
No code changed => no new code defects. tidy_advisory.json + memories are
housekeeping, not shipped artifacts / not code. DATA_CONTRACTS unchanged &
correct (serve slot UNWIRED per SPEC; census offline).

Verdict: ISSUES (1 carried SPEC doc-sync finding INS-10-001; ZERO code
defect, ZERO new finding). resolved_ids: [].
LESSON (reaffirmed): advisory/memory-only working trees have nothing to
certify, but carried doc-sync findings do NOT auto-close — re-read the
actual SPEC paragraph + the actual code symbol and confirm the divergence
byte-for-byte before deciding resolved vs open.
