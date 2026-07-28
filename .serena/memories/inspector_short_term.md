# Inspector Short-Term Observations

## 2026-07-28 (re-review #9) — Born carrier + band-split (F025), uncommitted tree

Scope: uncommitted working tree vs HEAD. SAME code tree as passes 1-8
(diffstat identical: _born.py +313, channels.py +175, surrogate_census.py +66,
likelihood.py 20, __init__.py 4, tests grown: test_lensing_born.py +1060,
test_lensing_surrogate_census.py +25). SPEC.md NOT in git status (clean/unmodified).

### Re-verified this pass (own re-review, did NOT trust priors)
- Suite: pytest test_lensing_born.py test_lensing_surrogate_census.py -q
  => 37 passed, 13 skipped, 101.44s. GREEN (matches passes 3-8).
- Census six-way via import: sc._FALLTHROUGH_CATEGORIES =
  ('gamma-guard','dropped-sliver','born','cusp-window','refusal-ball','out-of-box').
  GAMMA_FENCE=0.75 in _born.py.
- born_lead_carrier body (lines 34-35): unpacks sqrt_mu,phi_geo,_,_,_ =
  _born_factors(...); returns sqrt_mu*cmath.exp(1j*w*phi_geo). Uses NEITHER
  a0 nor b1 (F009/F025). The 'a0'/'b1' substring hits are docstring-only.
- _born_factors callers (grep): line 274 (5-tuple, born_amplification),
  313 (_,_,_ placeholders, lead carrier), 487 (5-tuple, DIAGNOSTIC per comment).
  NO stale 4-tuple caller.
- likelihood.py slot (lines 1654-1666): Fact-4/8h-c1 F025 status comment;
  slot returns None (dormant/unwired, awaiting TRAIN_TIER residual chart). Correct.
- channels.born_carrier_from_partition (line 1294) defaults lead_carrier=
  _born.born_lead_carrier (line 1405), needs two real images (guard line 1422).

### FINDINGS (both PERSIST — SPEC.md unmodified; Librarian doc-sync work)
Confirmed by grep on SPEC.md this pass:
- INS-9-001 (trivial doc-sync; carries INS-1..8-001): SPEC.md line 54 (big
  table cell) still "5-way MECE fall-through breakdown (gamma-guard /
  cusp-window / refusal-ball / out-of-box / dropped-sliver)"; code is six-way
  with 'born'.
- INS-9-002 (trivial doc-sync; carries INS-1..8-002): SPEC.md lines 89-94
  "Born rung (DORMANT)" still: "placeholder", "disagrees with operator.F_op
  by up to ~13%"; superseded by F025 lead-only carrier + band split +
  None-returning dormant slot.

### resolved_ids this pass: NONE (both persist; SPEC not yet synced).
Verdict: ISSUES (2 persisting SPEC doc-sync only; ZERO code defect).
DATA_CONTRACTS unchanged: Born slot unwired, census offline; no shipped
artifact schema changed.
LESSON: 9th byte-identical re-review; still re-ran suite + import probes +
traced callers + inspected the served carrier body. Rule holds.
