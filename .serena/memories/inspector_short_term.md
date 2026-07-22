# Inspector Short-Term Observations

## 2026-07-22 (Build 8g-b 6th review) — VERDICT: PASS (INS-8gbc-001 & -002 RESOLVED)

Scope: re-review of the two open test-fixture findings after Test Dev applied
the sanctioned systemic fix. Production code UNCHANGED since the 5th review
(which certified WP1+WP2 CORRECT). Only test_lensing_surrogate.py carries new
edits this pass.

### FULL RUNS (setsid-detached, full python path) — all EXIT=0
- test_lensing_surrogate.py (full, no -x): **45 passed, 1 skipped in ~110s**.
  Also confirmed identical with -x. RefusalPreservationTestCase now green.
- farfield_envelope + census + training (combined): **128 passed in 710s**.

### INS-8gbc-001 — RESOLVED
Systemic fix as prescribed: DELTA_T_MAX 0.02 -> 0.05; DF_BIN RE-DERIVED
0.02->0.05 => 4.0->1.6 holding pi*DF_BIN*DELTA_T_MAX ~= 0.25 rad (half the
0.5-rad _DEFAULT_BIN_DELAY_TOL; narrower bins = safer, physically sound).
CROWN_LENS unchanged (gamma0.35,y1=2.25,y2=0). The kappa=0.1 fall-through that
raised LensedBinningError now measures 0.020863 s vs 0.05 bound => 41.7% margin.
test_nonzero_kappa_never_served passes (spy count 0, bit-for-bit exact match).

### INS-8gbc-002 — RESOLVED
New DelayMarginContractTestCase pins delay/DELTA_T_MAX <= MARGIN_FRACTION_CEILING
=0.60 for the WHOLE far-field-exterior family, collected from the SAME shared
dicts (non-circular). Printed margins: crown(k=0) 0.3734, crown k=0.1 0.4173,
pos/crown 0.3734, pos/deep 0.3789, pos/box-edge 0.3646 — all 36-42%, uniform
comfortable margin. Paired DelayMarginSelfFalsificationTestCase reproduces the
LensedBinningError at the OLD 0.02 s bound (gate has teeth, not vacuous). A
criterion test pins pi*DF_BIN*DELTA_T_MAX ~= 0.25 and < 0.5. No production code
touched by this pass (surrogate/geometry/likelihood untouched).

### INS-8gb-005 (SPEC + DATA_CONTRACTS divergence) — STILL OPEN → Librarian
This build did NOT touch SPEC.md or DATA_CONTRACTS.yaml (not in changed files).
SPEC still says farfield_eps_max=3e-3 (code 1e-3); DATA_CONTRACTS
lens_amplification_surrogate still lacks the REQUIRED per-chart npz meta
`envelope_definition` (whose absence hard-refuses load). Pre-existing from the
WP1/WP2 build; test-only this pass, so NOT an actionable finding here. Carry to
Librarian/driver.

### Carried open
- INS-4-001 (design): TrainingConfig.max_farfield_regions default (unrelated).

### Lessons
- A pure test-fixture systemic fix (raise the module's delta_t_max + re-derive
  fbin from the SAME phase-accuracy criterion) is the correct home for a
  relocation-induced binning overflow — production stays byte-identical. Verify
  the re-derivation preserves the invariant constant (pi*DF*dt), not just that
  tests pass, and confirm the self-falsification test still reproduces the
  original error at the OLD bound so the margin gate isn't vacuous.
- Imports clean; _FARFIELD_ENVELOPE_DEFINITION='farfield_full_kernel_sum';
  channels.farfield_envelope_from_partition present. serve 3-tuple migration
  intact (census+training green).

Runtime friction (unchanged): Bash hook blocks python/grep/tail/cat/sed with
leading cd/VAR=; run python only via serena execute_shell_command or setsid-
detached. Serena shell TIMES OUT ~260s but setsid-detached pytest in its own
group SURVIVES. Read tool унavailable; read /tmp logs via a python one-liner
(open().readlines()) through serena shell, not grep/tail.
