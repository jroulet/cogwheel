# Professor short-term (F075 fold_exterior_ghost REVIEW verdict, 2026-08-13)

## Verdict: PASS on the fast domain tests for the F075 build.

New test file `cogwheel/tests/test_lensing_fold_ghost_exterior.py`: 17/17 pass
(3.2s). Full `test_lensing_airy_fold.py` 145 pass/7 skip/2 xfail (33s).
`test_lensing_ghost_gate.py` + `test_lensing_channels.py` 34 pass (21s) — the
constant single-sourcing into `geometry` did NOT break channels consumers.

### Q4a fold refusal — CORRECT.
Guards `len(images)!=4 -> refuse` landed at the THREE census-known entry points
I recommended in the Q1 consult: `fold_amplification` (L470), `fold_ppgo_correction`
(L614), `born_carrier_from_partition` (L1608). NOT inside `_merging_fold_pair`
(avoids the _pearcey_cusp first-disjunct flip risk). Physics sound: exterior
positive-parity = 2 real (Morse0 min + Morse1 saddle), no genuine merging pair;
`_merging_fold_pair` returns the FAR pair -> Airy correction spurious. Exterior:
fold None, ppGO==geometric to machine precision, carrier bit-identical to no-fold.
Interior 4-image: guard is a no-op, fold stays active. Teeth prove guard is
load-bearing.

### Q4b ghost gate — CORRECT and refusal-conservative.
Two FREQUENCY-INDEPENDENT gates single-sourced in geometry: Im(tau_c)>=0.4,
min|x_a-x_c|>=0.7. They did NOT adopt the w-dependent floor the handoff floated
(the train/serve-skew tension I flagged in Q2) — good. serve/decline/refuse all
correct; boundary-flip teeth confirm BOTH gates live. GhostAbsentError (interior
-> decline None, interior serve byte-identical) vs GhostDomainError (on-axis
undecayed -> refuse None) correctly separated.

### Q4c sign pin — CORRECT, physically discriminating.
served = geometric_amplification + ghost.kernel*exp(1j w tau_c). Non-conjugated
tau_c with Im>=0.4>0 gives |carrier|=exp(-w Im tau_c) DECAY; conjugate would be
exp(+w Im) BLOW-UP (physically wrong). '+' and non-conj pinned to 1e-12 at low
w=12 (ghost term ~4e-4, resolvable); minus/conj mutants provably fail. Internal-
consistency pin by design (no oracle) — locks sign against refactor.

### Operator-deferred (as expected, per Q4 consult):
The 1e-2 arm bar value-vs-f_schwinger sweep over |y|/rc>=1.15, w in (60,150] is
the EXPENSIVE acceptance REPORT, not a fast test. Fast tests pin
structure/decision/sign only — appropriate. Heavy accuracy validation is the
operator ship gate.
