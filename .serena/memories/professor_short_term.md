# Professor Short-Term — Session 2026-07-18 (Build 3f SACR-C inference review)

## Verdict: PASS
Reviewed the SACR-C beat-free envelope decomposition build (working tree,
uncommitted; env cogwheel-newlal, py3.10). All fast domain suites green:
- channels + gauge: 62 passed
- operator: 22 passed (MacroMagnificationLimit + mpmath oracle)
- likelihood: 29 passed + 1 xfail
- fast_path + geometry + batched_operator + dd: 80 passed

## Gate mapping (all encoded in passing assertions, match report numbers)
- GATE1 recon identity: EnvelopeReconstructionGate::test_node_reconstruction_
  identity_is_machine_precise + ScaleAware::test_exact_total_matches_oracle;
  flat gate 5e-15, report ~2e-16.
- GATE2 greedy / GATE3 LOO node count: test_node_count_under_ceiling +
  test_node_count_is_config_independent; greedy N<=26 (worst 21), LOO N<=48.
- GATE4 |S_a H_a|<=2 at fold/cusp crossings (measured ~1.21):
  gauge::test_switched_saddle_kernels_are_bounded_at_crossings, with
  anti-vacuity (unswitched genuinely singular) + self-falsification
  (forced S_a=1 -> ~1e8) + AST fixture-independence guards (no channels.py).
- GATE5 deep-band macro limit: DeepBandMacroLimit (flat plateau + independent
  closed form 1/sqrt((1-k)^2-g^2), <1e-6 rel).
- F001 carrier phase: gauge::test_range_reduced_carriers_match_mpmath_at_large
  _phase, gate 1e-10, measured ~5e-13, independent mpmath oracle (AST-guarded).
- Regression all green: RB-vs-brute (RB_ATOL=1.5), near-cusp pin, zero-noise
  floors, MacroSector contrast, MacroSaddleRejection.
- Self-falsification tests present & passing => gates non-vacuous.

## One expected non-green + one doc concern (NOT physics failures)
- test_warm_lnlike_ms_ceiling_projected is XFAIL by design: 18ms is a
  machine-dependent PROJECTED bound; engine 1F1 ladder (out of likelihood
  scope) dominates ~89%, warm best-of-5 ~29ms here. Load-bearing gate
  test_public_entry_speedup PASSES (~47x lnlike vs bruteforce). Matches the
  deferred 10ms/envelope-surrogate lever note.
- Open finding INS-6-001 is SPEC.md doc-sync (line-55 fast-path sentence still
  describes removed _DEFAULT_KERNEL_NODES machinery) — librarian's job, not a
  correctness issue.
