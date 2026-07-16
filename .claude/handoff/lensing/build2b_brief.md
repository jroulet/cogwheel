# Build 2b: make the lensed relative-binning likelihood pass its crown gate

## Context
Build 2 delivered `cogwheel/lensing/waveform.py` (green suite) and
`cogwheel/lensing/likelihood.py` (`LensedRelativeBinningLikelihood`,
subclassing `BaseLinearFree`), plus the F005 closure in `operator.py` (now
independently reviewed and green: 21 tests + 69 subtests). The remaining gap is
the crown gate: `cogwheel/tests/test_lensing_likelihood.py` fails 5 of 13.
All uncommitted work is in the working tree. Where anything conflicts with
this brief, THIS BRIEF WINS.

## Measured failures — inputs, not questions (two runs, second on a quiet box)
1. `BruteForceAgreementTestCase`, config `near-cusp`: |RB lnL - brute lnL| =
   6.43e8 vs tolerance 1.5, and the value is BIT-STABLE across runs — a real,
   deterministic defect. Prime suspects, in order: (a) fiducial-vs-candidate
   image-count/label mismatch near the caustic (fiducial 4-image summaries
   evaluated at a 2-image candidate or vice versa); (b) K_a interpolation
   across a topology change; (c) refusal/branch asymmetry between the RB path
   and the brute-force path (both consume the same generator — verify they
   also see the same amplification decomposition, not one total and one
   per-image). Diagnose before fixing; state the mechanism in the change
   report.
2. NONDETERMINISM in the suite: config `two-image` passed run 1 and failed
   run 2 at |dlnL| = 10.9; the unlensed-floor checks read 0.106 (run 1) then
   0.33 / 0.28 (run 2). Deterministic tests are a repo convention (fixed
   seeds). Find the unseeded source (EventData noise draw?), seed it, and THEN
   judge what residual is real. The normalization floor failing at a stable
   value after seeding is a genuine bug (the F->1 cross-check exists to catch
   a normalization error the self-referential comparison cannot).
3. `ContractionTimingTestCase`: contraction 1.47e-3 s vs coarse-waveform-call
   budget 6.4e-5 s — 23x over, ON A QUIET BOX. First validate the gate itself
   (is the 64us baseline the right "coarse-node waveform call" the design's
   subdominance requirement references, or a mis-measured proxy?); if the gate
   is right, the contraction needs the designed additive shape (modes first at
   the delta_t nodes, THEN images via envelope interpolation at the n_img^2
   pair delays) — profile before optimizing and report where the time goes.

## Mission — production deliverables
1. Fix `cogwheel/lensing/likelihood.py` (and, only if the mechanism genuinely
   lives there, `channels.py` consumption) so the crown gate passes: RB lnL
   agrees with brute-force through the SAME LensedWaveformGenerator within the
   suite's stated tolerances across ALL its configs, including near-cusp.
2. Make the contraction meet its subdominance budget or, if the gate is
   mis-specified, correct the gate WITH a written justification of what the
   design's "subdominant to the coarse-node waveform call" actually requires.
3. Closeout deltas only if surfaces changed (FINDINGS entry for the near-cusp
   mechanism; changelog fragment for the fix).

## Tests — put ALL of these in `domain_test_descriptions`
Suite: `test_lensing_likelihood.py` (amend in place; it is the crown gate).
- Seed every stochastic input (noise draws, parameter jitter) — the suite must
  be bit-reproducible run to run; add a determinism check (two in-process
  evaluations of lnL at the same point must be identical).
- Keep every existing gate at its tolerance; do not widen anything to pass.
- Add a regression pinning the near-cusp mechanism once diagnosed (the exact
  failing config, asserted at the corrected value).
- Re-validate the timing gate per Mission item 2 (budget measured in-suite,
  same process, warm caches; document the baseline definition).

## Settled facts
- operator.py refusal semantics are FINAL as reviewed (truncation cut at
  1e-10 tail + round-off guard 2e-9): do not relitigate; likelihood code must
  HANDLE refusals (CancellationError) explicitly if any config can reach them
  — never let one path refuse while the comparison path returns a value.
- The suite's brute force is the oracle: it evaluates the SAME generator on
  the full grid. Any fix must move the RB path toward it, not the reverse —
  except the two documented test defects (seeding; possibly the timing
  baseline), which must be argued, not assumed.
- waveform.py and its suite are green; do not modify them for this build.
