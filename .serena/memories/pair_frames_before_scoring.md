# Pair frames before scoring

Migrated from Claude auto-memory `pair-frames-before-scoring` (type: project),
2026-08-13. Referenced from FINDINGS.md F075 as `[[pair-frames-before-scoring]]`.

In cogwheel lensing, an amplification value's meaning depends on the
DEMODULATION FRAME and carrier convention. Comparing objects obtained from
different accessors produces phantom O(1) "errors" that look exactly like real
physics defects.

Two measured instances on 2026-08-13, both fake:
1. demodulated arms scored against `F_op` in a different frame gave 6e-2 where
   the correct pairing gives 1.49e-4;
2. fold/ladder values scored against `ChangRefsdalChannels.evaluate()
   .exact_total` gave 32%-130% "errors" at well-resolved exterior configs
   where geometric optics should be good to ~1%.

**Why:** the frames differ by carrier factors and by the demodulation origin
(the min real-image `geometry.delay`, NOT `part.delays.min()`), which cancel
only when both sides share the convention.

**The pairing gate (standing practice — run it before ANY accuracy claim):**
score `operator.geometric_amplification` against the oracle at a
KNOWN-RESOLVED config (e.g. gamma=0.5, y=(1.8, 0.9), w=40) and require
< 1e-2. No claim from a harness that has not passed this gate.

**F069-safe oracle recipes:**
- Positive parity: the mass-sheet / eigenframe reconstruction
  (`operator._mass_sheet_map`) fed to `_schwinger.f_schwinger`. At
  kappa=0, beta=0 it collapses to `f_schwinger(w, source, gamma)` directly.
- Saddle parity: `operator._saddle_mass_sheet_map` + `f_schwinger`
  (validated to 0.0e0, including kappa=0.2).
- NEVER `F_op` above w=60 — it becomes a self-oracle there.
- NEVER `channels.evaluate().exact_total` as a reference for arm outputs
  without first proving the t_min pairing; it lives in the min-subtracted
  frame.
- Keep oracle points at w <= 60 to stay on the exact DD path (>60 is mpmath,
  >150 hard-refuses).

Related: `mem:geometric sanity` (picture the astroid and ask whether the
number is physically plausible before reporting it) and the standing rule that
oracles must call shipping code, never a re-transcribed formula.
