# Build 7a — Runtime index-theorem guard + cross-parity strong-shear Schwinger dispatch

## Mission

Two engine-layer hardenings that are preconditions for the saddle
channel layer (Build 7b):

1. **Runtime index-theorem guard.** A pre-existing, parity-agnostic
   dead zone in `find_images_quartic` silently drops a symmetric
   near-degenerate image pair for sources ~1e-10..1e-9 (relative
   angle; up to 1e-8 on positive parity) off a macro-matrix eigenaxis
   inside the 4-image region (FINDINGS F012). Below the wave ceiling
   this is harmless (the wave branch never consumes images), but above
   it the geometric branch returns an O(1)-wrong FINITE value — a
   certify-or-refuse violation. Add a runtime census check in every
   image-consuming path, enforcing the parity-dependent index-theorem
   invariant. The saddle-side invariant is CERTIFIED (Build 6): index
   sum −2, image-count/Morse multisets {1,1} (2-image) and {0,1,1,1}
   (4-image) — pinned in `test_lensing_saddle_geometry.py`'s census
   tests. The positive-parity invariant must be READ OFF the existing
   certified quartic path and the research note's census section — do
   not guess it; verify it on the deterministic source sets already in
   the suite before enforcing it. On violation raise
   `geometry.LensDomainError` with a named message (image census
   defect), never return images.

2. **Cross-parity strong-shear Schwinger dispatch.** The exact 1D
   Schwinger representation in `_schwinger.py` is valid for BOTH
   parities (design authority:
   `.claude/handoff/lensing/negative_parity_research.md`, §6.1 — the
   representation never specialized to a parity sign). The legacy
   positive-parity wave path (`_hyp1f1` kernel) hits its
   `operator.CancellationError` band near `gamma ~ 0.5` — VERIFIED
   pin: `cogwheel/lensing/prior.py:164-169` documents the sampling
   range `gamma in [0, 0.45]` as headroom against exactly that band,
   with the residual approach caught by the posterior-boundary refusal
   net. Route positive-parity evaluations that the legacy path refuses
   via `CancellationError` (and only those) to `f_schwinger` instead,
   behind the existing dispatch seam in `operator.F_op`/`F_op_grid`,
   subject to the Schwinger ceiling `w <= 60` (above it the refusal
   stands).
   Where BOTH paths certify, the legacy path stays authoritative and
   BIT-FROZEN — the dispatch only converts refusals into certified
   answers, it never changes an answer the legacy path already gives.

## In scope

- `cogwheel/lensing/chang_refsdal/geometry.py` — the index-theorem
  guard in image-consuming paths (and only there).
- `cogwheel/lensing/chang_refsdal/operator.py` — the strong-shear
  fallback arm in the parity dispatch.
- Tests beside the existing suites in `cogwheel/tests/`
  (`test_lensing_saddle_geometry.py`, `test_lensing_schwinger.py`,
  `test_lensing_geometry.py` may be extended; new module fine).
- Flipping `NearAxialQuarticDefectTestCase` from `@expectedFailure`
  (silent pair drop) to a positive test asserting the NAMED refusal.

## Out of scope — hard fences

- NO changes to `_schwinger.py` internals (it is certified; consume
  `f_schwinger` as-is, including catching
  `SchwingerCertificationError`).
- NO channel/likelihood/prior/waveform-layer work (that is Build 7b;
  the interim saddle guards in `channels.evaluate` and
  `LensedWaveformGenerator.__init__` stay in place).
- NO surrogate, homogenization, Airy-patch, or v-plane work (Build 8+).
- NO relaxation of the sampling layer's gamma bound in this build
  (that is a prior-layer change, Build 7b) — this build only makes the
  engine ready for it.
- The positive-parity path where it already certifies is BIT-FROZEN:
  any test that pins legacy outputs must keep passing byte-identically.

## Measured facts (pre-answered — do not re-derive)

- Schwinger certified vs an independent mpmath oracle: 9.1e-14 (w=20)
  .. 1.6e-11 (w=59.9); ceiling `w <= 60`
  (`W_CEILING_SCHWINGER = 60`); warm cost 30–125 ms/point, linear
  in w.
- Dead-zone geometry (F012): source at relative angle 1e-10..1e-9
  (pos. parity: up to 1e-8) off an eigenaxis, inside the 4-image
  (resp. saddle 4-image) region; the quartic path drops the mirror
  pair; `NearAxialQuarticDefectTestCase` in
  `test_lensing_saddle_geometry.py` reproduces it deterministically.
- Census invariants certified in Build 6 (saddle side, 200+ sources):
  index sum −2; multisets {1,1} / {0,1,1,1}. The positive-parity
  counterpart is NOT pre-answered here — read it off the certified
  code/tests as described in the Mission.
- Morse phase per image: e^{−i π n_j / 2}; parity is MST-invariant.
- `f_schwinger(w, y_eig, gamma_prime)` operates in the mass-sheet-
  reduced eigenframe; the operator saddle arm shows the exact
  reduce → rotate → reconstruct pattern
  (F = (1/λ)exp[iw(lnλ/2 − κ|y_s|²/2)] · schwinger_F) — the
  positive-parity fallback arm must use the same reduction with the
  positive-parity λ convention.
- FINDINGS F011: any float64 quantity identical across the paired
  N/2N rules and entering the result multiplicatively is a candidate
  silent fabricator — the reconstruct prefactor added by the fallback
  arm is OUTSIDE the certificate; it must be simple enough to bound by
  inspection (single exp/mul in float64 is fine; document why).

## Acceptance (build-level)

1. **Guard falsification (fast)**: the F012 reproducer now raises
   `LensDomainError` naming a census defect, on BOTH parities; a
   py_func-chain falsification (F010 idiom) proves the guard can go
   red (feed a doctored image set with a dropped pair and observe the
   raise).
2. **Guard non-interference (fast)**: the guard passes silently on the
   certified census sweep configurations (reuse the small
   deterministic source sets from `test_lensing_saddle_geometry.py`,
   not new bulk sweeps).
3. **Dispatch oracle accuracy (fast)**: on a small grid of
   strong-shear positive-parity points that the LEGACY path refuses
   (pick w <= 60), the fallback arm agrees with the existing
   AST-guarded mpmath dev-oracle pattern from
   `test_lensing_schwinger.py` at 1e-10 or better; above w=60 the
   named refusal survives.
4. **Bit-freeze (fast)**: on points where the legacy path certifies,
   outputs are byte-identical to pre-build HEAD (pin a handful of
   values in the test, don't diff trees).
5. All existing lensing tests stay green. FULL suite is a POST-BUILD
   driver step ("full suite green, driver-verified post-build") — do
   NOT write hour-scale test specs.

## Constraints

- numba compatibility for anything on the hot path; the guard runs per
  image-solve, so it must be cheap (integer census arithmetic — no new
  allocations in the jitted path).
- Certified-or-named-refusal contract (F005) everywhere: no code path
  may return a finite number it cannot certify.
- Spec/TODO workflow applies (behavior change in `cogwheel/`): todo
  fragment at plan time, completion fragment + FINDINGS/SPEC updates
  at close (F012 gets a "GUARDED as of Build 7a" addendum).
- Tests are stdlib `unittest` in `cogwheel/tests/`, fast/synthetic
  only.
