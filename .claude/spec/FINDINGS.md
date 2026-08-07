# Findings

Empirical discoveries, numerical-accuracy notes, and non-obvious gotchas
uncovered while working on cogwheel. Each finding gets an ID (`F001`, `F002`, ...),
a date, and a short writeup. Mark superseded findings rather than deleting them.

Use this for things like: tolerance levels at which relative binning diverges
from the exact likelihood, sampler convergence quirks, ASD-drift sensitivities,
waveform phase-convention pitfalls, numba compatibility traps.

---

## F001 — The two-channel cancellation law (2026-07-16)

The Chang–Refsdal amplification loses precision to alternating-series
cancellation through TWO channels. They are INDEPENDENT: they live at
SEPARATE code sites and do NOT compound into a single summed exponent.

- `L_1F1 = w * |y'|` — the confluent-hypergeometric (1F1) kernel series in
  `cogwheel/lensing/chang_refsdal/_hyp1f1.py`. Its partial terms reach
  `e**(w*|y'|)` while the sum is O(1). Double-double arithmetic is required
  HERE, and only here: Kahan summation does not help, because its error
  bound also carries `sum|term_i|`. dd holds the 1e-10 target out to
  `w*|y'| ~ 50` and degrades to ~1e-6 at the ceiling `w*|y'| = 60`.
- `L_op = w * gamma'/2` — the operator power series in `operator.py`. This
  channel is NOT rescued with extended precision; instead `F_op` MEASURES
  its own cancellation ratio `max_partial_term / |total|` and REFUSES
  (raising `CancellationError`) once it exceeds ~1e13. That runtime refusal
  is the operational form of the law: past ~13 lost digits the double-double
  substrate no longer protects the sum, so returning a plausible-but-wrong
  amplification is the failure mode being avoided.

Because the channels are independent, treating them as one summed exponent
`w*(|y'| + gamma'/2)` (as an earlier `_dd.py` docstring did) overstates the
precision demand and misplaces the dd requirement.

## F002 — The oracle-tautology trap in the lens-engine tests (2026-07-16)

A test fixture built by the very code it is meant to judge cannot fail, no
matter how broken that code is. Two concrete instances shaped the lens-engine
test design:

- The fold/cusp crossing-scenario builders that the label-continuity test
  judges `channels.py` against are constructed from `geometry`, `operator`,
  and `_gauge` ONLY. They must never import, call, or derive a value from
  `channels.py`; otherwise the ground truth is the tracker's own output and
  the test is vacuous. This is enforced with an AST import guard in the idiom
  the committed `test_lensing_gauge.py` already uses.
- A mass-sheet identity checked by comparing `F_op` against its own kappa-
  rescaling path is equally vacuous — the code agrees with itself by
  construction. Such identities are asserted on OBSERVABLES (the delay
  differences `Delta tau` and flux ratios `|K_a/K_c|`, which are exactly
  kappa-invariant) or gated against an INDEPENDENT mpmath computation, never
  against the code's own rescaling path.

## F003 — mpmath is an undeclared test dependency (2026-07-16)

The committed lens-engine test suites import `mpmath` as a high-precision
oracle, but `mpmath` is declared nowhere in `pyproject.toml` (no runtime
dependency, no test/dev extra). It is present only because it happens to be
installed in the `cogwheel_310` environment; a clean install would fail to
collect these tests. Recorded here per the Build 1b brief as an observation
to be resolved deliberately (e.g. a test extra), NOT fixed in this build.

## F004 — boundary-domain tests need float64-exact boundary points (2026-07-16)

`macro_matrix` rejects `(kappa, gamma)` iff `1 - kappa <= |gamma|` (strict
positive-parity `1 - kappa > |gamma|` required). A test intending to hit the
EQUALITY boundary must choose values where `1 - kappa == |gamma|` holds
bit-for-bit in float64. `(kappa=0.7, gamma=0.3)` does NOT: `1 - 0.7` evaluates
to `0.30000000000000004`, a hair above `0.3`, so that point is genuinely just
inside the domain and correctly does not raise — a test asserting it must raise
fails against correct code. Use powers-of-two endpoints (`0.5/0.5`, `0.75/0.25`)
where `1 - kappa` equals `|gamma|` exactly. Caught in the first delivered
lens-engine test suite; the code was right, the test's boundary point was not.

## F005 — wave-branch contraction: silent-nan closed, gap NARROWED to a named refusal (2026-07-16, NARROWED)

Original defect (OPEN): the float64 operator contraction in `F_op`
(`z_powers @ (table*radial) @ zbar_powers`, complex128 — deliberately NOT
double-double, per the two-channel error model F001) drifted below the 1e-10
target for cancellation exponents `L = w*sqrt(s)` above roughly 30 at high `w`,
and near `L ~ 40` the intermediate products overflowed to a SILENT `nan` — no
`CancellationError`, no named refusal — violating the module's "named error,
never a silently wrong number" contract inside the nominally certified
`L <= 48` band. `estimated_relative_tail` does NOT bound this error (it tracks
the kernel series tail, not contraction round-off/overflow).

Resolution shipped (WP1). `F_op` now does two things, WITHOUT adding
double-double to the contraction (F001's two-channel model is preserved — dd
stays in the 1F1 kernel only):

1. Overflow-safe contraction. Before the matmuls it factors the peak radial
   magnitude `max|derivs|` out as an EXACT power of two (`np.frexp` picks the
   exponent, `np.ldexp` rescales with no rounding). Every scaled entry is then
   O(1), so no intermediate overflows; the whole summation runs in units of
   `2**scale_exp` and the total is scaled back exactly. For the previously
   certified region this is bit-for-bit identical (the power-of-two factor
   commutes through multiply/add without rounding, and `scale_exp` is small
   enough there that no meaningful entry underflows), so the `L <= 25`
   oracle behaviour is unchanged.
2. Named certification refusal, via TWO independent cuts for two independent
   error sources (revised 2026-07-16 after direct 70-dps-oracle calibration;
   see below). (a) TRUNCATION cut: `estimated_relative_tail`
   (max of the operator-series last-term ratio and the kernel per-order tail)
   `> _CONTRACTION_TARGET = 1e-10` — binding at small `max_order`, where the
   shear series has not converged (this closed a pre-existing SEPARATE silent
   hole: `large-shear` w=40 at the default `max_order=42` returned 1.26e-4 with
   `converged=False` and no refusal). (b) CONTRACTION round-off cut: the
   first-order float64 round-off `eps * (sum|term| / |total|)`
   (`positive_total / |total|`, an all-positive companion contraction)
   `> _CONTRACTION_GUARD = 2e-9` — binding once the series has converged and the
   float64 derivative-ladder cancellation dominates near `L ~ 45`, the regime
   the truncation cut goes BLIND to (its tail collapses to ~1e-14). Both name
   `w, y, gamma, kappa`. A non-finite-total backstop (and a non-finite
   reconstructed-value backstop) catch any residual overflow, since
   `nan > threshold` is False and would otherwise slip past the ratio gates. The
   gamma-channel refusal (`max_partial_term / |total| > _CANCELLATION_REFUSAL =
   1e13`, F001) is unchanged and still fires first.

Net status — NARROWED, not RESOLVED. WP1's overflow-safe rescaling did MORE
than remove the nan: direct 70-dps-oracle measurement (2026-07-16) shows the
returned-and-accurate ceiling is now near `L ~ 45` (true relative error stays
below 1e-10 out to `L ~ 44` on the `y=(0.9,0)`, gamma=0.2 sweep, crossing 1e-10
near `L ~ 45-46`), NOT the pre-WP1 `L ~ 30`. The accuracy of the wave branch
WAS extended by the rescaling (no dd added; F001 preserved). What the refusal
adds on top is the CONTRACT: the residual band `L in [~45, 48]` (below the
geometric branch's `L > 48` onset) now exits through a named
`CancellationError` — never a silent `nan`, never a finite-but-wrong number.

Calibration caveat (why the guard is 2e-9, not the 1e-10 target). The
round-off bound `eps * (sum|term| / |total|)` is a WORST-CASE upper bound, loose
by ~20-30x and NOT a rigorous 1e-10 proof: across shear it can INVERT (an
accurate `y=(0.9,0)` L=45 config reads bound 5e-9 while an accurate
`large-shear` w=40 config, true 2.7e-11, reads 7.4e-10; a low-shear gamma=0.1
config at `L ~ 50` is inaccurate at 3e-10 yet reads bound 4e-13 — that last is
kernel degradation past `L ~ 48`, out of the wave band and handed to the
geometric branch). So the round-off cut is a MEASURED COARSE NET, not a proof;
`_CONTRACTION_GUARD = 2e-9` is pinned to the clean 2.7x gap between the largest
must-return bound (CERT L=43.5 at 1.12e-9; large-shear at 7.4e-10) and the
smallest must-refuse bound (CERT L=45 at ~3-5e-9) on the tested configs. An
interim driver value of 1e-8 was too loose and let `L ~ 45-48` leak as
finite-but-wrong returns (rel err ~1.3-1.5e-10) — retightened to 2e-9.

Consequence for callers and tests. `channels.py` reaches the wave branch only
when `select_branch` returns `'wave'` (i.e. NOT both resolved and `L > 48`), so
an unresolved config with `L in [~45, 48]` now propagates a named
`CancellationError` instead of a silent nan — the intended "certified or
named-refusal everywhere" contract the Build-2 likelihood relies on.
`ContractionCertificationTestCase` asserts the certify-XOR-refuse contract
across `L in [24, 48]`; `GeometricOpticsSlopeTestCase` keeps `L <= 24.3` (w to
27 at `|y'| = 0.9`) so it exercises only certified returns. Do NOT reopen the
gap by widening tolerances.

## F006 — near-cusp (h|h) blow-up: edge-secant kernel coefficients alias the caustic-sharpened amplification (2026-07-16, SUPERSEDED by F008 2026-07-17)

> **SUPERSEDED — mechanism attribution sign-disproven (F008, 2026-07-17).**
> History preserved below; do NOT act on the fix rationale in this entry.
>
> What F006 got RIGHT: densely sub-sampling each bin and reducing the kernel
> by per-bin least squares (raising `kernel_subsamples` 2 → 8) changed the
> physics not at all once the real cause was fixed — nsub=2 and nsub=8 agree
> to `< 1e-4`. That null result was the correct diagnostic signal that the
> defect lay ELSEWHERE than the contraction algebra, the frequency moments, or
> the `F→1` normalization (all of which F006 correctly cleared and are still
> correct). The symptom (`|RB lnL - brute lnL| = 6.43e8` on `near-cusp`,
> bit-stable, deterministic) is also accurately recorded.
>
> What F006 got WRONG: the attribution to an edge-secant slope-squaring
> aliasing failure. It is sign-disproven — the spurious `(h|h)` drove the norm
> term NEGATIVE-huge (RB − brute = **+6.43e8**, i.e. brute ≫ RB after the sign
> in `lnL = (d|h) - (h|h)/2`), and squaring a real per-bin slope `k1` can only
> ADD a positive quantity to `(h|h)`; it cannot produce a negative-huge
> excursion. The dense sub-sampling that F006 shipped was therefore
> COMPENSATING for the true bug (an unbounded upstream kernel) at 8× the engine
> cost, not correcting an aliasing artifact of the reduction. The real cause —
> the `_channel_switch` real-only neighbourhood bug versus the paper's
> Eq. (delay-separation) — is recorded in **F008**, and Build 2c reverts the
> sub-sampling default to 2.

Symptom (Build 2b crown gate): `LensedRelativeBinningLikelihood` disagreed
with its brute-force oracle by `|RB lnL - brute lnL| = 6.43e8` (tolerance 1.5)
on the `near-cusp` config, BIT-STABLE across runs — a deterministic defect, not
sampling noise. The RB `(d|h)`/`(h|h)` summaries and their mode→image
contraction algebra (`_data_term`/`_norm_term`) were verified CORRECT term by
term against the truncated linear-kernel × linear-ratio × linear-phase model;
the fault was upstream, in how the per-bin amplification kernel `K_a(f)` was
reduced to coefficients.

Confirmed mechanism. Near a cusp the merged-image regime is *unresolved*: in
`channels.exact_transition_channels` the smootherstep switch
`smootherstep(w*delay_sep, 0.5, 4)` goes to 0, so each channel kernel collapses
to the artificial single-image split `K_a = alpha_a * exp(-i w tau_a) * F(w)`,
which carries the FULL amplification oscillation `F(w)` — a rapidly varying,
caustic-sharpened function of `f` within one relative-binning bin. The old hot
path (`_edge_linear_coefficients`) built the per-bin `(k0, k1)` from the TWO bin
EDGES only (a secant). When the two edges happen to straddle a phase alias of
that oscillation, the secant midpoint value `k0` collapses toward zero while the
secant slope `k1` blows up; `_norm_term` then SQUARES `k1`, manufacturing the
~6.43e8 spurious `(h|h)`. This is an aliasing/undersampling failure of the
edge-secant reduction, NOT an error in the contraction, the moments, the
delay-phase model, or the normalization.

Fix (WP1, `cogwheel/lensing/likelihood.py`). New hot-path method
`_amplification_coefficients` evaluates `ChangRefsdalChannels` on a per-bin
sub-sample grid (`kernel_subsamples = 8` interior midpoints per bin, strictly
increasing and positive in `w`) and reduces each channel kernel to `(k0, k1)`
by a per-bin least-squares line — offsets symmetric about `f_center`, so
`k0 = mean` and `k1 = <offset, K>/sum(offset^2)`. Densely sampling within the
bin resolves the caustic-sharpened `F(w)` that the edge secant aliased, so the
fitted slope is bounded and the squared-slope blow-up disappears. The
contraction, frequency moments, ratio path, image-delay guard
(`LensedBinningError`), and all three design decisions are UNCHANGED — only the
kernel-coefficient reduction improved. `_edge_linear_coefficients` is
retained (the ratio path uses it); `_amplification_at_bins` was removed as
dead code (INS-3-002: nothing in the hot path, `lnlike_bruteforce`, or the
test suite referenced it). Engine refusals stay symmetric across the RB and brute-force
paths: `geometry.LensDomainError` (macro saddle) and `operator.CancellationError`
(uncertifiable contraction) propagate unswallowed on both.

F→1 normalization: AUDITED, NO code change. The unlensed-limit floor readings
(0.10–0.33 across runs in the failing report) were traced to an unseeded
`EventData` noise draw (nondeterminism, seeded in the test suite), NOT a
normalization bug. The moment prefactor `4*df`, blued strain, `wht_filter**2`,
and `asd_drift**-2` match `CBCLikelihood._compute_d_h`/`_h_h` and
`RelativeBinningLikelihood`; at `F→1` the `p=0` moments sum over bins to the
exact integral, so `(d|h)`/`(h|h)` are exact to ~1e-8. Because WP1 changed no
normalization code, this finding records the audit outcome only — the F005
overflow-safe contraction and its refusal contract are untouched.

> Correction (F007, 2026-07-16): the "exact to ~1e-8" bound above holds only
> for the `p=0` moment with an exactly-constant per-bin ratio and kernel and
> matched fiducial/candidate frequency sets. The full `F→1` evaluation path is
> NOT exact to 1e-8: `_set_summary` builds `_h0_edges` with precession forced on
> and `_stall_ringdown` applied, while `_candidate_bin_ratios` builds the
> candidate `h_edges` with neither, so at `r ~ 1` the ratio is not identically
> `1` in the in-band ringdown of the fixture and a template-construction
> asymmetry `delta-h/h ~ 1e-3` survives. See F007 for the mechanism and the
> zero-noise anchor that gates it test-side.

## F007 — the timing gate was mis-specified, and the F→1 floor is a template-construction asymmetry, not a normalization bug (2026-07-16)

Two Build-2b diagnoses that are NOT defects in the shipped
`LensedRelativeBinningLikelihood` math, recorded so neither is re-opened as a
correctness bug. Both concern `cogwheel/lensing/likelihood.py`; cited by symbol
name. F006 (near-cusp secant-aliasing) is the separate, real correctness fix and
is unchanged by this entry.

### Timing gate — the baseline was RB's co-cost, not its competitor

`ContractionTimingTestCase` originally asserted `t_contract < t_coarse_waveform`,
where `t_coarse_waveform` is the cost of the coarse `get_strain_at_detectors`
call on the relative-binning `fbin` grid. That baseline is WRONG: the coarse
strain call is a per-eval CO-COST that relative binning itself incurs, not the
thing RB competes against. RB exists to eliminate the FULL-GRID matched filter —
that full-grid evaluation is exactly what `lnlike_bruteforce` does. So the coarse
call is not a meaningful ceiling for the contraction, and the measured `23x`
overshoot (`1.47e-3 s` contraction vs `6.4e-5 s` coarse call) is the additive
`M^2 + n_img^2` design behaving AS DESIGNED, not a regression: with IMRPhenomXPHM
(`n_m` up to 4) the contraction runs `~n_m^2` mode-pair einsums plus an
`(n_img, n_img, n_det, n_bins)` image reduction — roughly an order of magnitude
more numpy-dispatch work than a single cached coarse higher-mode strain call.
Beating one cached strain call was never the design's subdominance requirement.

The gate additionally EXCLUDED the F006 `_amplification_coefficients` cost from
measurement entirely, so it left the genuinely new per-eval work (the
Chang–Refsdal / 1F1 special-function engine, now evaluated at
`n_bins * kernel_subsamples` points every eval) completely unguarded.

The two correct gates:
1. `lnlike` faster than `lnlike_bruteforce` by a conservative margin — the
   actual RB speedup claim, stated against the PUBLIC entry points so it
   survives internal refactors.
2. `t_contract < t_amplification` — the pure `_data_term` + `_norm_term`
   contraction must be subdominant to the `_amplification_coefficients` call.
   The special-function engine at `n_bins * kernel_subsamples` points is the
   unavoidable per-eval cost of microlensed evaluation; a likelihood whose
   contraction is dwarfed by its own special-function evaluation is the correct
   shape for a special-function-dominated microlensing runtime. Baseline (2)
   also closes the hole the old gate left by excluding
   `_amplification_coefficients` from measurement.

### F→1 floor — a template-construction asymmetry, left in place for Build 2b

> Scope note (F009, 2026-07-17): this section is the UNLENSED limit
> (`gamma = kappa = 0`), where `F -> 1` exactly and the residual floor is
> construction noise. The SHEARED small-``w`` floor (`0.1214`/`0.1307`, similar
> magnitude, different mechanism) is NOT construction noise — it is the exact
> macro-magnification limit `F(w->0) = sqrt(mu_macro) != 1`. See F009; do not
> conflate the two.

The unlensed-limit floor readings (`0.10`–`0.33` across runs) are NOT a
normalization error. `_compute_d_h` / `_compute_h_h` apply `asd_drift**-2` once
per detector and BOTH oracles route through them; the `4*df` prefactor, blued
strain, and `wht_filter**2` match `CBCLikelihood` / `RelativeBinningLikelihood`.
The residual is a template-construction asymmetry: `_set_summary` builds
`_h0_edges` with `disable_precession=False` forced and `_stall_ringdown`
applied, while `_candidate_bin_ratios` builds the candidate `h_edges` with
NEITHER. So at `r ~ 1` the per-bin ratio is not identically `1` in the in-band
ringdown of the 60+45 M_sun fixture (`delta-h/h ~ 1e-3`), and beaten against the
(previously unseeded) noise this reads as `~ rho * (delta-h/h) ~ 0.05`–`0.3`.
For scale, the physically expected lensing residual at `w ~ 1e-7` is `~1e-4` —
about four orders of magnitude BELOW the observed floor, confirming the floor is
construction noise, not lensing signal or a normalization slip.

This construction asymmetry is a KNOWN residual mechanism left in place for
Build 2b: aligning the two template builders risks the currently-green
brute-force suite and is out of scope here. The crown gate handles it test-side
with a ZERO-NOISE `F→1` anchor (the deterministic unlensed-limit check that
removes the noise projection so the `delta-h` term cannot be amplified into a
spurious floor). This entry also corrects F006's "at `F→1` … exact to ~1e-8"
line: that bound holds only for the `p=0` moment with exactly-constant
ratio/kernel and matched frequency sets; the full `F→1` path carries the
construction-asymmetry `delta-h`.

### Static verification of the crown-gate deliverables (WP1 re-dispatch, 2026-07-16)

Recorded to resolve INS-3-001, which was an Inspector-SESSION access failure
(Bash false-denial, Serena/Read timeouts) — not a code or test defect. A
session with working file access statically confirmed that the shipped
artifacts exist and encode exactly what F006/F007 document:

- `cogwheel/lensing/likelihood.py`: `_amplification_coefficients` (the F006
  dense per-bin sub-sample + least-squares kernel reduction) is present as a
  `LensedRelativeBinningLikelihood` method, alongside its subsampling machinery
  (`_build_kernel_subsampling`, `_kernel_dense_f`, `_kernel_fit_value`,
  `_kernel_fit_slope`), the contraction (`_data_term`, `_norm_term`), and the
  retained `_edge_linear_coefficients` (`_amplification_at_bins` was later
  removed as dead code, INS-3-002).
- `cogwheel/tests/test_lensing_likelihood.py`: `ContractionTimingTestCase`
  asserts the two F007 baselines — (a) `lnlike` faster than `lnlike_bruteforce`
  by `SPEEDUP_MIN`, and (b) the `_data_term`+`_norm_term` contraction
  subdominant to the `_amplification_coefficients` engine call — and its
  docstring explicitly declines the old coarse-strain baseline as a per-eval
  co-cost. `NearCuspRegressionPinTestCase`, `NormalizationFloorZeroNoiseTestCase`
  (the zero-noise `F→1` anchor), and `DeterminismTestCase` are all present.

UNVERIFIED (handed to a runtime-capable reviewer, NOT the Coder — code and the
suite that blesses it must not share an author): the actual pass/fail state of
`python -m pytest cogwheel/tests/test_lensing_likelihood.py
cogwheel/tests/test_lensing_operator.py` at the stated tolerances. The static
read establishes the deliverables are real and structurally consistent with the
documentation; runtime green-ness must still be confirmed before Build 2b is
committed, and no tolerance may be widened to achieve it.

## F008 — the real near-cusp cause: `_channel_switch` measures delay separation against real channels only, not the full cluster (2026-07-17)

> **ADDENDUM (2026-07-18, Build 3f / SACR-C): switch keying superseded, F008's
> lesson preserved.** The SACR-C channel construction replaced the full-cluster
> pairwise switch keying with the CRITICALITY separation `delta_a = \|tau_a -
> tau_c\|` (each image's delay distance from the parked critical carrier). The
> switch scale is now the demodulation distance of the single transition
> envelope by construction, so the bounded-phase theorem applies (envelope
> phase <= RHO_END = 4 rad — beats impossible). For genuine mergers
> `tau_a -> tau_c`, so the new gate is AT LEAST as conservative as F008's
> full-cluster rule (measured `max\|S_a H_a\| <= 1.21` through fold/cusp
> crossings at eta = +-0.002); ACCIDENTAL delay degeneracies between
> non-merging images no longer stall the switch (small carrier separation =
> no beat, so staying unswitched is harmless). F008's underlying lesson —
> never key a cluster decision on a real-only neighbour set — remains valid
> and is embodied in `tau_c` coming from the parked/critical carrier.

This is the actual mechanism behind the Build-2b crown-gate accuracy failures
that F006 mis-attributed (F006 now SUPERSEDED). The defect is one line in the
lens engine, not in the relative-binning likelihood.

Root cause. `_channel_switch` (`cogwheel/lensing/chang_refsdal/channels.py`)
ramps a real channel over to the divergent saddle-kernel branch based on that
channel's delay separation from its neighbours. It computed the neighbour set
as `others = real_ids[real_ids != channel]` — REAL channels only. The paper's
Eq. (delay-separation) takes the minimum over ALL members of the image's
cluster, INCLUDING the labels parked at the critical point (the "virtual"
labels). On the 2-image side of a caustic a near-critical image's actual
cluster-mates ARE those parked virtual labels: the measured gap to a virtual
label was `5.5e-5` at the crown `near-cusp` config, versus `0.856` to the
nearest persistent real image. Keying only on real neighbours, the switch saw
the large `0.856` gap, spuriously ramped to 1, and handed the channel to the
divergent saddle kernel `H` (`|H_0| ~ 1.8e8` there, growth `~ gap^-2` toward the
cusp), which then flooded all four channels through the residual projection
(`|K_a| ~ 5.2e5`, cancelling coherently to `|F| ~ 3`). Squaring that in the
norm term is what produced the negative-huge `(h|h)` excursion — an UNBOUNDED
upstream kernel, not an aliased reduction slope (which is why F006's
slope-squaring story was sign-wrong).

Fix (Build 2c, WP1). Replace the neighbour set with all cluster labels except
self (`np.delete(np.arange(_N_CHANNELS), channel)`), so parked virtual labels
count as the legitimate neighbours Eq. (delay-separation) intends. The fix is
one-directional (it can only LOWER a switch value) and a no-op wherever all
four labels are real (4-image regions, near-fold-inside): there `real_ids` is
the full set, so old and new neighbour sets are identical and the result is
bit-for-bit unchanged.

Measured effect (two independent agents, scratch probes; repo untouched):

| config    | switch        | max\|k0\| | RB − brute lnL | 1.5 gate |
|-----------|---------------|-----------|----------------|----------|
| two-image | current (bug) | 40.9      | +9.768         | FAIL     |
| two-image | fixed         | 0.922     | +0.080         | PASS     |
| near-cusp | current (bug) | 5.22e5    | +6.43e8        | FAIL     |
| near-cusp | fixed         | 0.975     | +0.329         | PASS     |

- `kernel_subsamples = 2` under the fixed switch: `+0.069` (two-image) /
  `+0.316` (near-cusp) — both PASS. This is why F006's dense-subsampling
  compensation is unnecessary and Build 2c reverts the default 8 → 2 (WP2).
- Moment orders are irrelevant to the fix: `p+s ≤ 3 == p+s ≤ 4 == p+s ≤ 5` to
  `< 1e-4` once the kernels are bounded — no norm-moment change is needed; the
  contraction algebra is correct as shipped.
- Brute force is switch-independent (it uses `exact_total`; reconstruction
  `~1e-16` under either switch) — the oracle never moved, so the disagreement
  was entirely the RB path riding the unbounded kernel.
- Reconstruction error IMPROVES under the fix: `2.5e-10 → 5e-16`.

Sibling audit — `_min_delay_separation` NOT implicated. The sibling
`_min_delay_separation` (feeds the wave/geometric branch gate) carries the same
real-only pattern but is deliberately kept real-only: the geometric branch
replaces `F_op` with the stationary-phase sum over REAL images only (virtual
labels carry no saddle), so its resolution gate must key on real-image
separation. `exact_total` is unaffected by the switch bug, confirming this
sibling is not implicated in the crown failures. Recorded here so the two
real-only patterns are not "fixed" together by mistake.

Cross-references. F005 (wave-branch contraction overflow/refusal, NARROWED)
and F007 (timing-gate spec and the `F→1` template-construction asymmetry) are
UNAFFECTED by this fix — the switch correction touches neither the operator
contraction refusal thresholds nor the template builders. It does, however,
retire F006's dense-subsampling rationale and lets the crown gate pass at the
original tolerances (`RB_ATOL = 1.5`) with the sub-sampling default back at 2.

## F009 — the small-``w`` floor is the exact macro-magnification limit, not a `gamma/(2w)` singularity (2026-07-17)

The Build-2d episode. After Build 2c landed (F008), three tiny-``w``
observations were read as a fresh engine defect; a consult misdiagnosed them
as a series singularity and prescribed an engine short-circuit; the Architect
rediagnosed them as an EXACT closed-form physical limit, and the test ratified
that to 16 digits. No engine code changed. Recorded so the short-circuit is
never re-proposed.

### 1. Symptom (three post-2c "failures")

* Zero-noise likelihood floors of `0.1214` and `0.1307` against a `0.01`
  expectation, in the sheared small-``w`` regime.
* A small-mass `|F| - 1` that sat FLAT at `2.062e-2` — refusing to vanish as
  the mass (and hence ``w``) shrank, where a diffraction signal was expected
  to switch off.
* The channel-tracker "flat-gate" premise (that `|F| -> 1` as `w -> 0`) was
  apparently void.

### 2. Misdiagnosis (consult) — FALSIFIED

A consult attributed the flat `|F| - 1` to a `gamma/(2*w)` small-``w`` series
singularity in the operator power series (the prefactor `i*gamma/(2*w)` grows
as `w -> 0`), and prescribed an ENGINE SHORT-CIRCUIT: detect tiny ``w`` and
return `F -> 1 + O(w)`. This is sign- and scale-disproven below and must not be
implemented. The operator series is summed in the shear eigenframe and its
value is bounded as `w -> 0`; the growing prefactor does not make `F` diverge,
it makes `F` approach a NON-UNIT constant.

### 3. Actual physics (Architect rediagnosis, ratified by test)

`F` is normalized to NO LENS AT ALL, not to the macro image. As `w -> 0` the
point-mass diffraction switches off, but the smooth QUADRATIC macro potential
(convergence + shear) integrates EXACTLY — a quadratic Fermat phase makes the
diffraction integral a Gaussian — so

    F(w -> 0) -> 1 / sqrt((1 - kappa)**2 - gamma**2) = sqrt(mu_macro),

a real, positive, MASS- and FREQUENCY-INDEPENDENT constant, not `1`. The flat
`|F| - 1 = 2.062e-2` at `gamma = 0.20`, `kappa = 0` is exactly
`1/sqrt(0.96) - 1` to 16 digits — the closed form, not roundoff. The
zero-noise floor is then the exact projection of that constant offset,
`0.5 * <|F - 1|**2> * (h0|h0) = 0.1214`, not a defect. `|F| - 1` vanishes as
`w -> 0` ONLY when `gamma = kappa = 0` (the unsheared control), which is why
F007's `F -> 1` premise holds only there.

Gate (source of truth, `operator.py` "NORMALIZATION AND THE w -> 0 MACRO LIMIT"
docstring): `test_lensing_operator.py::MacroMagnificationLimitTestCase` pins
`|F_op|` to the LITERAL closed form `1/sqrt((1-kappa)**2 - gamma**2)` — never
built from `F_op`/`channels`/`geometry` (F002 oracle-tautology trap) — to
relative `7.85e-9` across a 48-point positive-parity grid (4 shears x 2
convergences x 2 shear orientations x 3 frequencies) spanning THREE DECADES of
tiny ``w`` (`1e-8, 1e-10, 1e-12`); the plateau's frequency-independence is the
signature that separates the exact limit from a `1/w` singularity. The
prescribed short-circuit would inject a real 2% DISCONTINUITY at the crossover
and DESTROY the exact pure-shear limit — never add one.

### 4. Residual pin (what the RB anchor is, and is not)

The RB zero-noise anchor decomposes as an INHERITED `8.962e-3` standard
`RelativeBinningLikelihood` stall floor plus a `2.676e-3` lensing-layer
increment (gated at `5e-3`); their sum is the `1.164e-2` fft-comparison
regression PIN — a pin, not a physical claim. The `8.962e-3` term is upstream,
not a lensing defect: the stall-ringdown is applied to the REFERENCE
(`_h0_edges`) only, BY DESIGN, so at fiducial parameters and zero noise the
per-bin ratio `r = h/h0 != 1` in the ringdown band and its linear-model
residual IS that floor. The stall is LOAD-BEARING (it buys a smooth
interpolable reference) and MUST NOT be removed or weakened; the requirement is
lnL accuracy, driven upstream through the normal build workflow
(`todo.d/likelihood_standard-rb-zero-noise-floor.md`), not by touching the
stall.

### 5. Lesson (verbatim, the Architect)

> a "numerical artifact" that is FLAT across many decades of the
> supposedly-singular parameter is almost never roundoff — match it against a
> closed form before planning around it.

## F010 — numba compilation silently voids Python-level test instrumentation (2026-07-17)

When the Build 3/3b fast path njit-compiled the dd primitives, the 1F1
ladder, and the operator contraction, three previously-green tests in the
OLD suites broke — none of them accuracy regressions, all of them
Python-introspection assumptions that compilation invalidates. Recorded
so the next numba migration re-checks this list instead of rediscovering
it:

1. **Module-global patching does not reach compiled code.** numba freezes
   module globals at compile time, so `mock.patch.object(_dd, '_SPLITTER',
   ...)` never reaches the njit `_split`. Worse, patching only the OUTER
   function to its `py_func` is not enough when it calls another njit
   function via module globals: `_two_prod.py_func` resolved `_dd._split`
   to the frozen njit dispatcher, so the dd splitter falsification
   (`test_broken_splitter_breaks_two_prod`) passed vacuously-green in
   reverse — the broken splitter no longer broke anything. IDIOM: patch
   the ENTIRE call chain to `py_func` bodies (`_split` AND `_two_prod`)
   for the duration of the sensitivity check.
2. **Call-counting wrappers count zero.** `LadderComplexityTestCase`
   replaced `_hyp1f1.dd_mul`/`dd_complex_mul` with counting wrappers; the
   njit ladder core binds them at compile time, so the counters read 0.
   Same idiom: swap `_shared_numerator`/`_ladder_sum`/`_ladder_core` to
   their `py_func` bodies while counting — the py bodies are the same
   algorithm numba compiles, so the counts still measure the shipped
   design.
3. **Internal-shape assumptions break silently.** `_amplification_
   coefficients` now returns the COARSE-node partition (~`n_kernel_nodes`
   points), not the dense sub-sample partition; a diagnostic helper that
   zipped it against `_kernel_dense_f` (506) crashed shape-wise. Tests
   reading engine profiles must evaluate `ChangRefsdalChannels` directly
   at the grid they want, not scrape likelihood internals.

The falsifiability lesson generalizes: after ANY compilation/JIT change,
re-run the self-falsification tests and confirm they can still go RED —
a falsification test that cannot fail anymore is worse than no test, and
nothing else in the suite will tell you.

## F011 — the certification-blind eps64 class in paired-rule quadrature (2026-07-19, Build 6)

A paired-rule (N-vs-2N) quadrature certificate is BLIND to any error
that is bit-identical in both rules. In the Schwinger saddle evaluator
two such errors shipped, each a single float64 operation amplified by
the `e^{pi w/4}` prefactor into silent fabrication above `w ~ 20`
(O(1) wrong by w=45, |F| ~ 8.5e3 at w=59.9 — returned WITH a green
certificate):
1. The IBP endpoint evaluated at `t_cap` while both rules split at
   `e^{fl(log t_cap)}` — the identity is split-point-arbitrary but
   demands ONE split point; fixed by deriving endpoint and domains from
   the same `u_mid`.
2. `1/s` computed in float64, entering the endpoint and A-pieces but
   not B, so it can never cancel; fixed with a dd reciprocal.
Extinction was PROVEN, not assumed: post-fix error vs the independent
oracle is 6.6e-15 (w=30) and 1.6e-11 (w=59.9), while any surviving
class member would measure ~3e+4 at the ceiling — fifteen orders
excluded. Audit rule for any future paired-rule certificate: every
float64 quantity that (a) enters the certified identity inconsistently
or (b) multiplies the accumulated result, and is identical across both
rules, is a candidate silent fabricator — carry it in dd or prove it
benign (parameter roundings consumed self-consistently are benign; the
domain-truncation margin is N/2N-invisible but analytically bounded).

## F012 — near-axial quartic dead zone: silent wrong-finite wedge above the wave ceiling (2026-07-19, pre-existing, both parities)

`find_images_quartic` silently drops the symmetric near-degenerate
image pair for sources ~1e-10..1e-9 (rel. angle; wider on positive
parity, to 1e-8) off a macro-matrix eigenaxis inside the 4-image
region — the dead band between the axial-path threshold and the
generic path's removable-singularity guard. Index-theorem violation
(signed Morse sum wrong). Consequences: for `w <= 60` the wave branch
is IMMUNE (it never consumes images); but above the ceiling,
`_real_delay_min_separation` computed from the surviving images
misreports a truly-unresolved config as resolved and the geometric
branch returns an O(1)-wrong finite value — a certify-or-refuse
violation confined to {dead zone} x {w > 60} (~1e-10 hit probability
per proposal, but silent). Tracked by
`NearAxialQuarticDefectTestCase` (@expectedFailure). REQUIRED Build-7
precondition: a runtime index-theorem check (signed Morse sum ==
sign(det A) - 1) in every image-consuming path, converting the dead
zone into a named refusal on both parities.

GUARDED (Build 7a): `_check_image_census` now runs at the end of
`find_images_quartic` (the single image producer — `find_images` is a
pure alias and `_centered_source_images` is internal to the quartic
path), enforcing the parity-agnostic invariant
`sum (-1)^{n_a} == sign(det A) - 1` with no tolerance band (a dropped
mirror pair shifts the sum by an even +-2; a fold-merging pair
contributes 0 and cannot false-positive). The dead zone is now a
named `LensDomainError` census refusal on BOTH parities;
`NearAxialQuarticDefectTestCase` was flipped from @expectedFailure to
a positive assertRaises with a positive-parity twin.

## F013 — the negative-parity (macro-saddle) branch: certified summary (2026-07-19, Build 6)

Full story in `.claude/handoff/lensing/negative_parity_research.md`
(design authority) and the two suites
`test_lensing_saddle_geometry.py` / `test_lensing_schwinger.py`.
Measured certification: census (index sum -2, multisets
{1,1}/{0,1,1,1}) over 200+ sources incl. both deltoid lobes; Schwinger
vs independent AST-guarded mpmath oracle 9.1e-14 (w=20) to 1.6e-11
(w=59.9); deep band |F| -> 1/sqrt(gamma^2-(1-kappa)^2) at 4.4e-5
(w=1e-4) with Morse intercept -pi/2 to 1.5e-7 (F009-S); mass-sheet on
observables 1e-13-class, lam <= 0 refused (F004 float64-exact
boundaries); positive parity pinned bit-for-bit to pre-extension HEAD.
The single cancellation channel is `L_S = pi*w/4`, y-independent
(F001-S); ceiling w <= 60 with certify-XOR-refuse (F005-S). Warm cost
~30-125 ms/point (w-linear) — the surrogate is load-bearing for the
homogenized architecture. INTERIM LAYER CONTRACT: the channel/waveform
layer refuses saddles by name (guards in `channels.evaluate` and the
`LensedWaveformGenerator` constructor) until Build 7 delivers the
saddle-domain channel layer.

## F014 — `lnl_marginalized` integrates arrival time with a unit-density (1 per second) prior — DELIBERATE convention (2026-07-19, stock cogwheel)

The coherent-score marginalized likelihood
(`MarginalizationInfo.lnl_marginalized`, consumed by
`MarginalizedExtrinsicLikelihood.lnlike` and the lensed subclass) is
not normalized over arrival time: the only time factor in the QMC
numerator is `sky_prior = (dOmega/4pi) * (1/sky_dict.f_sampling)`,
which carries units of SECONDS. Per the owner, this is a DELIBERATE
design choice (Javier's): with the integrand exp(<d|h> - <h|h>/2), a
silent stretch of window contributes e^{-<h|h>/2} ~ 0, so
integral(L dt) is invariant to how much data the analyst happened to
include — evidences stay comparable across analyses with different
window lengths, which a proper 1/T prior would break. Sky
(dOmega/4pi), orbital phase (1/2pi), polarization (1/pi), and
distance (uniform-in-volume to `lookup_table.d_luminosity_max`;
verified exact against direct quadrature) are all proper.
Consequences:
- `lnl_marginalized` = ln[ integral L dt <proper angles/distance> ] —
  it is ln(seconds)-offset relative to any oracle that uses a proper
  uniform t prior 1/T. A validation oracle must ADD ln(T_oracle) to
  compare. The offset is intrinsic-independent and cancels in
  posterior sampling; for model comparison, evidences computed under
  this convention are mutually consistent (that is its purpose) but
  differ from a proper-1/T-prior evidence by ln(T/1s). In particular
  the program's headline lensed-vs-unlensed Bayes factor is safe by
  construction: both hypotheses use the same coherent-score
  machinery, so the convention cancels identically in the ratio —
  only mixing in an evidence from a proper-1/T-prior code requires
  the ln(T/1s) adjustment.
- Verified empirically to 1.5 mnat: widening an importance-sampling
  oracle's t window 3.906x moved the gap by ln 3.906 while the
  marginalized value did not move (n_eff ~ 19-21k of 40k draws,
  Student-t proposal; probes marg_oracle_probe5*.py, 2026-07-19).
- After the convention correction a residual remains: marg LOW by
  0.146 +- ~0.03 (unlensed control) and 0.193 +- ~0.03 (lensed
  MAIN_LENS point) — both inside the 0.3-nat oracle gate, and their
  difference (0.05 +- 0.04) shows no lensed-fold defect. The ~0.15
  residual is an upstream stock-cogwheel effect (candidate: sky-delay
  discretization at f_sampling); track before using marginalized
  values as absolute evidences.
- Beware numerology: the uncorrected unlensed gap (2.0795) matched
  ln 8 to 1e-4 by pure coincidence — the T-scaling test is what
  discriminates, not constant-matching.

## F016 — the rescued strong-shear lnlike gap is RB-binning/noise-limited, NOT envelope-limited (2026-07-20, Build 7b falsification)

Build 7a's rescued-node measurement (ratio/direct paths agreeing with
each other but ~0.9 nats from brute force at gamma' ~ 0.94) was
root-caused during 7b planning as envelope under-resolution (the
max|F| error-currency normalization under-weighting deep-cancellation
troughs) and WP4 tightened the LOO stop to fix it. Test authorship
FALSIFIED that mechanism: the gap is INSENSITIVE to the stop (1e-3 ->
1e-5 moves 0.72 -> 0.75 nats) and SWINGS WITH THE NOISE SEED
(gamma'=0.8: 0.004 vs 0.150 nats across two seeds; gamma'=0.94: 0.72
vs 1.35). It is an RB-binning / data-noise effect of the same class
the crown gate prices at RB_ATOL=1.5 — larger here because the
strong-shear F(w) has more unresolved structure per bin. Consequences:
- The gamma'-keyed two-tier LOO stop (_LOO_STOP_STRONG=1e-3 for
  gamma' >= 0.5) is JUSTIFIED ONLY by the research's saddle-side
  envelope gate (reconstruction eps < 1e-3, enforced in
  test_lensing_saddle_channels), not by the nat-gap story.
- Rescued/strong-shear likelihood accuracy is gated at the standard
  RB tolerance; saddle-family configs (gamma' ~ 1.25-1.3) measure
  ~0.04 nats and pass the tight 0.1-nat gate comfortably.
- A principled strong-shear RB-binning audit (bin density vs F(w)
  structure) belongs to the Build 8 program alongside the surrogate.
- S2 node-count note: under the tightened saddle stop, genuine
  2-decade saddle windows converge at N ~ 40-42 envelope nodes (below
  the 48 cap; true reconstruction error 2-4e-4) — the research's
  N <= 30 was calibrated on <=1-decade windows.

## F017 — theta of the nearest caustic point is gauge, and the old Brent was the less accurate party (2026-07-20, Build 8b-levers)

The Newton reimplementation of `nearest_caustic_point` (analytic
g'/g'' on the stationarity condition; wedge-clamped per lobe/branch;
bounded-Brent fallbacks at cusps/discriminant clamp) preserves the
PHYSICAL observable to certification grade: distance within 9.3e-12
relative over 5677 both-parity configs, the HEAD pins at ~1e-16, zero
global-min basin misses, branch/lobe selection identical off
degeneracies. Theta, however, drifts up to ~5e-9 at SHALLOW minima —
and the evidence shows the drift is dominated by the OLD path's error
(Brent at xatol=1e-12 on a near-flat objective under-converges; the
Newton iterate sits closer to the independent dense oracle).
Dispositions to reuse:
- Theta is internal parametrization (gauge), not a certified
  observable (Professor ruling): its bit-exact pin component was
  re-certified at 1e-10 (well-conditioned) / 1e-8 (shallow-minimum)
  vs the pinned values, distance kept at places=14.
- The right theta gate currency is ARC LENGTH (theta_gap x
  caustic_speed) against the independent oracle — cusp-safe (speed ->
  0 tolerates the genuinely ambiguous angle) and immune to the
  old-code-was-worse trap that a HEAD-referenced theta gate falls
  into. Falsified by a forged non-global theta going red.
- General lesson: when a reimplementation is MORE accurate than the
  incumbent, value-preservation gates against the incumbent must be
  scoped to the physically certified quantities; gauge quantities get
  gated against an independent oracle, or the gate punishes the
  improvement. Timing: caustic search 1.23 -> 0.095 ms (positive
  parity), 4.54 -> 0.99 ms (saddle; the residual is 4 cusp-Brent
  fallbacks). The operator contraction fusion alongside it is 0-bit
  different across the certified sweep with refusal parity exact.

## F018 — the tube chart's measured advantage is against the extrapolating far-field, not a fair same-band raw chart; design-claim currencies matter (2026-07-20, Build 8c census tests)

The 8c design falsifiable was stated as "tube beats an equal-budget
raw chart by >= 3x at eps_95; raw fold-approach slope ~ -1/2". The
independent census test developer measured extensively: in the
census's MAX-NORMALIZED error currency (max|dE|/max|E|, the
F016-correct currency for lnL impact) a FAIR raw Cartesian chart
trained ON the same narrow band TIES the tube at fixture scale
(ratio ~0.8-1.1) and its fold-approach error IMPROVES inward
(slope ~ +0.5) — because the fold divergence of |E| itself inflates
the denominator and masks absolute-error growth. The tube's measured
advantage appears against the EXTRAPOLATING far-field chart (trained
outside the caustic, eta_overlap_min = 0.05) — which is the actual
production alternative the tube replaces: p95 ratio 2.93, deep-
caustic gap 3.01, raw slope saturating at ~ -0.19 (not -1/2; same
denominator effect). Dispositions:
- The Professor's 3x / -1/2 numbers came from derivative counting in
  a POINTWISE currency; any design-advantage claim must name its
  error currency, and gates must use the currency that tracks lnL.
- The tube design still stands on (a) the curved-band tiling
  argument (a fair raw band chart exists only on a short arc segment;
  globally, axis-aligned boxes fight the curved caustic and straddle
  image-count changes) and (b) the asymptotic convergence-rate gap
  (h^4 in u vs h^{1/2}-limited in eta), which fixture-scale grids are
  too coarse to expose.
- In-build bars (binding, coarse-fixture): tube-vs-extrapolant >= 2x
  at p95 / >= 1.5x at max; tube ray slope |s| < 0.15; raw-extrapolant
  slope < -0.05; deep-ratio >= 2. The literal 3x figure moves to the
  production-density training report (post-8e), measured in BOTH
  currencies.

## F019 — the two exact evaluators' ceilings live in different variables; a homogenization plan missed it and the census caught it pre-commit (2026-07-21, Build 8d)

The legacy operator series refuses on the PRODUCT w*sqrt(s) <= 60
(the y-dependent dd channel, F001/F005); the Schwinger quadrature
refuses on the frequency ALONE, w <= 60 (its cancellation channel
L_S = pi*w/4 is y-independent, F011/F013). The sampled prior box
bounds the PRODUCT (w*sqrt(s) <= 58 by construction) while w itself
runs to ~443 — so the two domains are NOT nested, and "route
everything through Schwinger" (the approved 8d plan, incl. Professor
Q2(b) "Schwinger-refuses => production refuses", driver-co-signed)
would have wholesale-changed the disposition of the ~25% of prior
draws carrying w > 60 non-geometric nodes. The WP3 geometry census
measured this BEFORE commit. Dispositions:
- OWNER RULING: 8d ships as the PURE homogenization anyway — the
  corner refuses by name until Build 8e serves it, because sampling
  is PARKED (ruling A): nothing production evaluates the corner in
  the interim, so coverage parity with pre-8d was a non-goal. (A
  driver per-band legacy revival was implemented and REVERTED on
  this ruling.)
- The legacy series also truncation-refuses part of the corner
  (shear-ladder degradation with w, measured at w=100 gamma'=0.2),
  so the 25% is an UPPER BOUND on what 8e must serve, not a
  pre-8d-served fraction; the truly-refused sub-fraction needs the
  8e scoping census (per-node evaluation).
- Standing lessons: domain claims comparing two evaluators must name
  the VARIABLE each ceiling lives in; reverse-coverage (B-refuses-
  where-A-served) is part of any evaluator-swap verification; and a
  cheap geometry census run BEFORE commit is what turned this from a
  shipped regression into a plan amendment.
- Cost note: homogenization re-prices the exact path (~90 ms/node,
  crown lnlike ~751 ms default; brute comparisons ~138 s/call) —
  brute-heavy accuracy tests are gated behind COGWHEEL_BRUTE_ACCURACY
  (driver post-build tier); the production hot path (surrogate/
  geometry) is unaffected.

## F015 — fold-degenerate images crashed the geometric kernel with a raw LinAlgError (2026-07-19, surfaced in production, fixed Build 7a)

The headline marginalized sampling run died mid-flight (bound 17+)
when a nautilus proposal produced an image on a fold:
`geometry._saddle_metric` inverted the projected Fermat Hessian with a
bare `np.linalg.inv`, the exactly-singular matrix raised
`numpy.linalg.LinAlgError`, and the unnamed exception sailed past
`LensedPosterior`'s refusal net (which maps `LensDomainError` /
`CancellationError` to -inf) and killed the whole sampler. This is
the unresolved-near-caustic corner (the fold/cusp Airy-patch gap)
manifesting as a CRASH class rather than the known accuracy class.
Fix: `_saddle_metric` catches the `LinAlgError` itself and re-raises
it as the named `LensDomainError` ('Fold-degenerate image') — refusing
EXACTLY the crash class and nothing more. Two threshold attempts were
wrong before this landed: `1e-13 * ||P||_F^2` amputated near-fold
configs the channel sweep legitimately serves (det ~ 40*eps), and even
`4*eps * ||P||_F^2` broke the channel layer's on-cusp rows (det ~
2*eps), because the SACR-C/F008 switch design DELIBERATELY consumes
huge near-singular metrics and multiplies the divergent
stationary-phase target away. Lesson: a guard for a crash class must
be scoped to the crash condition itself, not to a nearby-looking
numerical neighborhood. Two open notes:
1. NEAR-singular (not machine-singular) projected Hessians still
   return finite, increasingly divergent SPA kernels — the principled
   near-fold validity bound belongs to the fold/cusp uniform (Airy)
   asymptotics program, not to an ad-hoc threshold.
2. Audit rule: any raw `np.linalg.inv/solve/cholesky` on the physics
   path is a latent member of this class (a repo grep found exactly
   this one; `_newton_polish` already guards its solve with an lstsq
   fallback).

## F020 — a far-field chart trained below `w_floor` fits a divergent object; and a tile centred on a cusp ray fits a slope kink (2026-07-27, Build 8h-d triage)

Two INDEPENDENT causes of an inflated held-out `eps` in exterior chart
fixtures, both invisible to inspection and both worth orders of magnitude.
Recorded because a prior pass searched only box PLACEMENT (three `rho_c`
values, halves down to `(0.1, 0.03)`, narrower gamma bands), found nothing,
and recorded the degradation as unexplained. Neither cause is reachable by
varying tile geometry.

1. WRONG LABEL BELOW `w_floor`. `FARFIELD_KERNEL_SUM` subtracts the real
   image kernels, which is the correct label only in the mid band
   `[w_floor, w_trust)`. Below `w_floor` no real pair separates and the
   residual is the divergent diffractive-bottom object; the correct label
   there is `FARFIELD_DIFFRACTIVE` (subtract nothing). A fixture training the
   kernel-sum label from `w = 0.0248` against a region `w_floor = 0.661` put
   ~11 of ~15 log-`w` nodes under the floor and measured `eps = 7.67` against
   a `3e-3` bar. Correcting the band bottom to the region floor
   (`_farfield_region_w_floor`) alone took `eps` to `9.0e-3` — a factor 847.

2. CUSP-RAY TILE CENTRES. For the astroid the cusps sit at
   `theta_c = 0, pi/2, pi, 3pi/2`, where `r_caustic` has a slope KINK. A tile
   centred on one asks a cubic spline to represent a non-smooth map. Moving
   the same fixture's centres off the rays (to 0.5 / 0.95 / 1.4, all inside
   `(0, pi/2)`) took `eps` from `9.0e-3` to `3.4e-4`. Combined with (1):
   `7.67 -> 3.4e-4`, a factor 22,500, with the bar and the poison factor
   UNCHANGED.

DIAGNOSTIC SIGNATURE for (2): the ordering inverts. A deliberately POISONED
chart scored BETTER than the "healthy" one (2.61 vs 7.67) because the healthy
tile straddled a cusp ray while the poisoned tile sat between cusps. An
incoherent ordering between a control and its degraded twin means neither is
fitting anything — look for a structural cause, not a tolerance.

The production tiler already addresses (2) with cusp-ALIGNED columns
(`_cusp_aligned_theta_tiles`), which put the ray on a column EDGE; see
`OnCuspColumnEdgeTestCase`. Any fixture or chart path that does NOT route
through that alignment inherits this defect.

## F021 — agent short-term memories are tail-capped, so a build's findings can be evicted before the Dreamer consolidates them (2026-07-27)

Inlined short-term memories are truncated to a 24 KB tail. When several
builds run between Dreamer passes, the OLDEST entries are dropped first — so
a finding recorded early in a busy day never reaches long-term memory, and
its absence is silent: consolidation reports success having promoted only
what survived.

Observed 2026-07-27: after three builds in one day, two substantive findings
(F020 above, and the deletion of 25 structural test classes with a restore
SHA) were absent from every short-term memory at consolidation time. The
Dreamer correctly REFUSED to promote them rather than inventing content, and
flagged the gap — which is the right behaviour and the only reason the loss
was noticed.

Mitigations, in order of preference: run the Dreamer between builds on a busy
day rather than at the end; write durable findings to `FINDINGS.md` (this
file) at the time they are measured, not via memory; and treat a Dreamer
report that flags "pattern not present" as a signal to recover from git or
the build handoffs, never as a reason to drop the finding.

## F022 — the far-field carrier guard measures `arg`, but the interpolant splines re/im; a phase flip at an amplitude null is smooth in re/im and refinement cannot remove it (2026-07-28, Build 8h-d2 triage)

`_assert_farfield_carrier_continuity` (`lensing/surrogate.py`) rejects an
exterior tile whose frame-invariant label `E_tilde` winds by `>= pi/2` in
`arg` between adjacent spatial nodes at the top of the band. Three coarse
fixtures trip it: `test_lensing_surrogate_census.py::_pos_farfield_dense`,
`test_lensing_ppgo_bandsplit.py::BandSplitReconstructionTestCase.setUpClass`,
and `test_lensing_exterior_admission.py::_build_guard_chart`.

**The refinement test is the discriminator, and it must be run before any
conclusion about this guard.** A carrier / under-resolution story predicts the
per-gap step falls like `1/n` as nodes are added. An amplitude-null story
predicts it stays pinned at `pi` forever. Measured on the two `gamma`-wall
guard boxes:

| n_gamma | max wind (rad) | rel. amplitude at that pair | `d|re|/span` | `d|im|/span` |
|---|---|---|---|---|
| 4 | 2.68 | 0.157 | 0.430 | 0.239 |
| 6 | 3.00 | 0.051 | 0.153 | 0.015 |
| 8 | 3.07 | 0.621 | 0.392 | 1.572 |
| 12 | 2.97 | 0.948 | 0.119 | 1.938 |
| 16 | 3.12 | 0.0027 | 0.0036 | 0.0160 |

(box `gamma in (0.5, 1.5)` behaves identically, reaching 3.14 rad at relative
amplitude 8.8e-6.) The step does not shrink — it converges to `pi` — while the
amplitude at the offending pair collapses and the re/im increments go smooth.
That is a null, not a carrier. This reproduces independently what the 8h-d2
Coder found by instrumenting the census and band-split boxes (refine
`n_gamma` 4 -> 6 -> 8 -> 12, step does not shrink; census trips even at
`w_max = 4`), and it confirms Inspector finding INS-5-001.

**Root cause.** `FarFieldChart` stores and splines `envelope_real` and
`envelope_imag` as separate real fields. Near an amplitude null the complex
label passes close to the origin: `arg` swings by `pi` while `re` and `im`
both pass smoothly through zero. The guard therefore measures a quantity the
interpolant never sees. `pi/2` is the right Nyquist bound for a *carrier*, but
`arg`-winding is not the right observable for a re/im spline — the failure is
in the metric, not the threshold.

**Two wrong turns taken here, both from reasoning instead of measuring:**

1. *Parity-flip story* (driver): the `(0.5, 1.5)` box straddles the `gamma = 1`
   parity wall and showed a ~`pi` step, which looked conclusive. Refuted by the
   `(1.0, 1.6)` box, which lies entirely in the saddle region with the
   `gamma = 1` node refused out of every pair, and still winds 2.5-2.7 rad.
2. *"Genuinely too coarse" story* (driver): supported by one measurement at one
   resolution showing the flips at 9-29 % relative amplitude — healthy, so not
   noise. Refuted by the refinement table above: at `n = 16` the same flip sits
   at 0.27 % amplitude with smooth re/im. Measuring a single grid cannot
   distinguish a carrier from a null; only the `n`-sweep can.

**Why the prescribed floor was not enough.** A relative-magnitude floor is the
right direction and INS-5-001 stands, but a floor alone does not retire the
bypasses: at the fixtures' actual coarseness the worst pair sits at 0.157
relative amplitude, far above any sane floor, because a coarse grid straddles
the null at moderate amplitude. A 1e-3 floor was implemented and excluded
nothing.

**FIXED 2026-07-28.** The guard now measures the complex increment
`|E_lead - E_trail|` normalized by the peak `|E_tilde|` over the WHOLE grid,
against `_FARFIELD_CARRIER_STEP_MAX = 1.0`, replacing
`_FARFIELD_CARRIER_WIND_MAX = pi/2`. All three `_skip_carrier_guard=True`
bypasses are removed and the kwarg is deleted from `from_engine`, so the
escape hatch cannot be reached for again.

Calibration across every known fixture (this is what the bound is set from,
and re-deriving it is how to change it safely):

| fixture | slice-norm | whole-norm | all-slices | verdict |
|---|---|---|---|---|
| synthetic continuous | 0.1997 | 0.1997 | 0.1997 | pass |
| synthetic zeroed-flip | 0.1997 | 0.1997 | 0.1997 | pass |
| `gamma1 (0.5,1.5)` | 1.0861 | 0.1160 | 1.0619 | pass |
| `gamma1 (1.0,1.6)` | 1.2928 | 0.1556 | 1.3703 | pass |
| band-split box | 1.3542 | 0.0000 | 1.0004 | pass |
| census dense box | 1.5289 | 0.0000 | 0.2881 | pass |
| synthetic pathological | 1.8980 | 1.8980 | 1.8980 | **raise** |

Whole-grid normalization is the only column with a real margin: worst
must-pass 0.1997 against must-raise 1.8980, a 9.5x gap, bound placed at 1.0.
Two alternatives were measured and REJECTED — top-slice normalization (margin
1.24x) and scanning all `w` slices (margin 1.38x). Do not re-propose either
without re-running the calibration.

The bound is tuning-free to state: a violation means the label changed by more
than the entire chart's peak magnitude across one node gap. At full amplitude
that is `pi/3` of winding, i.e. STRICTER than the retired `pi/2` where the
label is strong, and permissive only where it has decayed to noise. The change
does not weaken the guard; it re-points it at the quantity the spline sees.

**What the all-slices experiment taught, and why it is recorded.** Scanning
every `w` slice flags the census box at 1.37 — yet that chart meets its
accuracy bar (1.08e-3 against 5e-3). So large mid-band increments are
COMPATIBLE with an accurate chart, which means this guard is a cheap
GROSS-ALIASING SCREEN and not an accuracy proxy. The held-out eps gate
(`_gate_chart`) is the real falsifier. Treating the carrier guard as an
accuracy check is what produced both over-strict designs.

**Lesson.** For any guard that measures the phase of a complex field, the
`n`-refinement sweep is the cheap discriminator and should be run *first*: a
carrier shrinks like `1/n`, a null pins at `pi`. Reporting the relative
amplitude at the offending pair (the guard now does this in its error message)
tells you which regime you are in without a probe, but only the sweep proves
it. Two plausible mechanisms were adopted here without that sweep and both
were wrong.

## F023 — the Born rung's stated rationale is backwards, its series was missing a term, and its `b1` had the wrong sign; the chart absorbs `ln w` for free (2026-07-28, Professor commission)

`chang_refsdal/_born.py` shipped DORMANT with `b1 = 1.0`, a placeholder, and a
stated purpose that measurement contradicts. Four separate results, all
measured against the exact engine rather than argued.

**1. `b1` is derived, and the placeholder had the WRONG SIGN.**

    b1 = -lam * (x0 . A^-1 . x0) / |x0|**2        # reduced: -(1 + g'P)/(1 - g'**2)
    a0 = -lam * gamma * P / det_a                  # P = cos 2(theta_x0 - beta)

with `P` referred to the MACRO IMAGE direction, not the source. A pure point
mass gives `b1 = -1`, not `+1`. Both collapse onto `r0_sq` and `x0_dot_y`,
which `_born_factors` already computes — no new geometry, no fifth convention
site. Three algebraic forms agree to 2.2e-14 over 4000 both-parity draws.

**2. The mandated series structure was INCOMPLETE.** Expanding the Kirchhoff
integral leaves TWO terms at order `1/q2r`, not one: the imaginary `b1` term
the ansatz had, and a real, `w`-independent `a0` term it did not. Worth
1-20 % depending on shear. The build implemented its mandate faithfully; the
mandate was wrong. A structure handed down as fixed is still a hypothesis.

**3. The WHY premise is measurably backwards.** The module says the low-`w`
far zone "varies on the Einstein scale, so trained tiles there are
prior-sized". Once `exp(1j*w*phi_geo)` is demodulated out, that variation is
ENTIRELY GONE — it lives in the closed-form phase. Measured: the demodulated
residual needs 4 y-nodes across the whole annulus at `w <= 0.2`, versus 9-17
at `w >~ 1`. Einstein-scale fringe motion is a MID/HIGH-`w` problem, the
opposite of the documented rationale. The rung is worth having, for the
reverse reason.

**4. The chart absorbs `ln(w/2)` at ZERO node cost, so no low-`w` analytic
rung is needed.** The term the mid-`w` carrier omits is
`(1j*w/2)*[ln(w*r0_sq/2) + EULER_GAMMA + 2*ln(Lam)] + pi*w/4`. With `u = ln w`,
`w*ln w = exp(u)*u` is ENTIRE in `u`, and the charts already carry a `log_w`
axis. Node counts to a given held-out error are IDENTICAL with and without the
log term at every tolerance from 4e-3 down to 1e-5. The residual is also
bounded as `w -> 0` (tends to `-a0/q2r`).

**Measured ladder (positive parity), residual as an interpolation object:**

| band | serve | residual / `max|F|` | nodes (`log_w`, y) |
|---|---|---|---|
| `w < 0.5` | carrier `(a0,b1)` ALONE | 2.4e-2 - 8.7e-2 | 4-15, 4 |
| `w >= 0.5` | ppGO (both real images, full C1/C2) + complex ghost where admitted | 1.6e-3 - 2.5e-2 | 4-8, 4 |

No gap: the seam `[0.05, 0.5]` is covered by the carrier at 7-15 nodes with
prior-universal tiles. Mixing the bands the other way is catastrophic — adding
ppGO below `w = 0.05` inflates the residual by FIVE ORDERS OF MAGNITUDE through
the `1/w**2` kernel.

**Terminology, resolved because it nearly caused a double-count.** The faint
near-lens SECOND REAL IMAGE (`x_c ~ -y/|y|**2`, Morse index 1, from
`find_images`, worth 4.4e-2 - 8.4e-2 here) is NOT the COMPLEX-saddle ghost that
`farfield_ghost_term` implements (conjugate quartic pair, `Im tau_c > 0`,
Picard-Lefschetz, gated on geometric separation >= 0.7). Both are real
contributions and must be counted ONCE EACH. The complex ghost is large below
`w ~ 0.2` but UNUSABLE there — its own stationary-phase kernel diverges as
`1/w`, `1/w**2` — and is worth 2.5e-3 -> 1.6e-3 in `[0.5, 8]`.

Also measured: `ghost_kernel` raises `GhostDomainError` at
`(|y|=3.6, theta=0.5, gamma=0.25, kappa=0.3, beta=0.5)` while `find_images`
returns 2 real images, so the complex ghost is NOT universally available in the
annulus and the serve path must tolerate its absence (it does — the error is a
`LensDomainError`).

**A methodological correction worth keeping.** The first pass reported that
adding the second image made the chart WORSE (121-241 nodes). That was an
artifact of using the leading `sqrt|mu|` amplitude only; with the full C1/C2
`image_kernel` the same residual collapses to 4 nodes. An approximation used
while EVALUATING an architecture can condemn the architecture. When a
measurement says a component hurts, check that the component was measured at
full fidelity before believing it.

**What `b1`/`a0` actually buy:** ~10-25 % smaller residual in the low band at
the same node count, superseded by ppGO above `w ~ 0.5`. The load-bearing
results of this commission were the sign fix, the `a0` omission, the regime
diagnosis, and the `log_w` absorbability measurement — not an accuracy gain.
The rung's own T1 target of 1e-3 was never the right bar: in the
carrier-plus-chart architecture the criterion is how CHEAPLY THE RESIDUAL
SPLINES, not how accurate the analytic term is standing alone.

## F024 — the band-split currency `w*r0_sq` was a positive-parity coincidence; the invariant is `w*Delta_tau`, and the annulus is only "exterior" for part of the prior (2026-07-28, Professor saddle commission)

Three results, each of which invalidated something already written down.

**1. The split currency.** F023 set the carrier/ppGO band split at
`w * r0_sq ~ 8`. That is correct ONLY at positive parity, and only because
`Delta_tau ~ r0_sq / 2` happens to hold there — a coincidence of the regime,
not an identity. On the macro saddle `x0_i = y_i / a_i` with `a1 = lam - gamma
< 0`, so `r0_sq` swings 1700x with angle (3721 -> 2.2 at gamma = 1.05) while
`Delta_tau` does not; measured `r0_sq / (2*Delta_tau)` spans 0.16 to 35.6.

The invariant is `w * Delta_tau ~ 4`, with `Delta_tau` the Fermat-delay
difference of the two real images — already available from the partition via
`geometry.delay`. It coincides with SACR-C's own switch scale `RHO_END = 4`.
Confirmed both directions: at `gamma=1.6, |y|=4.24, theta=0.9`,
`Delta_tau = 0.294` gives `w_split = 13.6` and the carrier still works at
`w = 8` (N=7); at `gamma=1.2, theta=0.3`, `Delta_tau = 35.3` gives
`w_split = 0.113` and the carrier has already failed by `w = 0.5` (N=161).
`w * r0_sq` mispredicts both by two orders of magnitude.

**2. The annulus is not always exterior.** Measured caustic extent on a 241^2
source grid: max `|y|` inside the caustic is 1.85 at `gamma=0.60`, 2.95 at
0.75, 4.35 at 0.85, and `>= 6.00` for `gamma >= 0.95`; on the saddle side
`>= 6.00` at `gamma = 1.005`, 3.71 at 1.02, 2.49 at 1.05, 1.70 at 1.30.

So the target annulus `3.0 < |y| <= 4.2426` is a FAR EXTERIOR region only for
`gamma <~ 0.75` and `gamma >~ 1.03`. Between them — roughly 17 % of the
prior's uniform `(0, 1.6)` shear range — it straddles or lies INSIDE the
caustic, with fold crossings in the tile and a (0,1,1,1) census. The exterior
ladder cannot close that band; it is either an interior-chart problem or a
named refusal.

This RETRO-SCOPES F023, whose positive-parity ladder was measured at
`gamma` in {0.2, 0.25, 0.3, 0.45} only. F023's conclusions stand for
`gamma <~ 0.75` and are NOT established above it. The Professor flagged this
against its own prior report rather than letting it stand.

**3. The saddle carrier is LEAD-ONLY, and the complex ghost is harmful there.**
`|a0|, |b1| ~ 1/(gamma' - 1)` (a0 = 10.24 at gamma = 1.05), so `a0/q2r` is
O(1) wherever `q2r` is small, and `q2r` falls to 1.4 near the
positive-eigenvalue axis. Measured on `[1e-3, 0.05]`: lead-only
`sqrt|mu_macro| * exp(-1j*pi/2) * exp(1j*w*phi_geo)` gives residual
1.0e-2 - 7.4e-2 at N=4 on both `log_w` and the y-axes, while the full
`(a0,b1)` carrier gives 1.7e-2 - 1.42 and needs N = 23-65 in the y-plane. The
correction injects theta-structure that is not in `F`. Drop it on the saddle.

The complex ghost is worse than useless there: at `gamma=1.6, |y|=4.243,
w=5`, ppGO alone gives residual 1.4e-3 at N(theta)=4; adding the admitted
ghost gives 4.2e-2 at N(theta)=14. Two causes — the admission set flips across
theta inside a tile (43-54 of 65 points admitted), and, more fundamentally,
`geometry.ghost_kernel` pins its sqrt branch with
`reference_amplitude = exp(-0.5j*pi)` justified in its own docstring by "the
two real images continue into a Morse-index-1 saddle". THAT IS A
POSITIVE-PARITY STATEMENT: on the macro saddle both real images are ALREADY
index-1. The branch reference has not been re-derived for `det A < 0`. Refuse
the complex ghost on the saddle branch until it is.

**Method caveat recorded by the Professor against its own numbers:** its
theta-direction node counts demodulate by the SINGLE carrier
`exp(1j*w*phi_geo)`. Above the split the residual inherits the other image's
carrier (`Delta_tau` varies ~25 per radian, so at `w=5` the fringe spacing is
~0.05 rad), which is why ppGO's theta counts look bad at `gamma=1.2` despite a
residual of 8.2e-3. The correct object above the split is the SACR-C switched
envelope. Residual SIZES are demodulation-independent and stand; the
above-split theta NODE COUNTS are pessimistic and must be re-measured through
`switched_analytic_channels` before they size a tile.

**Pattern.** Both F023 and F024 corrected a fact that had already been written
down and acted on — F023 the `b1` sign and the missing `a0`, F024 the split
currency and the annulus's exteriority. In each case the wrong version was not
a guess but a measurement taken over too narrow a slice. A measured fact
carries the scope of its measurement; F023 said "the split is at
`w * r0_sq ~ 8`" when what was measured was "the split is at
`w * r0_sq ~ 8` FOR gamma in [0.2, 0.45]". State the sweep next to the result.

## F025 — `a0` violates F009 and belongs in no serving path; the exterior fence is `gamma < 3/4` exactly; and F023's node counts were swept in one direction only (2026-07-28)

Third correction from the same Professor thread, and the one that most
simplifies the build. Sweep for every number: `gamma` in {0.45, 0.50, 0.55,
0.60, 0.65, 0.70, 0.75}, `|y|` in {3.05, 4.2426}, `theta` in {0.3, 0.9, 1.35},
`kappa = 0`, `beta = 0` (production pins both), azimuthal sweeps at 65 points.

**1. `a0` hurts at EVERY gamma, including 0.45 — the value F023 reported on.**
Azimuthal node counts at eps 4e-3, `w = 0.01`:

| gamma | N(F) | N(lead-only) | N(a0+b1) |
|---|---|---|---|
| 0.45 | 4 | 4 (size 1.9e-2) | 11 (size 1.2e-1) |
| 0.60 | 4 | 4 (size 2.0e-2) | 20 (size 2.5e-1) |
| 0.75 | 4 | 4 (size 2.3e-2) | 44 (size 5.5e-1) |

`N(F) == N(lead)` EXACTLY at all 18 sampled (gamma, w, radius) combinations.

**2. The mechanism is an F009 violation, not a coefficient blow-up.**
`F(w -> 0) = sqrt(mu_macro)` exactly (F009), but the carrier returns
`sqrt(mu_macro) * (1 + a0/q2r)`. `a0` is a RESOLVED-IMAGE amplitude
correction, valid only for `w * Delta_tau >> 1`; below the split it is a
constant offset of size `|a0|/q2r`. `b1` alone is indistinguishable from
lead-only — its term carries `w` and vanishes correctly — so it is harmless
but buys nothing.

CONSEQUENCE: `(a0, b1)` serve NOWHERE. Below the split lead-only wins; above
it ppGO wins. Use the lead-only carrier
`sqrt(mu_macro) * exp(1j*w*phi_geo)` below the split at ALL gamma on BOTH
parities — one rule, no branch, matching the saddle recommendation. The
coefficients remain correct physics and the right macro-limit diagnostic, and
should stay in the module as such; they should not be in the serving path.

**3. Why F023 missed it — a METHOD error, not a sampling one.** F023's y-plane
node counts swept `|y|` RADIALLY at fixed `theta = 0.3`. The `a0` pathology is
AZIMUTHAL: `q2r` varies strongly with angle through
`x0 = (y1/(1-gamma), y2/(1+gamma))`. Same gamma, same radii — the radial sweep
gives N=4 and size 2.4e-2; the azimuthal sweep gives N=11 and size 1.2e-1. One
direction was swept and reported as "per y-axis".

**4. The exterior fence, in closed form.** With `u = 1/|x|**2` on the critical
curve `u = gamma*cos2theta +- sqrt(1 - gamma**2 sin**2 2theta)`, the outermost
astroid cusp is

    max |y| on the astroid = 2*gamma / sqrt(1 - gamma)             (kappa = 0)
                           = sqrt(lam) * 2*gp / sqrt(1 - gp)       (gp = gamma/lam)

verified against the measured extent to 4 decimals at five gammas (0.60 ->
1.8974, 0.70 -> 2.5560, 0.75 -> 3.0000, 0.80 -> 3.5777, 0.85 -> 4.3894).
Solving `2 s**2 + R s - 2 = 0` with `s = sqrt(1 - gamma)`:

* the annulus INNER edge `|y| = 3.0` is breached at `gamma = 3/4` EXACTLY;
* the outer edge `|y| = 3*sqrt(2)` at `gamma = 0.8423291`.

So the annulus is fully exterior for `gamma < 3/4`, straddles the caustic for
`3/4 <= gamma < 0.84233`, and is fully interior above. The earlier "`<~ 0.75`"
was a geometric guess that happened to land on the right number; encode the
derived form.

**5. F023's node counts were floors, not ceilings.** Over `gamma <= 3/4` the
true ceilings are 5 on `[1e-3, 0.05]`, **31** on `[0.05, 0.5]` (F023 said
4-15), and **27** on `[0.5, 8]` (F023 said 4-8), with 4 and 14 per y-axis in
the two sub-split bands. The `[0.5, 8]` ppGO residual rises to 2.0e-1 near the
`y2`-axis cusp and PLATEAUS rather than diverging — that cusp reaches
`|y| = 3.0` exactly at `gamma = 3/4`, which is the same fence.

**The pattern, now three deep.** F023 corrected the placeholder; F024
corrected F023's split currency and scope; F025 corrects F023's node counts
and retires its headline recommendation. None of the wrong versions was a
guess — each was a real measurement whose SWEEP was narrower than the claim
built on it. F023 swept four gammas and one angular direction. The discipline
that would have caught all three: state the sweep in the same sentence as the
result, and treat any claim that outruns it as unmeasured.

## F026 — one closed form gives the caustic extent on BOTH parities; the saddle fence is a BAND; and F024's measured extent table under-reported the spike (2026-07-28)

Sweep: `kappa = 0`, `beta = 0`; closed form checked against a direct caustic
parametrisation (2e6 points per branch) at 16 gammas spanning both parities
(0.45 to 3.0) — agreement to 4 decimals at all 16. `kappa != 0` enters only
through the mass-sheet reduction (`gp = gamma/lam`, `|y| -> sqrt(lam)*|y'|`).

**The closed form.** The critical curve satisfies
`u**2 - 2*gamma*cos(2*theta)*u - (1 - gamma**2) = 0` with `u = 1/|x|**2`, and
the caustic is `y = ((a1-u)x1, (a2-u)x2)`, `a1,a2 = 1 -+ gamma`. Eliminating
`theta`, `|y|**2` becomes a function of `u` ALONE:

    |y|**2 (u) = 2*u - 3 + 2*gamma**2/u + (1 - gamma**2)/u**2,
                                       u in [abs(1-gamma), 1+gamma]
    f'(u) = (2/u**3) * (u - 1) * (u**2 + u + 1 - gamma**2)
        -> stationary at u = 1 and u_c = (sqrt(4*gamma**2 - 3) - 1)/2

`u_c > 0` IFF `gamma > 1`. That single fact is the whole difference between the
two parities.

* POSITIVE PARITY: `u**2 + u + 1 - gamma**2 > 0` for `u > 0`, so `f` falls then
  rises and the maximum is at the endpoint `u = 1 - gamma`:
  `max|y| = 2*gamma/sqrt(1 - gamma)`, UNCONDITIONALLY for all `gamma < 1`.
  The fence recorded in F025 is right with no caveat.
* SADDLE: `u_c` is always interior (`u_c > gamma - 1` iff `gamma > 1`), and
  there `gamma**2 = u_c**2 + u_c + 1`, collapsing `f` to

      max|y|_saddle = sqrt(max( 4*u_c + 1/u_c - 2,  4*gamma**2/(gamma + 1) ))

  the two candidates being the OFF-AXIS cusp (`u_c`) and the ON-AXIS cusp
  (`u = gamma + 1`, giving `2*gamma/sqrt(gamma + 1)`).

**The non-monotonicity is real: a cusp switch.** The outermost point sits on
the off-axis cusp for `gamma < 1.177651` and on the on-axis cusp above; the
`u_c` branch falls while `2*gamma/sqrt(gamma+1)` rises, so the extent MINIMISES
at the switch: `max|y| = 1.596072` at `gamma = 1.177651`.

**F024's extent table under-reported and is RETIRED.** Its 241^2 source grid
misses the thin spike. True vs grid: `gamma=1.005` -> 9.939 vs 6.002
(grid-capped); `1.02` -> 4.886 vs 3.712; `1.05` -> 3.008 vs 2.491. Use the
closed form.

**The saddle fence is a BAND, not a one-sided inequality.** Solving
`4v + 1/v - 2 = R**2` then `gamma = sqrt(v**2 + v + 1)`:

| edge | exact | value |
|---|---|---|
| inner, `\|y\| = 3.0` | `sqrt((189 - 15*sqrt(105))/32)` | 1.0502342 |
| outer, `\|y\| = 3*sqrt(2)` | `sqrt(63 - 24*sqrt(6))/2` | 1.0261879 |

and the RISING branch reaches the inner edge again at `4*gamma**2/(gamma+1) = 9`,
i.e. `4*gamma**2 - 9*gamma - 9 = 0`, `gamma = 3` EXACTLY (outer edge at
`gamma = 5.342329`). So the annulus is exterior for
`1.0502342 < gamma < 3`. The prior stops at `gamma = 1.6` where
`max|y| = 1.9846`, clear of 3.0 by a factor 1.51 — safe, but write the fence as
a band in case the prior is ever widened.

**Both sides diverge as `|gamma - 1|**(-1/2)`** — positive `2/sqrt(1-gamma)`,
saddle `1/sqrt(2*(gamma-1))` — so they join at the wall in power but not in
amplitude: the astroid is `2*sqrt(2) ~ 2.83x` larger at equal distance from it.

**The deltoid is spiky too, but a scalar fence costs far less there.** Angular
width beyond radius R (upper bound on the inside-fraction): `gamma=1.02` gives
<= 3.95 % beyond `|y| = 3.0`, `1.05` gives <= 0.029 %. On a uniform `(0, 1.6)`
prior the POSITIVE fence `gamma < 3/4` discards 15.6 % of the shear range while
the SADDLE fence `gamma > 1.0502342` discards only 3.1 %. Ship the scalar fence
on the saddle; spend per-theta admission on the positive branch, where it is
worth 5x more.

**METHOD WARNING, and it applies to the driver's own probe.** A 721-point ring
scan reported 0.00 % inside at `gamma=1.03, |y|=3.6` where the true spike is
NARROWER than the 0.0087 rad sampling. The driver's 1441-point scan
(0.0044 rad) returned 0.42 % at `gamma=0.80` and 2.91 % at `0.90`, just under
the analytic bounds 0.58 % and 2.98 % — it held, but only just. TRUST THE
PARAMETRISATION, NOT RING SCANS, FOR SPIKE GEOMETRY. A caustic that is a
directional spike cannot be characterised by sampling angles; sample the curve
itself.

**Fourth correction in one thread.** F023 fixed the placeholder; F024 fixed
F023's split currency and scope; F025 fixed F024's node counts and retired
F023's headline; F026 retires F024's extent table. Every wrong version was a
real measurement whose resolution or sweep was narrower than the claim built on
it. Grid at 241^2, sweep at four gammas, radial-only, ring scan at 0.0087 rad —
four different ways to under-resolve, four wrong tables.

## F027 — the ghost branch reference is CORRECT on both parities; the real defect is the near-axis non-decaying ghost, ungated since 8h-d1 retired the decay condition (2026-07-28)

Three hypotheses died here, two of them the driver's. Sweeps stated per claim.

**1. The sqrt-branch reference is parity-independent.** `geometry.
_branch_pinned_amplitude` resolves the `+/-` ambiguity of `1/sqrt(det H_c)` by
matching phase to `reference_amplitude = exp(-0.5j*pi)`, justified in the
docstring by "the two real images continue into a Morse-index-1 saddle" — a
positive-parity phrasing. It is nonetheless CORRECT for `det A < 0`:
`tr Hess = 2*lam > 0` forbids index-2 images on BOTH branches, so an (1,2)
merge is impossible and every A2 fold annihilates an (index 0, index 1) pair.
Measured radial scans across the caustic (theta = 0.30, 36 radii): saddle
`gamma = 1.60` and `1.20` both go `0111` inside -> closest pair at the fold is
(0,1) -> `11` outside, the same annihilating pair as positive parity.

Confirmed numerically (30 log-spaced `w` in [0.3, 6], saddle `gamma = 1.2`,
`|y| = 3.05`, `theta = 0.30`, `Im tau_c = 0.919`, `Delta_tau = 16.25`,
separation 1.57): `-G` is NEVER best at any `w`; `+G` wins at all 16 points
with `w <= 1.41`. The DRIVER'S claim that the subtraction is "actively wrong"
on the saddle was FALSE.

**2. The driver's fallback hypothesis — an admission-set discontinuity inside
the tile — was ALSO false.** Tested on a theta interval with no boundary
crossing (`theta in [0.1, 0.7]`, 61 samples, all admitted): `ppGO + G` is still
7x to 64x worse than ppGO alone, while NODE COUNTS BARELY MOVE (4->5, 5->6,
4->4, 11->10). The damage is to residual SIZE, not to splineability, so it is
not a tiling problem.

**3. The actual mechanism: the near-axis ghost stops decaying.** As the source
approaches a principal axis `Im tau_c -> 0` (stated in `ghost_kernel`'s own
docstring: "the on-axis ghost is pure oscillation with no decay"), so the ghost
no longer falls off with `w` and swamps a ppGO residual that has become tiny.
Measured at `w = 5`, `|y| = 4.2426`, theta from 0.90 down to 0.02:

| gamma | Im tau_c | \|G\|/\|F\| | r(ppGO) | r(ppGO+G) |
|---|---|---|---|---|
| 1.60 | 2.14 -> 0.099 | 5.2e-6 -> 1.04e-1 | 3.3e-4 | 1.03e-1 |
| 1.20 | 3.15 -> 0.139 | 1.9e-8 -> 4.3e-2 | 3.5e-5 | 4.33e-2 |
| 0.45 POSITIVE | 9.51 -> 0.394 | 3.0e-22 -> 1.44e-2 | 1.5e-5 | 1.44e-2 |

At `gamma = 0.45, theta = 0.02` admitting the ghost is 1000x WORSE — on the
POSITIVE-parity branch, in code shipped by `31ee133`.

**THIS IS A REGRESSION WE INTRODUCED.** Build 8h-d1 re-keyed the ghost gate
from decay (`w_min * Im tau_c >= _FARFIELD_WINDOW_RADIANS = 2.0`) to geometric
separation (`min_a |x_a - x_c| >= _GHOST_SEPARATION_MIN = 0.7`) to kill
train/serve skew. The two gates are ORTHOGONAL: separation guards near-cusp
coalescence, decay guards the near-axis non-decaying ghost. Retiring the decay
condition left the near-axis case ungated on BOTH branches. At
`gamma = 1.6, theta = 0.02, w = 5` the retired gate gives `5 * 0.099 = 0.497`
-> refuse, which is correct.

Measured separation on the saddle NEVER binds: `min_a |x_a - x_c| in
[0.942, 2.421]` over 121 theta-samples x 4 (gamma, |y|) configs, always above
0.7. (This also corrects F024's "43-54 of 65 admitted", which was the
EXISTENCE boundary — no complex-conjugate pair — not the separation gate.)

**The fix keeps both goals.** Pin `w_min` to the CHART BAND FLOOR — a property
of the chart, identical at train and serve — rather than to whichever `w` grid
each side happens to hold. That restores decay protection without
reintroducing the skew that motivated the retirement. Re-admit the ghost gated
on band-floor decay AND separation together.

**Tile alignment, if wanted:** the operative boundary is the ghost EXISTENCE
locus (where the conjugate pair merges onto the real axis) — a level set of the
image quartic's discriminant, computable in closed form from the coefficients
`geometry.image_quartic_coefficients` already returns. Clean: exactly ONE flip
per 121-sample theta sweep, at `theta ~ 1.06` (1.6, 3.05), `1.09` (1.6, 4.24),
`1.32` (1.2, 3.05), `1.31` (1.2, 4.24).

**Interim status.** Refusing the ghost remains SAFER than admitting it under
the separation gate alone: the near-axis failure is 1000x in the wrong
direction and silently biases lnL, whereas refusing costs only the modest
`w <~ 1.4` improvement. So the saddle build's BEHAVIOUR is fine; its stated
JUSTIFICATION ("underived branch reference") is wrong and the comment must be
corrected. The same decay gate is owed on positive parity regardless.

**Pattern.** An orthogonal guard was retired as a duplicate. Both gates
mentioned the ghost, both looked like admission criteria, and one was replaced
by the other on the strength of that resemblance. Before retiring a guard, ask
what it refuses that the replacement does not — here, a whole boundary of
parameter space that neither the tests nor the census was measuring.

## F028 — the uniform fold Airy arm SERVES wrong values on the positive-parity path: its `xi`-only certificate cannot see distance from the caustic, and `q = 0` cannot represent an asymmetric fold (2026-07-28)

Found while trying to measure C6. Four things, in the order they were forced.

**1. `F_op` is NOT an independent oracle above `w = 60`.** `_positive_parity_grid`
(line ~1524) and `_saddle_grid` (line ~957) both hand every `w >
W_CEILING_SCHWINGER` node to `_uniform_arm_value` — the same fold Airy arm.
A first attempt to measure `|F_op - F_Airy|` returned IDENTICALLY `0.0` in
every cell because both sides were the same call. Any future accuracy claim
about the uniform arms must pin `w <= 60` (quadrature) or use
`geometric_amplification`, never `F_op`.

**2. The two routing rules differ, and the positive-parity one has no geometric
branch.** `_saddle_grid` offers the arm only when the node is UNRESOLVED
(`w * delta_min < RHO_END`); `_positive_parity_grid` offers it to EVERY node
above the ceiling. So on the positive-parity path — the whole sampled prior box
— a perfectly resolved config at `w > 60` is served by the Airy arm rather than
by geometric optics, which is exact to `1e-5` there.

**3. Measured: the served value is wrong by O(1).** Positive parity, off-cusp
ray `t = 0.55`, oracle `geometric_amplification` (cross-checked against the
Schwinger quadrature at `w = 45..60`, agreeing to `1e-5`–`1e-4`):

| gamma | eta/R_c | w | w*Dtau | \|F_arm/F_geo\| | rel err |
|---|---|---|---|---|---|
| 0.70 | 0.40 | 70 | 35.2 | 0.348 | 7.5e-1 |
| 0.70 | 0.40 | 500 | 251.6 | 1.846 | 2.7e+0 |
| 0.90 | 0.40 | 500 | 564.2 | 0.192 | 9.4e-1 |
| 0.90 | 0.20 | 500 | 176.4 | 1.715 | 2.7e+0 |

The error does NOT shrink with `w` — it grows and oscillates. This directly
contradicts `_airy_fold`'s module docstring ("the large-`xi` limit reproduces
`geometry`'s exact geometric two-image sum by construction"). Meanwhile the
arm's own certificate reads `1.2e-2`–`4.7e-2`, passing the `envelope_bar = 0.05`
gate: it is optimistic by 20x–100x.

**4. Mechanism — `q = 0` is a SYMMETRIC-fold assumption, not a leading-order
truncation.** `fold_amplification` sets the `Ai'` amplitude `q` to zero (its
docstring calls this "the pure-phase symmetric-fold result"). With `q = 0` the
served form is a single `Ai(-xi)` term, whose large-argument limit is ONE
sinusoid of fixed amplitude. The true two-image sum has two independent complex
amplitudes, equal only when the merging pair has equal magnification — i.e.
only AT the caustic. Away from it the fold is asymmetric and no choice of `p`
alone can represent it, so the error is O(1) however large `xi` becomes.

**Why the certificate misses it.** `xi = (3 w DT / 4)**(2/3)` is large in TWO
different regimes: deep in the asymptotic limit near the caustic (where the arm
is valid) and far from the caustic at any `w` (where `DT` is large and the fold
normal form has broken down). `xi` alone cannot distinguish them. The missing
ingredient is exactly COVERAGE_DESIGN C6's caustic-relative distance `eta/R_c`:
admission must bound the fold ASYMMETRY, which `xi` does not measure.

**Corollary — the arm's certifying set and its serving set are nearly
disjoint.** Random sweep, 4000 draws with a merging fold pair, saddle routing
rule: 906 were routed to the arm and only 2 (0.22%) certified; a further 712
certified where the routing never calls them. So on the saddle path the uniform
rung is very nearly dead code, while on the positive-parity path it is live and
wrong.

Probes: `probe_c6_window.py`, `probe_arm_reachable.py` (scratchpad).

## F029 — the geometric branch's residual error is controlled by DISTANCE TO THE CAUSTIC, not by delay resolution; the existing ghost primitive does not repair it (2026-07-29)

Companion to F028, found while auditing the gate F028's fix routes through.
`select_branch` admits a node to geometric optics on `w * delta_min >= RHO_END`
AND `L > L_MAX`. Neither term measures distance to the caustic, and that is
what the residual error tracks.

Driver sweep, 1200 positive-parity draws at `w in [55, 60]` (below
`W_CEILING_SCHWINGER`, so `F_op` is the Schwinger quadrature and a legitimate
oracle — above the ceiling it would serve THROUGH the uniform arm, F028),
restricted to nodes the authoritative gate ADMITS. Error of
`geometric_amplification` vs the quadrature, binned by `eta` = distance to the
caustic:

| eta | n | median err | max |
|---|---|---|---|
| < 0.02 | 5 | 1.02e+0 | 7.4e+1 |
| 0.02–0.05 | 15 | 4.96e-1 | 2.5e+0 |
| 0.05–0.1 | 22 | 2.55e-1 | 5.1e-1 |
| 0.1–0.3 | 143 | 1.82e-3 | 1.8e-1 |
| > 0.3 | 1015 | **2.08e-7** | 8.3e-3 |

Five orders of magnitude, monotonic. What it is NOT:

* NOT the oracle degrading. Error does not track `L` — outliers appear in every
  `L` bin including `L > 100` (`L` outlier median 60.8 vs 96.8 for the rest).
  The `L ~ 45-46` Schwinger accuracy limit (F005) is not the mechanism.
* NOT a delay-resolution deficit. `w * delta_min` barely separates (outlier
  median 208 vs 261), which is why raising that floor from `RHO_END` to 100 did
  not move the tail at all. This REFUTES the reviewing Professor's hypothesis
  of a near-degenerate non-merging delay pair.
* NOT a near-critical magnification divergence. Every outlier has exactly 2
  real images (outside the caustic) with modest `|mu|max` of 1.1–8.1.

**Mechanism.** Just outside a fold, the two images that annihilate AT the
caustic have become complex saddles with small `Im tau_c`. Measured at the
outliers: `Im tau_c = 1e-4 .. 5.6e-3`, so at `w ~ 56` the damping
`exp(-w Im tau_c)` is ~1 — the complex saddles are entirely undamped, and a
real-image-only sum is missing an O(1) contribution. Same near-axis
non-decay as F027, reached from the caustic side.

**The existing ghost primitive does NOT fix it.** Adding
`channels.farfield_ghost_term` (frames aligned, absolute-frame reconstruction)
to the geometric sum on the six worst configs: `|G|/|F|` came out 0.94 to 21.5,
the correction HELPED 3 and made 2 substantially worse (3.45 -> 18.1), and the
single worst config was refused outright by the `_GHOST_SEPARATION_MIN` gate. A
"correction" tens of times the signal means the single-saddle expansion is
itself out of validity there. The near-caustic region needs the UNIFORM (Airy)
form, not a ghost term added to geometric optics.

**Symmetry with F028, and the design consequence.** The fold arm is admitted
FAR from the caustic where it is invalid (its `xi` certificate cannot see
distance); geometric optics is admitted NEAR the caustic where it is invalid
(its `w*delta_min`/`L` gate cannot see distance either). Both failures have one
cause: no admission term measures distance to the caustic. This is
COVERAGE_DESIGN Part 0 ("no absolute length may appear where the only scale is
the caustic") arrived at a third independent time.

SCOPE: measured on POSITIVE PARITY ONLY. No saddle sweep exists; do not
generalise these numbers to `det A < 0`. `_certify_geometric_census` passes
every one of these points, so it is not a guard against this.

Probes: `probe_geo_gate.py`, `probe_tail_origin.py` (scratchpad).

## F030 — the test suite has NO valid oracle in the regime where the uniform arms actually serve (2026-07-29)

Found while consolidating duplicate routing pins. Explains why F028 survived
undetected and why the suite is structural rather than value-based above the
Schwinger ceiling.

**The chain.** Above `W_CEILING_SCHWINGER = 60` the production exact evaluator
(`_schwinger.f_schwinger`) refuses. The suite therefore had no production path
to compare against and fell back on two things that cannot fail for a wrong
value: asserting WHICH RUNG served the node, and byte-identity against
production. The latter is circular by construction — `F_op` serves THROUGH the
arm above the ceiling (F028), so `F_op == arm` holds however wrong the arm is.

**The near-miss.** `test_lensing_batched_operator._oracle_fop` is an
independent mpmath operator-series reconstruction with no FREQUENCY ceiling,
so it looked like the missing oracle. It is not, and the difference matters:
it is the LEGACY operator series demoted in Build 8d precisely because it
cancels catastrophically at high `L = w * |y'|` (F005: certified to
`L ~ 25-30`, certified-or-refused through 48).

**Measured 2026-07-29.** At the F028 configs (`gamma = 0.70, 0.90`,
`eta/R_c = 0.40`, `w = 70..100`, so `L ~ 100-200`) BOTH the uniform arm and
the geometric serve report relative error exactly `1.000e+00` against
`_oracle_fop` — the oracle is the outlier, not the thing under test. So the
one candidate reference the suite owns is invalid exactly where the arms live.

**The mechanism is TRUNCATION, not cancellation** (a first write-up of this
finding said cancellation; that was wrong and is corrected here). Measured
`max|term| / |total|` is only `0.61` and `0.77` at those two configs — no
catastrophic cancellation at all. What happens instead is that the operator
series never satisfies its own convergence criterion within
`ORACLE_MAX_ORDER = 100`, and the loop then FELL OUT and returned the
truncated partial sum as though it were a reference. There was no
convergence flag, so no caller could tell a converged answer from a
truncated one.

**Fixed.** `_oracle_fop` now raises `OracleConvergenceError` instead of
returning a truncation — the certified-or-refuse contract the production
code obeys everywhere and the test oracle did not. An oracle that fails
silently is worse than no oracle: it converts "we cannot check this" into a
confident false comparison. Blast radius measured before the change: 1 node
of 39 across the two production bands (`CERT_LS` 17/17 converged,
`XOR_BAND_LS` 21/22, only `L = 59.4` truncating).

**Consequence.** The uniform arms are UNFALSIFIABLE by the current suite in
their own serving regime. Every accuracy claim about them is additionally
gated behind `COGWHEEL_BRUTE_ACCURACY`, which by policy never runs in a build
— so nothing that runs in a build has ever compared an arm to truth.

**What IS gated now.** `test_served_band_values_match_the_oracle_above_the_
ceiling_too` gates the above-ceiling GEOMETRIC serve at `L in [24, 59.4]`
against `_oracle_fop` (worst measured 3.7e-5 against a 1e-3 gate). That is a
consistency gate between two independent reconstructions at moderate `L`, and
it stops the routing regressing silently. It does NOT reach the arms.

**What is still owed, and what it is.** A reference valid at `L ~ 100-200`.
The Schwinger quadrature refuses there by frequency; the operator series fails
to converge there.

THE DESIGNATED ANSWER IS GLoW, already ruled on by the driver and recorded in
`.claude/handoff/lensing/META_PLAN.md`: *"GLoW is a cross-oracle only, NOT a
qd replacement"*. It is the right instrument because it is genuinely
INDEPENDENT — every oracle this suite owns is a re-implementation of the same
operator reduction, whereas GLoW builds the time-domain `I(tau)` and
transforms to frequency. That difference favours us: in the time-domain
picture the high-`w` behaviour is set by the singularity structure of
`I(tau)`, so GLoW is strongest exactly where we are blind.

STATUS. GLoW has been evaluated before on this project (owner, 2026-07-29) and
found to WORK for the positive-parity image case with a LARGER RADIUS OF
CONVERGENCE than either in-repo reference. That is exactly the needed
combination: every arm defect measured so far (F028, F029, F031) is positive
parity, and the larger convergence radius reaches the `L ~ 100-200` regime
where the quadrature refuses by frequency and the operator series fails to
converge.

The positive-parity restriction is STRUCTURAL, not a configuration choice:
Chang-Refsdal is not axisymmetric, so it needs `time_domain.It_SingleContour`,
which follows a single contour around the MINIMUM and therefore cannot see the
macro saddle. The saddle stays unoracled — the same gap already standing in
F028/F029/F031.

Cloned to `/home/tejaswi/Work/GLoW` (github.com/miguelzuma/GLoW_public — note
the repo is `GLoW_public`, not `GLoW`). Imports cleanly into the project env;
the pure-Python path needs only numpy/scipy. Chang-Refsdal is expressible as
`lenses.CombinedLens({'lenses': [Psi_PointLens(), Psi_Ext({kappa, gamma1,
gamma2})]})`. GLoW's own `shear()` docstring defines
`det A = (1-kappa)**2 - gamma**2`, matching this repo's convention.

ATTEMPTED 2026-07-29, NOT YET WORKING. Recorded in detail so the next attempt
does not repeat it.

WHAT IS ESTABLISHED:
* Install: `github.com/miguelzuma/GLoW_public` (the repo is `GLoW_public`, NOT
  `GLoW`), cloned to `/home/tejaswi/Work/GLoW`. Pure Python needs only
  numpy/scipy. The C wrapper BUILDS: system GSL 2.5 is present, and
  `pip install cython` into the env, then
  `make -C wrapper/glow_lib -j4 && <env python> wrapper/setup.py build_ext
  --inplace`. Do NOT run the top-level `make` -- it hardcodes system `python3`,
  which has no numpy.
* Chang-Refsdal IS expressible:
  `lenses.CombinedLens({'lenses': [Psi_PointLens(), Psi_Ext({kappa, gamma1,
  gamma2})]})`. VERIFIED: despite `class CombinedLens(PsiAxisym)`, the shear
  survives the combination -- `psi` varies with polar angle by 1.92e-1 at
  `|x| = 0.8, gamma = 0.3`, exactly matching `Psi_Ext` alone. The MRO is
  `CombinedLens -> PsiAxisym -> PsiGeneral`; `PsiAxisym` is a helper base, not
  an assertion of symmetry.
* GLoW's own `shear()` docstring defines `det A = (1-kappa)**2 - gamma**2`,
  matching this repo.

WHAT IS NOT: no It/Fw method pair has yet produced a usable number for this
lens. Tried, with the exact failure of each:

| It method | Fw method | outcome |
|---|---|---|
| `It_SingleContour` (py) | `Fw_FFT_OldReg` | ANSWERS SILENTLY AND WRONGLY -- see below |
| `It_SingleContour_C` | `Fw_DirectFT_C` | refuses: "More than one critical point found" |
| `It_MultiContour_C` | `Fw_DirectFT_C` | NaN; "could not find bracket for R(tau)", GSL Bessel domain errors |
| `It_AreaIntegral_C` | `Fw_DirectFT_C` | refuses: "no critical points (p_crits) found in It" -- the Fw method needs data this It does not produce |

THE GOTCHA WORTH THE WHOLE EXERCISE: the PURE-PYTHON `It_SingleContour`
returns confident values for a lens it cannot represent, where the C
implementation REFUSES BY NAME on identical input. Its output looked
plausible (|F| ~ O(10)) and moved with `Nt` (14.6 -> 29.0 at `w = 70` between
`Nt` 500 and 2000), which reads as an under-resolved transform and is
actually an invalid method. A convergence sweep chasing that is wasted work.
Same class as this repo's own `_oracle_fop` truncation bug fixed the same day:
a reference that fails silently converts "cannot check" into a confident false
comparison.

CONSEQUENCE: an earlier F009 "agreement" at 8e-4 was computed through
`It_SingleContour` and is therefore SUGGESTIVE ONLY -- right lens, invalid
method. It must be re-established on a working method before any anchor is
trusted.

ROOT CAUSE FOUND (2026-07-29), and it is OURS, not GLoW's.

The correct pairing is `It_MultiContour_C` + `Fw_FFT_C` (owner: Codex got
Chang-Refsdal working with it; the contour warnings are spurious). With that
pairing the contour is CLEAN -- `It_grid` contains zero NaN at every shear
tested -- but `Fw` still returned NaN. The discriminator is `tmin`, the
minimum of the Fermat potential:

| gamma1 | tmin | It NaN | Fw(0.1) |
|---|---|---|---|
| 0.000 | +0.057339 | 0 | 1.0701 - 0.1363j |
| 0.050 | +0.015004 | 0 | 1.0719 - 0.1314j |
| 0.080 | -0.011755 | 0 | **nan** |
| 0.200 | -0.130910 | 0 | **nan** |

`Fw` is NaN EXACTLY when `tmin < 0`, with a clean sign crossing between
`gamma1 = 0.05` and `0.08`. External shear drives the Fermat minimum negative,
and GLoW's time grid is LOG-sampled (`t range [0.01, 1e6]`, `sampling: 'log'`),
which cannot represent a domain whose origin is negative.

THAT HYPOTHESIS IS REFUTED. A shear-SIGN sweep breaks the correlation: at
`gamma1 = -0.20` (`tmin = +0.204`) and `gamma1 = -0.05` (`tmin = +0.097`) the
transform still returns NaN despite a POSITIVE `tmin`. The clean crossing seen
earlier was an artifact of sweeping positive `gamma1` only, where `tmin` and
shear magnitude move together. `tmin < 0` is a symptom, not the cause; do not
build a potential-offset fix on it.

WHAT ACTUALLY WORKS, AND THE FIRST CROSS-CODE NUMBER: at
`gamma1 = +0.05, |y| = 0.5, w = 5` on the `It_MultiContour_C` + `Fw_FFT_C`
pairing, GLoW returns `|F| = 0.526053` against cogwheel's `0.528894` --
**agreement to 0.5%**. This is the first genuinely INDEPENDENT validation of
this engine (different code, different authors, time-domain rather than
frequency-domain), and it validates the conventions too: a convention error
would not land within half a percent.

THE REMAINING LIMIT IS SHEAR MAGNITUDE, not `tmin` and not the method. GLoW
succeeds at `|gamma1| = 0.05` (positive only) and returns NaN at
`|gamma1| = 0.20` and at `gamma1 = -0.05`. F028's configs need
`gamma = 0.7-0.9`, far outside the working range found so far. So F028 remains
confirmed only against geometric optics.

MECHANISM IDENTIFIED (owner relayed the lead from Codex: the point-lens
regularization). `Psi_PointLens` carries `p_prec = {'xc': 1e-10}` --
"Point mass regularization (Plummer sphere)" -- effectively zero by default.
Sweeping it surfaced the real error, which the NaN had been hiding:

    ERROR: even number of centers found (n=2) [init_all_Center:62]
    ERROR: initialization of centers was unsuccessful [find_birth_death:1154]

GLoW's contour initializer requires an ODD number of images (the odd-number
theorem). A PURE point mass violates it numerically: the central image is
infinitely demagnified and absent, so outside the caustic GLoW counts 2 and
refuses. A Plummer sphere with finite `xc` restores that central image and
makes the count odd -- which is exactly why the regularization is the knob.

NOT YET SOLVED. Sweeping `xc` over 1e-10 .. 1e-2 at `|y| = 0.5, w = 5`:
`gamma = 0.05` works (ratio 0.995 vs this engine) except at `xc = 1e-3`,
where the same even-count error fires; `gamma = 0.20` and `0.70` fail at
EVERY `xc` tried. So the mechanism is right but the working `xc` for
`gamma >= 0.2` was not found, and by `1e-2` the softening is large enough
that it is no longer a point mass -- `xc` cannot simply be raised without
bound. The usable value must be finite AND on a plateau where `|F|` stops
depending on it; no such plateau was reached above `gamma = 0.05`.

OPEN QUESTION for whoever continues: what `xc` (or additional
`It_MultiContour_C` setting) makes the central image resolvable so the count
is odd at `gamma >= 0.2`? Note the failure is NOT monotonic in `xc`
(`gamma = 0.05` fails at 1e-3 but works at 1e-4 and 1e-2), which suggests
the centre-finding is sensitive to resolution rather than to the softening
scale alone.

This also explains the earlier partial successes: the pure-Python
`It_SingleContour` path agreed with F009 at `w -> 0` because
`F(w->0) -> sqrt(mu_macro)` is invariant under a shift of the delay ORIGIN,
and fell apart at high `w` where the phase `exp(i*w*tau)` is not.

CORRECTION to an earlier version of this entry: `It_MultiContour_C` was
recorded as "NaN, broken". That was wrong -- the contour half was always fine;
it had been paired with `Fw_DirectFT_C`, which cannot consume it, and then run
with a negative `tmin`. Method blamed for a convention bug.

STILL TO COLLECT: apply the offset, re-run the F009 gate on the working
pairing (the earlier 8e-4 agreement was measured through an invalid method and
does not carry over), then the F028 anchors.

Design when it is done, so it does not become a runtime dependency or a new
slow tier: use GLoW OFFLINE to generate a small set of anchors at the
F028/F029 configs (high `L`, above the ceiling, near AND far from the
caustic), and freeze them as literals with provenance — the frozen-anchor
idiom the suite already uses. That converts
`test_lensing_airy_fold::test_served_arm_accuracy_is_unverified_pending_an_oracle`
from a documented `@expectedFailure` into a real gate on the uniform arms.

CONVENTIONS FIRST, before any arm comparison. Cross-code agreement is worth
nothing until `w`, the Fermat-potential normalization, the `kappa`/mass-sheet
convention, the phase sign, and the magnification normalization are aligned —
this repo has been bitten by convention drift repeatedly (the delay frame at
four sites, the ghost frame, IMRPhenomXP vs Pv2). F009 is the free check:
`F(w->0) = sqrt(mu_macro) = 1/sqrt((1-kappa)**2 - gamma**2)`. If GLoW
reproduces that closed form across several `(gamma, kappa)`, the conventions
are aligned; if it does not, no high-`w` agreement means anything. OPEN: GLoW's
strongest support is axisymmetric lenses, and Chang-Refsdal (point mass +
external shear + convergence) is not axisymmetric — confirm it can express
this lens before relying on it. If it cannot, the `gamma -> 0` point-lens
limit is still cross-checkable but does not reach the arms.

Other candidates, if GLoW cannot express the lens: direct high-dps numerical
evaluation
of the diffraction integral at a handful of anchor configs, or a stationary-
phase-plus-correction reference with an independently bounded remainder. Until
one exists, any statement that the uniform arms are accurate is unverified —
see F028 for what was measured when geometric optics was used as the stand-in
reference.

## F031 — `L_MAX = 48` is a genuine geometric-onset threshold, but it is HALF the gate: the missing term is distance to the caustic (2026-07-29)

`L_MAX` was calibrated as a proxy for the LEGACY operator series' accuracy
ceiling (F005). That series no longer serves anything, so the threshold needed
re-deriving on its own terms or retiring. It survives — and the same sweep
identifies what has to sit beside it.

Driver sweep, 2600 RESOLVED positive-parity samples at `w in [5, 60]` (the
band where the Schwinger quadrature is a legitimate oracle), median relative
error of `geometric_amplification` vs the quadrature:

| eta band | L 0-20 | L 20-35 | L 35-48 | L 48-70 | L 70-120 |
|---|---|---|---|---|---|
| 0-0.1 | 5.27e-1 | 4.53e-1 | 3.83e-1 | 6.07e-1 | (n=6) |
| 0.1-0.3 | 7.77e-2 | 3.21e-2 | 2.17e-2 | 4.79e-3 | 7.93e-4 |
| 0.3-1 | 2.77e-4 | 3.84e-5 | 1.12e-5 | 3.32e-6 | 1.00e-6 |
| 1-inf | 3.52e-5 | 6.29e-6 | 1.68e-6 | 6.57e-7 | 1.76e-7 |

**1. `L` is a real onset variable.** At FIXED `eta`, error falls monotonically
with `L` — 100x to 280x across the range. This is measured against the
quadrature, with no reference to the operator series, so the threshold no
longer depends on the retired path for its meaning. `L_MAX = 48` stays.

**2. `L` alone is not sufficient, and this IS F029's tail.** The `L > L_MAX`
leg buys, per eta band (p90):

| eta | L<=48 | L>48 | gain |
|---|---|---|---|
| 0-0.1 | 2.10e+0 | 1.17e+0 | 1.8x |
| 0.1-0.3 | 2.90e-1 | 5.62e-2 | 5.2x |
| 0.3-1 | 1.17e-2 | 7.65e-5 | 153x |
| 1-inf | 8.19e-5 | 1.54e-6 | 53x |

At `eta < 0.1` the row is FLAT in `L` and the gate still admits nodes with p90
= 1.17 — 117% error. No amount of `L` rescues the near-caustic regime, because
geometric optics has no validity there (F029: just outside a fold the
annihilated pair are undamped complex saddles the real-image sum omits). The
~1% O(1) tail F029 measured and could not localise is exactly this population.

**Measured gate.** `L > L_MAX` AND `eta >~ 0.3` takes worst-case p90 from
1.17 to 7.65e-5 — four orders of magnitude.

**IMPLEMENTED** (`4318dab`, 2026-07-29). `select_branch` gains the `eta` leg
described above (`ETA_MIN_GEOMETRIC = 0.3`), wired into both operator grids
(`_positive_parity_grid`; `_saddle_grid` passes `eta = inf`, positive-parity
only per the caveat below) and into `channels._exact_total`'s call site,
which had been missed on first wiring -- the same shape as F028: the
training-label path and the serving path went out of sync until the third
caller was found and fixed. This is a REFUSAL-INCREASING change: nodes
failing the eta leg fall to the uniform arms (wrong, F028) or to a named
refusal, trading coverage for correctness -- an owner-endorsed trade, not a
cleanup. Two caveats remain live: the sweep is POSITIVE PARITY only (no
saddle data, same gap as F028/F029), and it is measured below the Schwinger
ceiling then extrapolated into the above-ceiling regime where the gate
actually runs, because no oracle exists there (F030).

Probe: `probe_lmax_rederive.py` (scratchpad).

## F032 — F028 CONFIRMED by an independent code: the uniform fold arm is 60-64% wrong where GLoW can adjudicate (2026-07-29)

F028 measured the uniform fold Airy arm at 60%-267% relative error using
`geometric_amplification` as a stand-in reference, because no true oracle
existed in that regime (F030). GLoW now provides one, and it agrees.

**Working recipe** (owner + Codex; every element was necessary):
* `Psi_PointLens({'psi0': 1.0}, {'xc': 1e-4})` INSIDE the caustic. The
  `xc = 2e-3` value is the OUTSIDE-caustic workaround and produces a spurious
  six-centre topology here.
* Break exact source/shear axis alignment: at `gamma2 = 0` the two off-axis
  saddles share a delay and GLoW's birth/death bookkeeping fails. A rotation
  of `beta ~ 0.015` rad at fixed `|gamma|`, folded into `(gamma1, gamma2)`,
  is enough.
* `It_MultiContour_C(lens, y, p_prec={'Nt': 600, 'tmin': 1e-3,
  'tmax': 80.0, 'parallel': False})`
* `Fw_FFT_C(it, p_prec={'wmin': 0.1, 'wmax': ..., 'eval_mode': 'exact',
  'parallel': False})`
* HEALTH CHECK, mandatory: the centre list must be
  `['min', 'min', 'saddle', 'saddle', 'sing/cusp max']` -- FIVE, odd. An even
  count means initialization failed, and GLoW RETURNS VALUES ANYWAY. A prior
  run read a verdict off six-centre output before this check existed.

**Result.** `|F|` compared (frame-independent; GLoW is in the absolute delay
frame, this engine is min-subtracted, related exactly by `exp(i*w*tmin)` --
verified to five figures).

| gamma | w | geo err | arm err |
|---|---|---|---|
| 0.70 | 10 | 3.0e-3 | arm refuses |
| 0.70 | 70 | 2.3e-2 | 6.4e-1 |
| 0.90 | 10 | 5.6e-3 | arm refuses |
| 0.90 | 30 | 7.1e-2 | 6.3e-1 |
| 0.90 | 50 | 1.9e-1 | 6.4e-1 |

Geometric optics tracks GLoW to sub-percent at low `w` and 2.3% at `w = 70`;
the arm is 63%-64% off throughout. Independent, and consistent with F028's
original magnitude and sign. **The `ETA_MIN_GEOMETRIC` floor (F031) was the
right call, and the arm's `q = 0` defect is real.**

**SCOPE.** Positive parity only, two configurations, one external code.

The `w <= 70` cap originally stated here is SUPERSEDED by F035: the apparent
high-`w` degradation was not GLoW's, it was the `beta_break` symmetry
workaround making GLoW evaluate a slightly rotated lens. With `beta_break`
extrapolated toward zero the agreement holds to 2.4% at `w = 100` and 6.1% at
`w = 200` and keeps improving. The `w = 200-500` rows quoted in the original
verdict table were computed at a single `beta_break = 0.015` and carry its
phase error; do not cite those NUMBERS, but the confirmation itself is not
confined to `w <= 70`.

**Established along the way.** GLoW agrees with this engine to ~1e-4 in `|F|`
at `w = 0.5..50` outside the caustic, and `xc` has a convergence plateau at
the 1e-4 level across a 4x sweep -- far below the effect under test. This is
the first independent validation this engine has ever had.

## F033 — the fold arm's far-field error is the CUBIC NORMAL FORM's O(eta) truncation, not `q = 0`; the b4 refinement cannot fix it (2026-07-29)

F028's todo listed two routes to repairing the uniform fold arm: derive the
`b4` quartic refinement of the `Ai'` amplitude `q`, or fence the arm to where
the fold is near-symmetric. This measurement closes the first.

**The test.** The CFU uniform form wants both amplitudes of the merging pair:

    p = (s_+ + s_-)/2 * w^{-1/6} * xi^{ 1/4}
    q = (s_- - s_+)/2 * w^{ 1/6} * xi^{-1/4}

with `s_a = sqrt|mu_a|`. Production instead builds `p` from the FINITE
cubic curvatures (hard eigenvalue and soft-axis `b3`) and sets `q = 0`. If
`q = 0` were the whole defect, the two `p` values would agree and only `q`
would be missing.

**They do not agree, and the disagreement is structured.** Sweeping `eta` at
FIXED `gamma` (w = 100):

| eta (gamma=0.70) | p_CFU / p_prod | eta (gamma=0.90) | p_CFU / p_prod |
|---|---|---|---|
| 0.0147 | 1.0069 | 0.0195 | 1.0135 |
| 0.0735 | 1.0348 | 0.0977 | 1.0709 |
| 0.1470 | 1.0702 | 0.1954 | 1.1509 |
| 0.2941 | 1.1427 | 0.3907 | 1.3437 |
| 0.5659 | 1.3292 | 0.7814 | 1.8781 |

Monotonic, `-> 1` as `eta -> 0` (0.7% at eta = 0.015), and
`ratio - 1 ~ 0.5*eta` (gamma = 0.70), `~ 0.7-1.1*eta` (gamma = 0.90).

**Conclusion.** The amplitude convention is correct -- the two agree wherever
the cubic normal form is valid. What fails away from the caustic is the
NORMAL FORM ITSELF, at O(eta), the first neglected order. `p` is wrong there
by the same mechanism as `q`, so setting `q` from `b4` cannot recover the
far-from-caustic region. THE ETA FENCE IS THE CORRECT PERMANENT TREATMENT,
not a stopgap, and the `b4` derivation should NOT be undertaken for this
purpose.

**Consequence for the fence threshold, and it is not comfortable.**
`_ETA_MAX_FOLD = 0.3` was set as the complement of `ETA_MIN_GEOMETRIC`,
which was measured for the GEOMETRIC branch -- never for the arm. At
`eta = 0.3` the arm's amplitude is already off by 14% (gamma = 0.70) to 29%
(gamma = 0.90). At `eta = 0.1` it is 3%-7%. So the current fence still admits
the arm at tens of percent error. Tightening to ~0.1 is indicated but was NOT
applied here: it needs its own served-error measurement (this is an amplitude
ratio, not a served |F| error) and a coverage cost, exactly as F031 did for
the geometric side.

Probes: `probe_b4_fix.py`, `probe_p_ratio_eta.py` (scratchpad).

## F034 — the MACRO SADDLE needed the eta floor too, and was worse than positive parity (2026-07-29)

F031 measured the eta leg on positive parity and the saddle was given
`eta = inf` -- the leg switched OFF -- because no saddle sweep existed. That
was recorded as an OPEN question rather than a safe default. It was not safe.

Driver sweep, 2000 RESOLVED macro-saddle samples (`kappa = 0`,
`gamma in [1.05, 1.60]`, so `det A < 0`), `w <= W_CEILING_SCHWINGER` so
`F_op` is the Schwinger quadrature. Error of `geometric_amplification`
against it, binned by distance to the caustic:

| eta | n | median | p90 | max |
|---|---|---|---|---|
| 0-0.1 | 87 | 5.69e-1 | 1.68e+0 | **4.84e+2** |
| 0.1-0.3 | 210 | 2.82e-2 | 2.50e-1 | 1.04e+1 |
| 0.3-1 | 927 | 4.52e-5 | 1.10e-2 | 6.86e-1 |
| >1 | 776 | 1.04e-5 | 9.13e-4 | 1.05e-1 |

Applying the positive-parity floor: p90 goes from 8.95e-1 (`eta < 0.3`,
n=297) to 4.54e-3 (`eta >= 0.3`, n=1703) -- a factor of ~200, over 15% of
resolved draws.

Worse than positive parity, where the same band peaked at p90 1.17 with no
484x outlier. `_saddle_grid` now passes the measured `eta`; the cancellation
leg stays vacuous (`inf`), since `cancellation_exponent` is positive-parity
bookkeeping with no saddle analogue.

**The reasoning error worth keeping.** Passing `inf` was justified at the
time as refusing to extrapolate an unmeasured threshold onto another branch
-- which is the right instinct. But "unmeasured, so leave it alone" is only
conservative when the STATUS QUO is known safe. Here the status quo was
serving a 484x error. An unmeasured branch is a queued measurement, not a
defensible default.

Probe: `probe_saddle_eta.py` (scratchpad).

## F035 — GLoW reaches the F028 regime; the high-`w` "disagreement" was the symmetry-break workaround, not physics (2026-07-29)

F032 confirmed F028 but capped its scope at `w <= 70`, because above `w ~ 100`
GLoW disagreed with geometric optics by 28% (w=100) rising to 406% (w=500).
That cap was right to impose and wrong in its stated reason.

**Not a resolution artifact.** `|F|` is stable to <1.5% across `Nt` 600 ->
38400 (64x) and to 4-5 decimals across `wmax` 300 -> 5000 (17x). The values
are numerically solid.

**It was the symmetry break.** GLoW's contour finder needs the exact
source/shear axis alignment broken (F032), so the lens it evaluates is rotated
`beta_break` away from the one handed to `geometric_amplification`. In a
4-image interference pattern that phase error grows as `w * Delta_tau`.
Shrinking it:

| beta_break | rel(w=70) | rel(w=100) | rel(w=200) |
|---|---|---|---|
| 0.030 | 5.78e-2 | 5.89e-1 | 1.52e+0 |
| 0.015 | 2.09e-2 | 2.76e-1 | 8.55e-1 |
| 0.008 | 8.46e-3 | 1.37e-1 | 4.44e-1 |
| 0.004 | 3.26e-3 | 6.10e-2 | 1.91e-1 |
| 0.002 | 1.11e-3 | 2.38e-2 | 6.11e-2 |

Monotonic in every column, roughly `beta**1.4`, still falling at the small
end. At `beta = 0.002`: 0.11% at `w = 70`, 2.4% at `w = 100`, 6.1% at
`w = 200`.

**Consequences.**
1. GEOMETRIC OPTICS IS VINDICATED at high `w` -- it does not degrade, and was
   never the suspect. F031's accuracy claim extends above the Schwinger
   ceiling rather than being extrapolated there on faith.
2. GLoW REACHES the F028 regime. F032's confirmation is not confined to
   `w <= 70`; the cap was a comparison bug.
3. PROCEDURE for any future GLoW comparison: `beta_break` is a systematic,
   not a free knob. Either use the smallest value GLoW tolerates and quote the
   induced error, or run the `beta -> 0` extrapolation. Comparing at a single
   `beta` silently charges its phase error to whichever code is under test,
   and at `w = 70` that error is ~2% -- small enough to read as success.

**Process note.** Codex's recipe arrived with the instruction that the
configuration "needs an explicitly checked `beta -> 0` limit". The frequency
range was pushed without performing that check, and four subsequent probes
(Nt convergence, wmax edge, two verdict runs) were spent chasing a
discrepancy the instruction had already named.

Probes: `probe_glow_highw.py`, `probe_glow_wmax_edge.py`,
`probe_glow_beta_break.py` (scratchpad).

---

## F036 — no `|y|` threshold can bound the caustic: `r_caustic` DIVERGES at the parity wall (2026-07-29)

The coverage design carved the source plane with a constant, `ANNULUS_INNER_RADIUS = 3.0`,
inherited from `_Y_SCALE_CAP` — the PRIOR BOX half-width. That is a sampling
bound used as a physical boundary, and the failure is not that 3.0 is badly
chosen. It is that no value exists.

Directional caustic reach `geometry.r_caustic(gamma, theta)` over the prior
`gamma in (0, 1.6)`, `kappa = 0`:

| gamma | min r_c | max r_c | anisotropy | where `\|y\| = 3` sits |
|---|---|---|---|---|
| 0.50 | 0.500 | 1.414 | 2.8 | outside the caustic |
| 0.70 | 0.700 | 2.556 | 3.7 | outside the caustic |
| 0.90 | 0.900 | 5.692 | 6.3 | **cuts the caustic** |
| 0.99 | 0.990 | 19.800 | 20.0 | **cuts the caustic** |
| 1.02 | 1.020 | 4.682 | 4.6 | **cuts the caustic** |
| 1.30 | 1.222 | 1.714 | 1.4 | outside the caustic |

`r_caustic -> inf` as `gamma -> 1`: the caustic blows up at `det A = 0`. At
`gamma = 0.99` its reach is 19.8, versus a prior box corner of 4.2426 — the
caustic is nearly five times the entire sampled region. ANY fixed `|y|`
threshold is therefore crossed by the caustic somewhere in the prior.

**Consequence for the design.** The four apparent serving regimes are not four
physical situations. They are ONE fixed boundary crossing a caustic whose
extent varies 28x and diverges. In caustic-relative units the relationship is
constant, and there are TWO regimes per parity: caustic-attached (interior +
tube shell) and exterior.

**Second consequence, easy to miss.** The anisotropy column reaches 20x near
the wall, so a SCALAR caustic-relative coordinate is not sufficient either. The
coordinate must stay directional (`rho = |y| / r_caustic(gamma, theta_c)`), as
positive-parity charts already are. `surrogate._caustic_reach(gamma)` is the
scalar conservative-guard path and is NOT a substitute for it.

**The window, and its expiry.** As of this finding every constant in the
COVERAGE_DESIGN Part IV violation table is INERT with respect to any served
value: `ANNULUS_INNER_RADIUS` and `GAMMA_FENCE` are read only by
`surrogate_census` for accounting, the eta/cusp constants are confined to
`surrogate_training`, `likelihood._surrogate_coefficients` returns `None` in
the Born slot so that rung never serves, and no trained chart artifact is
shipped. The coordinate change is therefore a pure-source edit with no
migration, no retraining and no value churn. That is true ONLY until something
is trained. Do not train — not the Born residual chart, not a full-box
surrogate — until the coordinates are final.

**How it happened, since the mechanism matters more than the instance.** The
prior box was chosen first, in coordinates that do not know where the caustic
is; regions were then defined against it; fences were derived from those
regions (`GAMMA_FENCE = 3/4` and the saddle fence `1.0502342` are both
CONSEQUENCES of the annulus radius, not independent physics). Each step was
locally reasonable. The check that catches this class is Part 0 of
COVERAGE_DESIGN: for every length-unit float, ask what sets it, and refuse
"the prior box", "a round number", and "it worked at one gamma".

## F037 — the small-gamma collar is THREE stacked causes, not one; C6 closes only the middle one (2026-07-29)

**Where:** `surrogate_training.stable_gamma_bands`,
`surrogate_training._min_curvature_radius`, coverage-map region 3.

The coverage map recorded the small-gamma near-caustic collar as a single
failure: `eta_max = 0.05` is absolute, the astroid shrinks below it as
`gamma -> 0`, `_min_curvature_radius` skips the tube chart, and the far field
excludes the same collar. Measured on the production path (positive parity,
`n_caustic_samples = 200`, `min_gamma_band = 0.02`, `eta_max = 0.05`), the
collar is actually three different failures stacked:

| gamma range | what happens | cause |
|---|---|---|
| `< 0.0281` | dropped topology sliver | `stable_gamma_bands` cannot find a stable band |
| `0.0281 .. 0.0462` | tube SKIPPED (`r_min = 0.0238`) | foot-of-normal guard |
| `0.0462 .. 0.0644` | dropped topology sliver | `stable_gamma_bands` |
| `0.0644 .. 0.1550` | tube SKIPPED (`r_min = 0.0518`, `0.0648`) | foot-of-normal guard |
| `>= 0.1550` | tube trained (`r_min = 0.1120`) | — |

So tubes serve NO gamma below 0.155 — about 9.7% of the sampled prior
`gamma in (0, 1.6)`.

**Why it matters for the plan.** C6 (curvature-relative `eta_max = f * R_c`)
makes the foot-of-normal guard vacuous by construction and therefore closes
the two SKIPPED rows. It does nothing for the two DROPPED rows: those are a
topology-detection instability, not a length-scale mismatch. An acceptance
reading "the small-gamma collar is closed" is therefore not achievable by C6
alone and must not be written into its brief.

**The sliver instability is not a resolution problem.** The raw
`band_caustic_structure((0.02, 0.07), +1)` failure is
`Arc served side / image count changes across gamma band: [(-1, 4), (1, 2)]`,
and it is IDENTICAL at `n_samples` 200, 800 and 3200. The arc's `inward_sign`
and `image_count` genuinely flip between the band edges as detected; densifying
the sweep does not help, so band bisection recurses to the `min_gamma_band`
floor and drops the sliver. Whatever is wrong is in the served-side detection
at small gamma, not in how finely it is sampled.

## F038 — the caustic curvature radius is a CLOSED FORM; the sampled circumradius estimator was never necessary (2026-07-29)

**SCOPE.** The durable content is the closed form and its verification
envelope. The bias measurement further down describes
`surrogate_training._min_curvature_radius` AS IT STANDS TODAY and expires when
step 1 of [[lensing_caustic_relative_coordinates]] deletes that estimator; it
is kept because it is why the step-1 acceptance is worded as it is, not
because it is open work. Nothing here needs fixing beyond shipping step 1.

**The closed form.** The caustic is an exact parametric curve `y(theta)`, so
its curvature radius is an exact function of `(gamma, theta, kappa, branch)`:
differentiate the closed form, do not sample it. Chain rule through
`u -> r -> y`, then `R_c = |y'|^3 / |y1' y2'' - y2' y1''|`. `beta` is a rigid
rotation and curvature is rotation-invariant, so
`R_c(theta; beta) = R_c(theta - beta; 0)`. Derived 2026-07-29: about 25 lines
of plain numpy, vectorised over `theta`, no new runtime dependency,
numba-compatible.

**The radial weight is `p_i = M_ii - lam*u`, NOT `M_ii - u`.** Componentwise
`y_i = p_i * r * T_i` with `T = (cos, sin)` and `1/r^2 = lam*u`. The two forms
coincide only at `kappa = 0`, where `lam = 1`. At `kappa = 0.3` they differ by
0.19-0.39 in absolute source-plane position — not a tolerance issue, a wrong
curve.

**THE FIRST VERIFICATION OF THIS FINDING WAS CIRCULAR, AND THAT IS THE MOST
REUSABLE THING IN IT.** The original envelope was measured with an mpmath
oracle that RE-TRANSCRIBED the caustic curve from the same (wrong) formula the
implementation used. It therefore checked the DIFFERENTIATION against a shared
CURVE error, agreed to 1e-16, and certified nothing at `kappa != 0`. The error
surfaced only when a build's Coder cross-checked the sign against
`critical_point`'s own `source = macro_matrix @ image - image / radius**2`
instead of trusting the transcription handed to it.

An oracle for a derived quantity must therefore be TWO-STAGE:

    STAGE 1  validate the transcribed curve against the shipping code's own
             output (`critical_point(...).source`) at float64
    STAGE 2  differentiate THAT validated curve at high precision

Stage 1 catches curve errors, stage 2 catches differentiation errors, and
neither can mask the other. "Sharing the curve definition is not circular" is
true ONLY once stage 1 has pinned the definition to shipping code.

**Verification envelope, re-measured against the two-stage oracle** over 110
configs — `gamma` in {0.05, 0.3, 0.9, 0.99, 1.02, 1.3}, both branches, `kappa`
in {0, 0.3}, `theta` in {0.02, 0.17, 0.5, 1.0, 1.3, 2.2, 3.9}: stage 1 agrees
to **5.14e-15**; the analytic cascade then agrees to **4.39e-13 on `y'`** and
**2.56e-14 on `y''`**, ZERO failures at `atol = 5e-13 + rtol = 1e-11`.

Use that MIXED tolerance, never a flat relative one: near-axial `theta = 0.02`
and the saddle `-1` branch legitimately send individual components through
zero, where a pure relative gate false-fails on noise.

**Two oracle-construction traps, both of which go COMPLEX rather than fail
loudly.** `critical_point` CLAMPS a slightly-negative saddle discriminant to
zero, and it IGNORES `branch` at positive parity (only the `+` root is a
positive radius there). An oracle that mirrors neither produces `mpc` values;
an IMPLEMENTATION that mirrors neither produces `nan` from
`sqrt(negative)` at positive parity with `branch = -1`, silently violating the
whole-call refusal contract.

The small-`gamma` astroid limit `R_c -> 3*gamma*|sin 2 theta|` is a further
independent scale-and-sign check, good to 4.4e-6..1.2e-4 at `gamma = 1e-4` and
degrading as `O(gamma^2)`; it is NOT a 1e-12 gate.

**Why this entry exists at all: the method was the bug, not a constant.** The
plan's Part 0 discipline — for every length-unit float, ask what sets it — was
applied here to a METHOD, and the answer was never "a finite difference". The
first draft of step 1 proposed relocating the estimator into `geometry` and
gating it against a symbolic curvature. That inverts the roles: it ships an
approximation and uses the exact answer as its oracle. Owner caught it. The
estimator is DELETED, not relocated.

**What the incumbent does, and why its numbers must not become a gate.** The
estimator takes three-point circumradii over a sampled arc and minimises them.
The minimum of the true curvature radius on a fold arc sits at an arc ENDPOINT
(curvature is worst toward the trimmed cusp windows), and a three-point
stencil's first usable centre is one sample step inside the endpoint. So it
reports the curvature one step in from where the minimum lives — biased HIGH,
converging only at FIRST order in the sample spacing:

| samples over a quarter-arc | rel. excess over exact |
|---|---|
| 100 | 30.2% |
| 200 | 14.9% |
| 400 | 7.4% |
| 800 | 3.7% |

On PRODUCTION arcs (cusp windows already trimmed by `band_caustic_structure`,
so the endpoints sit further from the cusps) the bias is milder:

| band | circumradius | exact | excess |
|---|---|---|---|
| (0.25, 0.35) | 0.16136 | 0.14717 | 9.6% |
| (0.45, 0.55) | 0.30895 | 0.28747 | 7.5% |
| (0.65, 0.75) | 0.46892 | 0.44167 | 6.2% |
| (0.85, 0.95) | 0.78344 | 0.74692 | 4.9% |

Once the band minimum runs over exact values the endpoints are evaluable and
this bias does not arise — there is no `n_samples`-dependent excess left to
document or tolerate.

**Consequence for step 1.** Do NOT assert byte-identity with the incumbent,
and do NOT assert the 5-10% margin either — both enshrine a discretization
artifact. The gate is 1e-12 against an independent oracle. The one behavioural
claim worth pinning is that the consumer decision `eta_max > 0.5 * r_min`
flips on NO production band — verified for `(0.25,0.35)`, `(0.45,0.55)`,
`(0.65,0.75)`, `(0.85,0.95)` and the small-gamma bands `(0.0281,0.0462)`,
`(0.0644,0.0825)`, `(0.0825,0.1550)`, `(0.1550,0.3000)`. The exact value is
SMALLER than the incumbent, i.e. the guard becomes marginally more willing to
skip — the conservative direction.

**Env note.** `SDK_CONDA_ENV = cogwheel-newlal` (from the repo-root `.env`)
carries mpmath 1.3.0 and sympy 1.14.0. mpmath produced the envelope above.
sympy is fine for deriving, but `lambdify` of the UNSIMPLIFIED second
derivative of this expression runs for minutes — simplify or `cse` first.

## F039 — the fold's two-image side is ANALYTIC; `_probe_arc_side` should not exist (2026-07-29)

**Where:** `surrogate_training._probe_arc_side` / `_PROBE_ETA`
(`cogwheel/lensing/surrogate_training.py`).

**RESOLUTION FIRST: there is nothing to measure here.** The caustic is the
image of the critical curve. At a critical point the source-map Jacobian `J`
is singular with soft eigenvector `e` (`J e = 0`), so displacing along `e`
kills the linear term:

    y(t) = y_c + (1/2) * D2y[e,e] * t**2 + O(t**3)

BOTH signs of `t` map to the same side — that is what makes it a fold — so the
two merging images live on the side the quadratic term points to, and

    inward_direction = D2y[e, e]

is exact and sign-definite with NO step, NO tolerance and NO image count. For
`y(x) = A x - x/|x|^2` only the point-mass term contributes, and contracting
its second derivative twice with a unit `e` gives, in closed form:

    D2y[e,e] = (4*(x.e)*e + 2*x - 8*(x.e)^2 * x/r^2) / r^4,   r^2 = |x|^2

with `x` and `e` already returned by `geometry.critical_point` as `.image` and
`.soft_axis`. Verified 2026-07-29 against a direct image count on each side at
32 `(gamma, theta)` points spanning `gamma` 0.005..0.99: **31 agree**. The one
exception is `gamma = 0.005, theta = 1.0`, where the image COUNTER returned 2
on both sides — `find_images_quartic` could not separate the barely-merged
pair at `eps = 6e-7` — and an `eps` sweep at that same gamma flips to
agreement as `eps` grows. The direction was right; the verification method was
the thing that degraded. That is the whole point: an analytic determination has
no step to degrade.

So step 3b is DELETE `_probe_arc_side` and `_PROBE_ETA`, not retune them. The
rest of this entry records how the numerical version failed, because it is the
reason the acceptance is what it is.

`_probe_arc_side` labels a fold arc by placing a test source `_PROBE_ETA` off
the caustic along the normal on each side, requiring the nearest-caustic
reconstruction to come back within `0.25 * _PROBE_ETA` in distance and 0.1 rad
in theta, and PREFERRING the side with more real images. The step is an
absolute 0.05 — the same numeric value as `_DEFAULT_ETA_MAX`, and, like it,
blind to how big the caustic actually is.

**Failure mechanism.** The probe must land INSIDE the caustic on the
image-pair side. It fails whenever the step exceeds the local caustic
half-extent: the 4-image side then fails its reconstruction check, only the
exterior probe survives, and the arc is silently labelled `(sign=+1,
image_count=2)` instead of `(sign=-1, image_count=4)`. This is not a
resolution effect — it is a step-size effect, which is why F037 measured the
consequent band failure as IDENTICAL at `n_samples` 200, 800 and 3200.

**Measured (positive parity, kappa = 0, branch +1).** Probe outcome
`(sign, n_img)`; `None` = neither side reconstructed:

| gamma | theta | caustic reach | step 0.05 | step 0.25*R_c |
|---|---|---|---|---|
| 0.02 | 0.6 | 0.0404 | None | (1, 2) |
| 0.02 | 2.3 | 0.0404 | (1, 2) | (-1, 4) |
| 0.07 | 2.3 | 0.1450 | (-1, 4) | (1, 2) |
| 0.15 | 1.0 | 0.3251 | (-1, 4) | (1, 2) |
| 0.30 | 1.0 | 0.7165 | (-1, 4) | (1, 2) |
| 0.70 | 1.0 | 2.5537 | (-1, 4) | (1, 2) |

**This resolves F037's second cause.** Holding `n_samples = 200` and shrinking
the step alone: `stable_gamma_bands((0.01, 0.30), +1)` goes from **4 stable
bands with 2 dropped slivers** at `_PROBE_ETA = 0.05` to **1 stable band with
0 dropped** at `_PROBE_ETA = 0.004`. The dropped topology slivers are not a
served-side detection bug needing new physics; they are the same
absolute-length disease as C6, on a different constant.

**`f * R_c` is NOT the fix either, and this is the trap that makes the
analytic route the only clean one.** `0.25 * R_c` flips `(sign, image_count)`
at gamma = 0.15, 0.3 and 0.7 — bands whose charts train successfully today.
`R_c` is a curvature radius, not a caustic THICKNESS: at gamma = 0.3,
`R_c = 1.05` while the whole caustic reaches only 0.72, so a quarter of `R_c`
steps clean through it. Every candidate step length is squeezed between "large
enough that `find_images` can resolve the pair" and "small enough to stay
inside the caustic", and at small gamma that window closes. There is no good
constant, which is the signature of a question that should not have been asked
numerically.

**Severity: latent, not live.** The mislabel is silent — the arc records a
different `image_count`, which is then stored on the chart and keys
`select_chart`. No trained chart artifact is shipped (F036), so nothing served
today depends on it. That is exactly the inertness window F036 describes, and
it is another reason not to train until the coordinates are settled: training
now would bake a probe-step-dependent side label into a shipped artifact.

## F040 — the cusp-exclusion half-width is DERIVABLE and scales as `w^(-1/4)`; the incumbent is w-INDEPENDENT and 2-50x too narrow (2026-07-29)

**Where:** `surrogate_training._find_cusps` (`delta_theta`),
`_CUSP_WIDTH_SAFETY`, `_CUSP_MIN_HALFWIDTH`, the `_SADDLE_*` variants, and
`surrogate._CUSP_ARM_COVERAGE`.

The cusp window was queued as a driver MEASUREMENT. It is not one. A cusp is
`y'(theta_c) = 0`, so the local structure is entirely in the Taylor tail, and
the width follows from the derivatives plus the cusp scaling SPEC already
records (`x ~ w^{1/2} delta_par`, `y ~ w^{3/4} delta_perp`):

    delta_par  ~ (1/2)|y''| dth^2       =>  dth_par  ~ sqrt(2/(|y''| w^{1/2}))
    delta_perp ~ (1/6)|y'''_perp| dth^3 =>  dth_perp ~ (6/(|y'''_perp| w^{3/4}))^{1/3}

Both go as `w^{-1/4}`, so their ratio is w-independent and the law carries ONE
dimensionless prefactor — set by the eps bar the chart must meet, not by a
sampling artifact.

**Measured** (astroid cusp at `theta = pi/2`, `kappa = 0`; derivatives from the
`critical_point`-validated curve). Half-width `max(dth_par, dth_perp)`:

| gamma | `y''` | `y'''_perp` | w=1 | w=10 | w=60 | incumbent |
|---|---|---|---|---|---|---|
| 0.05 | 0.308 | 0.585 | 2.549 | 1.434 | 0.916 | 0.0942 |
| 0.30 | 2.151 | 3.012 | 1.258 | 0.708 | 0.452 | 0.0500 |
| 0.90 | 17.076 | 3.415 | 1.207 | 0.679 | 0.434 | 0.0500 |

**Two consequences.** (1) The incumbent carries NO `w` dependence while the
physics is `w^{-1/4}`. That is structural: `cusp_windows` is STORED per chart
as a fixed `(theta_cusp, delta_theta)` pair, so the schema itself cannot
express the right answer — fixing this changes the chart schema, not a value.
(2) It is 2-50x TOO NARROW over the served band, i.e. charts are trained INTO
the Pearcey region a spline cannot represent. This retrodicts the failure
already recorded at `_SADDLE_CUSP_WIDTH_SAFETY` ("three saddle deltoid-arc
tube charts fit at eps 0.4..2.2 because their arcs are clipped to these
least-guarded ends"), whose fix was an empirical 2.5x saddle widening — right
direction, wrong reason: the deficit is the missing `w` scaling and the
astroid carries it too.

**Two candidate widths that are WRONG, recorded so they are not retried.**
(a) The Taylor SHAPE scale `3|y''|/|y'''|` — where the curve leaves its
osculating parabola — measures 1.58 rad at `gamma = 0.05` rising to 15 rad at
`gamma = 0.9`. It GROWS with gamma and is O(1) radians: a shape scale, not an
exclusion scale. (b) The tube self-intersection scale, `dth` solving
`R_c(theta_c +- dth) = eta_max`, runs 0.157 down to 3e-4 over the same range —
but C6 makes it VACUOUS by construction, since `eta_max = f * R_c` with
`f < 1` never crosses. Neither is the criterion; the envelope's own 2/3-power
structure is.

**So `_CUSP_ARM_COVERAGE` was never going to be pinned by a census.** It
defaults to `0.0` "until the census pins a nonzero coverage", but the quantity
it stands for is a FUNCTION of `w` and the local derivatives, not a constant a
census could report. That is why the measurement never happened.

**Addendum (2026-08-04):** `_CUSP_ARM_COVERAGE` was pinned at 0.07 rad by a
direct arm boundary sweep (`scripts/measure_cusp_arm_actual_boundary.py`,
commit ddd8980): the minimum angular offset from the cusp vertex at which
`cusp_amplification` actually serves, over gamma=[0.1..1.5], w=[10..40],
floored to 2 dp (conservative). This is a measured floor, NOT the analytic
w-dependent derivation F040 describes — the analytic derivation of the cusp
exclusion half-width `delta_theta` (w^{-1/4} scaling) remains open.

**Requires `y'''`, which build 1a does NOT deliver** — 1a exports `y'` and
`y''` only. Extend the cascade to third order before the cusp-window work.

## F041 — the fold-opening direction is nearly TANGENT to the caustic at small gamma, so an absolute `|dot|` guard reproduces the `_PROBE_ETA` bug (2026-07-29)

**Where:** `surrogate_training._make_arc`'s `abs(dot) > 0.1` guard and
`surrogate_training._tube_normal`.

Build 1b replaced `_probe_arc_side`'s absolute-length step with
`inward_sign = sign(fold_opening_direction . serve_normal)` — correct — and
guarded it with `abs(dot) > 0.1` to reject cusp-proximity. Measured, the
overlap scales LINEARLY with gamma:

| gamma | 0.02 | 0.04 | 0.06 | 0.08 | 0.10 | 0.30 | 0.90 |
|---|---|---|---|---|---|---|---|
| `\|dot\|` | 0.030 | 0.060 | 0.090 | 0.120 | 0.150 | 0.441 | 0.994 |
| angle to normal | 88.3 deg | — | 84.8 deg | — | 81.4 deg | 63.8 deg | 6.2 deg |

So `|dot| ~ 1.5 * gamma`, and an ABSOLUTE cut at 0.1 fails for every
`gamma < ~0.067`. That is the same category error as `_PROBE_ETA` (F039), one
level up: an absolute threshold on a quantity that scales with the caustic.

**Consequence: a REGRESSION, not a fix.** `stable_gamma_bands((0.01, 0.30),
+1)` previously returned 4 stable bands each with 2 arcs (mislabelled at small
gamma). After 1b two of the bands return ZERO arcs — `_make_arc` returns None
for every fallback fraction — so the small-gamma collar is now unserved for a
new reason. The labels ARE fixed where arcs survive (uniform `(-1, 4)` versus
the old `(1,2)`/`(-1,4)` split), so the geometry is right and only the guard
is wrong.

**The SIGN is not in doubt; only the magnitude guard is.** At `gamma = 0.02`
the overlap is 0.03 against a float64 noise floor of ~1e-16. Swept over the
prior the MINIMUM `|dot|` is 4.4e-3 (at `gamma = 0.01`), so the sign never
approaches ambiguity anywhere and needs no protection.

**`|dot|` is the WRONG QUANTITY, not a mis-scaled one — and the right form is
a DIMENSIONLESS RATIO OF LOCAL QUANTITIES.** `|dot|` measures how TRANSVERSE
the fold opening is, a legitimate gamma-dependent geometric fact. What the
guard wanted is CUSP PROXIMITY. Because `theta` is dimensionless, `|y'|` and
`|y''|` both carry length, so `|y'| / |y''|` is exactly the angular distance
to the cusp (where `y' = 0`); dividing by the arc half-span makes it fully
arc-relative. Measured at matched cusp proximity:

| gamma | 0.02 | 0.06 | 0.10 | 0.30 | 0.90 |
|---|---|---|---|---|---|
| `\|dot\|` | 0.0162 | 0.0486 | 0.0810 | 0.2418 | 0.6902 |
| `\|y'\|/\|y''\|` | 0.3071 | 0.3099 | 0.3125 | 0.3236 | 0.3188 |

`|dot|` swings 43x; the ratio moves 4%. That is the difference between a
quantity that tracks the caustic's SIZE and one that tracks its SHAPE, and
only the second belongs in a guard. GENERAL RULE for this package: a guard is
a dimensionless O(1) constant applied to a scale-free ratio of local
quantities carrying the same units. An absolute cut on a dimensionful local
quantity is the bug class this whole sweep exists to remove — `_PROBE_ETA`
(F039), `ANNULUS_INNER_RADIUS` (F036) and this guard are three instances of
it.

Deleting the guard outright is defensible: the arc bounds already exclude the
cusp windows, and `|y'|/|y''|` over the arc half-span measures >= 0.39
everywhere sampled. If a guard is kept, it must be the ratio.

**Why the near-tangency is real physics, not a bug.** `D2y[e,e]` is the
direction the fold OPENS, and for a fold it need only be TRANSVERSE to the
caustic, not perpendicular to it. As `gamma -> 0` the astroid degenerates and
the opening direction rotates toward the tangent. F039's image-count check
still passes at `gamma = 0.005` (31/32 overall), so the direction remains a
correct side-indicator throughout; it is the projection onto the normal that
shrinks, not the answer that degrades.

**Separately — and it really is separate: `_tube_normal` was missed by the
whole sweep, and still finite-differences.** It builds the caustic tangent as
`critical_point(theta + 1e-6) - critical_point(theta)` — a forward difference
of a closed form with a hardcoded step, on the serve-consistency path. It was
absent from the [[lensing_analytic_derivatives]] inventory (driver's miss, not
the build's) and is a one-line replacement: `tangent = y' / |y'|` from
`caustic_derivatives`. Added as target 5 there.

**DO NOT expect that fix to rescue the guard.** Measured both ways at the same
points: the analytic normal differs from the finite-differenced one by
2.9e-5 degrees, and the resulting `dot` moves by ~5e-7 —
`-0.029998000` becomes `-0.029997500` at `gamma = 0.02`. The near-tangency is
GEOMETRY, not discretization. So `_tube_normal`'s finite difference is wrong on
PRINCIPLE (a numerical derivative of a closed form, unjustified step) and not
wrong in EFFECT, while the `abs(dot) > 0.1` guard is wrong in effect and must
be fixed independently. Bundling them — as the driver's first write-up of this
finding implicitly did — invites someone to replace the difference, see the
arcs still missing, and go looking for a third bug that is not there.

## F042 — a knife-edge synthetic fixture tipped over 0.05 when the analytic cusp shifted its arc bounds; RESOLVED, re-based (2026-07-29)

**Where:** `surrogate_training` saddle tube training; surfaced by
`test_lensing_surrogate_training.py::SaddleTubeTailTestCase` under
`COGWHEEL_TRAIN_TIER=1`.

Retiring the numerical cusp/speed/curvature estimators for their analytic
closed forms (build 1b) moved the detected cusp ANGLES to their exact roots
(cusps are `|y'| = 0`), which shifts each fold arc's `theta_lo/theta_hi`
bounds by up to one former sampling step. For one synthetic saddle tube
fixture this tips the fix-on arc's held-out eps from just under the tube
registration bar to just over it:

    on_eps = 0.0591   vs   _TUBE_EPS_BAR = 0.05      (was ~0.0499 pre-1b)

At the fixture's coarse grid this tipped the fix-on chart from registered to
gated. That looked at first like a lost saddle coverage cell — the resolution
sweep below shows it is not (the fit is fine at any real grid).

**Why this was only caught now.** These are `@_TRAIN_TIER_SKIP` tests that
build real charts (minutes/class) and run only under `COGWHEEL_TRAIN_TIER=1`
as a driver post-build step. Build 1b CRASHED (API outage) before its own
train-tier verification ran, so the consequence sat undetected in the salvaged
tree until the F041 follow-up's driver tail ran the tier.

**Attribution: NOT the F041 fix.** The F041 guard (`abs(dot) <= 0.1`, this
build) binds only below `gamma ~ 0.067` — the astroid regime. Saddle arcs are
`gamma > 1` where `|dot| ~ 1`, untouched. This eps shift is a 1b analytic-cusp
consequence, independent of F041.

**RESOLVED (2026-07-29): a coarse-synthetic knife-edge, not a coverage loss.**
Answer (a). A resolution sweep of the SAME production fix-on arc holds the
saddle geometry fixed and grows only the chart grid:

| grid | fix-on held-out eps | vs 0.05 |
|---|---|---|
| 4x4x4 (the fixture) | 0.0592 | OVER |
| 5x5x5 | 0.0222 | UNDER |
| 6x6x6 | 0.0132 | UNDER |

eps falls 4.5x from grid 4 to 6, so the fit is fine at any real resolution;
the shipped trainer builds far finer than this synthetic. The 0.0591 was an
artifact of the deliberately-coarse 4x4x4 fixture being KNIFE-EDGE calibrated
(0.0499) against the pre-1b SAMPLED cusp bounds. The analytic cusp (correct:
`|y'| = 0` to 1.3e-16) shifted the arc bounds by ~one former sampling step,
enough to tip the knife-edge case over 0.05 at that one coarse grid only.

The analytic cusp is right, the production saddle tube chart is not at risk,
and no coverage cell was lost. FIX: the fixture (`_WP3_CONFIG`) is re-based
from grid 4 to grid 5, where fix-on clears the bar with margin (~0.022) and
the fix-off pathology still sits far above it — the exact 0.05 bar semantics
are preserved, not weakened. The two SaddleTubeTail tests are un-skipped and
pass. No production change.

**ROOT CAUSE (deeper than the knife-edge): the theta axis is placed in an
ABSOLUTE coordinate.** The tube chart grids `(log w, gamma, u, theta)`; `u` is
already analytic (`u = sqrt(eta)`, the fold's own variable) but `theta` is
`linspace(theta_lo, theta_hi)` — uniform in theta, blind to the caustic. Near
a cusp `|y'| -> 0`, so uniform-theta mis-allocates nodes relative to where the
envelope varies. Measured at the SAME n_theta = 4 on this arc, same held-out
set:

| theta placement | eps | vs 0.05 |
|---|---|---|
| uniform in theta (current) | 0.0592 | OVER |
| uniform in ARC LENGTH `s = int |y'| dtheta` | 0.0271 | UNDER |

Arc-length placement fits **2.2x better at identical node count** — so this is
node PLACEMENT, not node COUNT (grid = 5 merely throws more uniform-theta nodes
at it). And the uniform grid is what makes the fit knife-edge: nudging the arc
bounds by +-0.01 rad swings uniform-theta eps 0.0592 -> 0.0727 / 0.0575 (+-23%),
which is exactly the mechanism by which the analytic cusp's small bound shift
tipped this fixture. An arc-length grid tracks the geometry and is insensitive
to the bound.

The arc-length coordinate is `int caustic_speed dtheta` — computed from the
SAME `geometry.caustic_speed` shipped in 1a to retire an estimator. The cascade
that caused F042 supplies its proper fix.

**So F042's real resolution is theta-axis arc-length placement**, which is
[[lensing_collocation_from_local_scales]] (its theta item, C6 / step 2). The
grid = 5 fixture re-base is a STOPGAP that unblocks the two SaddleTubeTail
tests today; it is not the fix. Durable lesson: an analytic estimator strictly
MORE accurate than its sampled predecessor can still move a knife-edge
synthetic across a threshold when the synthetic was tuned to the old
estimator's exact output on a grid placed in the wrong coordinate. The fix is
the right coordinate, not more nodes.

## F043 — git-HEAD-relative "compare to the old implementation" tests are landmines: they pass in their own build's gate, then break when HEAD moves (2026-07-30)

**Where:** `test_lensing_caustic_cusps.py` (`_head_find_cusps` /
`_head_module_source`); previously `test_lensing_surrogate_training.py`
(the astroid byte-identity tests, retired the same day).

A test that reconstructs the PRE-change implementation from `git show HEAD:...`
to compare against the worktree is valid ONLY while HEAD is still the
pre-change commit — i.e. only during the build that introduces the change,
before it commits. The instant that build's commit lands, HEAD becomes the
NEW version and the cross-version comparison either compares a version to
itself or, worse, fails to reconstruct because the change deleted a symbol the
reconstruction needs.

Concretely: `_head_find_cusps` AST-extracts the old `_find_cusps` from HEAD,
requiring the module constants `_CUSP_SPEED_REL_FRAC`, `_CUSP_WIDTH_SAFETY`,
`_CUSP_MIN_HALFWIDTH`. Build 1b (`00bf8ae`) DELETED `_CUSP_SPEED_REL_FRAC`
(inlined as a local). The tests PASSED in 1b's own tree gate — that gate runs
BEFORE the commit, so HEAD still had the constant — and went RED in the very
NEXT build's (1c's) tree gate, which runs against the now-advanced HEAD:
`RuntimeError: HEAD cusp constants missing: ['_CUSP_SPEED_REL_FRAC']`. The
failure had nothing to do with 1c; 1c never touched `_find_cusps`.

This is the THIRD instance in one day (the two astroid byte-identity tests
were the first two, retired when 1b deleted their float-path baseline). The
`@skipUnless(_git_available())` guard did not help: git WAS available and
`git show HEAD` succeeded; the missing piece was a deleted CONSTANT, which the
guard does not check.

**Rule.** A regression guard that must survive its own baseline being
committed cannot key on a moving `HEAD`. Either (a) pin a specific historical
commit SHA (opaque, and still breaks when the rule legitimately changes), or
— preferred — (b) freeze the expected values as a GOLDEN TABLE of literals in
the test, computed once and readable without git. A `git show HEAD`
cross-version oracle is only legitimate as a WITHIN-BUILD transition check that
is retired the moment the transition commits. The two 1c-blocking tests are
retired (`@skip -> F043`); the durable window-width guard, if wanted, is a
golden-value table (owed to the F040 cusp-window build, which changes the
window rule anyway).

## F044 — the macro-saddle wedge edge is a REGULAR point of the caustic; only the `theta` parametrization is singular, so `_WEDGE_EPS` buys no safety and costs coverage (2026-07-30)

**Where:** `surrogate_training._WEDGE_EPS = 1e-3` (6 sites);
`geometry.caustic_derivatives` docstring.

At the wedge edge `theta_max = (1/2) arcsin(lam / |gamma|)` the discriminant
`1 - e**2 sin**2 2theta` vanishes, so `u` has a square-root branch point and
the theta-derivatives diverge. Measured (driver, `gamma = 1.3 / 2.0 / 5.0`,
`dtheta = 1e-2 .. 1e-12`), the rates are exactly

    |y'|  = A dtheta**(-1/2)      |y''| = (A/2) dtheta**(-3/2)

with `A` constant to 5 significant figures across four decades of `dtheta`
(`A = 0.85124 / 0.40826 / 0.14434`). Both were read as evidence of a geometric
singularity. They are not.

Reparametrise by `s = sqrt(theta_max - theta)`. Then `y(s)` converges to a
finite limit and `|dy/ds| = |y'| * 2s` converges to a NONZERO one
(1.70248 / 0.81652 / 0.28868, stable from `s = 1e-4` down to `1e-6`), and
`A = |dy/ds| / 2` exactly. A curve with finite position and nonzero tangent in
its own regular parameter is a REGULAR point — the two square-root branches
meet in a smooth turnaround, precisely as `_saddle_arcs`' docstring says
("walls, but not cusps"). `caustic_derivatives`' docstring calling it "the
deltoid cusp" is wrong: the deltoid's three cusps are the interior
`|y'| = 0` roots, and the wedge edge is not one of them.

**Consequence 1 — the standoff buys no safety.** `critical_point` serves the
edge exactly (`dtheta = 0` returns a finite point) and refuses by name
immediately outside it (`dtheta = -1e-12` raises `LensDomainError`, no silent
clamp); `caustic_derivatives` refuses at `dtheta <= 0`. The named refusals
already are the guard, and every sampler already skips `LensDomainError`
per-theta.

**Consequence 2 — the standoff costs coverage, measurably.** It excises a
sliver at both ends of every wedge sweep. `_lobe_winding_loop` is documented
as a closed lobe boundary and is fed to `_winding_number` as the saddle
interior-admission test (`abs(w) < 0.5` rejects a tile); with the standoff its
closure gap is 0.279 at `gamma = 1.05` (9.3% of the lobe reach), 0.107 at 1.3,
0.051 at 2.0. Sampling the wedge CLOSED takes the gap to exactly 0.0 and
changes nothing else — cusp count, arc count and reach identical at every
gamma tested, total arc span slightly LARGER. Against a standoff-free
reference interior, the open loop rejects 1/792 interior probes at
`gamma = 1.05`, reaching 0.059 in source-plane units INSIDE the lobe.

## F054 — the surrogate spends 90% of its time SAMPLING the caustic and 1.7% evaluating its spline (2026-07-30)

**Where:** `ppgo_map.caustic_geometry` (n_theta=720, both branches), reached
per serve via `surrogate._caustic_reach` -> `_to_caustic_fixed` -> `serve`.

Measured after the owner asked whether 31 ms per served likelihood is too
slow. It is, and the reason is not the spline:

| | per served `lnlike` | share |
|---|---|---|
| surrogate-served LENSED `lnlike` | 31.25 ms | 100% |
| `_surrogate_coefficients` | 27.88 ms | **89%** |
| `ppgo_map.caustic_geometry` | ~27.5 ms | **90% of the serve** |
| `geometry.critical_point`, **1440 calls per evaluation** | 0.679 s / 20 calls | |
| `_contract_tensor_spline` — THE SPLINE | 0.013 s / 20 calls | **1.7%** |

`caustic_geometry` is a Python double loop over 2 branches x 720 polar angles,
calling `critical_point` on each, to find the MAXIMUM source-plane radius by
scanning. It runs on every likelihood evaluation.

**Two independent defects, either of which is most of the cost.**

1. **It is a sampled scan where a closed form exists** — the same disease
   step 1 has been curing since F039/F041. The maximum of `|y(theta)|` is an
   extremum of a closed-form curve, i.e. a ROOT of `d|y|^2/dtheta = 0`, and
   the 1a cascade (`caustic_derivatives`) supplies the derivative. The model
   to copy is already in this package and already cited in
   [[lensing_analytic_derivatives]]: `geometry.r_caustic` "samples only to
   BRACKET and refines every root with brentq to 4*eps".
2. **It is recomputed per evaluation for a quantity that does not depend on
   the evaluation.** `reach` is a function of `(gamma, kappa)` alone — not of
   the source position. Nothing caches it, while `_schwinger`,
   `_pearcey_cusp` and `prior` all use `lru_cache`. This is the
   already-recorded pattern "values derived from (source, matrix) belong ON
   the partition, not re-derived inside hot-path functions", which cost
   ~250 us twice in one day earlier in this program.

**Why it went unnoticed:** the surrogate's own timing test asserts a SPEEDUP
against the exact engine (9.6-20.4x, comfortably passing), and the exact
engine is ~300-630 ms. A serve that is 100x too slow still looks like a
triumph next to that. Nothing measured the surrogate against what a spline
evaluation OUGHT to cost.

**Scale of the prize:** at 31 ms, 5M likelihood evaluations is ~43 core-hours;
at the ~3 ms the fast path already targets it is ~4 core-hours. Removing the
scan should recover most of the difference, since 90% of the serve is this one
function.

**Rule.** A surrogate must be benchmarked against the cost of the operation it
performs, not only against the exact path it replaces. "Faster than exact" is
necessary and nowhere near sufficient — it hides a factor of 100 whenever the
exact path is slow enough.

## F053 — an absolute wall-clock bar measures the machine; the speedup ratio measures the code (2026-07-30)

**Where:** `test_lensing_surrogate.py::TimingSmokeTestCase`; the new
`.claude/sdk/timing_pass.sh`.

Auditing the two never-run timing variables (F052) meant running them for the
first time. Result, serial, 44:44 wall clock: **13 passed, 1 failed.**

All four `COGWHEEL_STRICT_TIMING` branches PASSED. Every one asserts a SPEEDUP
RATIO (brute-force time over fast-path time), and a ratio of two measurements
on the same machine cancels that machine's speed — it means the same thing on
a quiet laptop and a loaded cluster node.

The single failure was the one ABSOLUTE bar:

    [TimingSmoke] saddle: sur=31.013 ms  exact=632.012 ms  speedup=20.4x
    AssertionError: 31.013 not less than 15.0

The surrogate beat the exact path by **20.4x** — the property the test is
named for — and failed a 15 ms wall-clock bar on a box where an unrelated
process had held ~98% of a core for 21 days. The code was fine; the bar was
measuring the machine.

**The test contradicted itself.** Its section header says "CI-skippable, never
a hard gate" and its class docstring says "machine-dependent -> opt-in only",
yet it asserted the absolute bar hard. That contradiction survived because
nothing ever set `COGWHEEL_RUN_TIMING_SMOKE`, so the assertion had never once
executed. **An assertion that has never run is a claim nobody has checked** —
including its author.

**Fixed** by making the assertion match the documented intent: the speedup
gate stays HARD, the absolute number is REPORTED with a note. Verified by
falsification — forcing the speedup to 1.0x still fails with "1.0 not greater
than 5.0", so the gate keeps its teeth.

**Rule.** Prefer a RATIO to an ABSOLUTE bar for anything timed. An absolute
bar cannot distinguish "the code regressed" from "the box is busy", so it
either fires spuriously or gets quietly relaxed until it means nothing. Where
an absolute number is genuinely wanted, report it and gate the ratio.

**Scheduling** (the other half): this tier is SERIAL by necessity and cost
44:44, because the strict branches time `lnlike_bruteforce` as their
reference. That does not belong in the per-build sweep — it would nearly
triple every build's post-step to re-check ratios that do not silently drift.
It now lives in `.claude/sdk/timing_pass.sh`, a driver command for before a
release or after touching the fast path. The `PARALLEL_UNSAFE` exclusion in
the sweep is therefore right for two reasons, not one: contention would
corrupt the measurement, AND the measurement is expensive enough to need its
own schedule.

## F052 — the train tier holds every build's acceptance gates and NO routine job ran it (2026-07-30)

**Where:** `.claude/sdk/post_build_sweeps.sh`; `_TRAIN_TIER_SKIP` in the
lensing test suite.

Slow tests never run in-build — correctly, that is a standing law. They are the
driver's post-build job, and `post_build_sweeps.sh` is the one command that
does it. But that script exported only `COGWHEEL_BRUTE_ACCURACY=1`. Nothing,
anywhere, set `COGWHEEL_TRAIN_TIER`.

So roughly 20 classes across four files — including
`ArcLengthBoundShiftMarginTestCase`, the knife-edge gate that IS build
1e-tube's central claim — were skipped by the in-build tree gate (correctly)
AND skipped by the post-build sweep (silently). The sweep reported
`test_lensing_surrogate_training 31 passed, 48 skipped` and read as green.

**A build could therefore pass every gate it could reach while the thing it
existed to prove went untested.** 1e-tube's acceptance ran only because an
unrelated commit gate blocked and forced a manual tier run; without that
accident the build would have been reported as verified on the strength of
gates that never touched its claim.

**Fixed**: the sweep now sets `COGWHEEL_TRAIN_TIER=1` alongside
`COGWHEEL_BRUTE_ACCURACY=1`. Cost is small — the 1e-tube tier ran 65 tests in
5:15, against a sweep that already takes ~25 minutes.

**Rule.** Same shape as F051, third instance in one day: a gate nobody executes
is not a gate. When a test tier is created, name the job that runs it and check
that job actually sets its variable. "It runs post-build" is a claim about a
script, and the script is checkable.

**AND THE FIRST FIX WAS ITSELF THE SAME MISTAKE.** Hardcoding
`COGWHEEL_TRAIN_TIER=1` beside `COGWHEEL_BRUTE_ACCURACY=1` fixed the instance
and left the class. Prompted by the gw driver — whose sweep DISCOVERS gated
files by grepping for the variable, so new gated files enrol themselves — an
audit of every gating env var in `cogwheel/tests/` found FOUR, of which **two
were still set by nothing**:

| var | refs | before |
|---|---|---|
| `COGWHEEL_BRUTE_ACCURACY` | 13 | set by the sweep |
| `COGWHEEL_TRAIN_TIER` | 4 | set (the F052 fix) |
| `COGWHEEL_STRICT_TIMING` | 7 | **set by nothing** |
| `COGWHEEL_RUN_TIMING_SMOKE` | 1 | **set by nothing** |

The sweep now greps the suite for `COGWHEEL_*` gate variables and enables what
it discovers, so a new tier enrols itself (verified: a probe file introducing
`COGWHEEL_BRAND_NEW_TIER` was picked up immediately).

**Timing tiers are excluded ON PURPOSE, and REPORTED.** A timing assertion is
meaningless under an 8-wide sweep's CPU contention, so
`COGWHEEL_STRICT_TIMING` and `COGWHEEL_RUN_TIMING_SMOKE` are named in a
`PARALLEL_UNSAFE` list and printed as skipped with the reason, rather than
silently dropped. **They remain uncovered**: the full-suite gate deselects
timing tests and runs them serially, but with `COGWHEEL_STRICT_TIMING` unset,
so the STRICT timing tier still runs nowhere. That is an open gap, now visible
in the sweep's own output instead of invisible.

**Meta-rule.** When a fix is prompted by one instance, ask what CLASS it
belongs to and enumerate the class before declaring it fixed. Three of the four
tier variables were discoverable with one grep.

## F050 — every revision-budget exhaustion in the SDK's history is the same unfixable SPEC.md finding (2026-07-30)

**Where:** the Inspector revision loop in `.claude/sdk/orchestrator.py`; the
brief/plan surface lists that feed it.

Audited all 28 builds on this box that ran a revision loop. **Three exhausted
their budget. All three were stuck on `.claude/spec/SPEC.md`. No other file has
ever produced a repeated finding.**

| build | rounds | stuck finding |
|---|---|---|
| `analytic_geometry_cascade` (1a, 07-29) | 3 | `INS-1-001 SPEC.md` x3 |
| `saddle_born_carrier` (07-28) | 3 | `INS-10-001 SPEC.md` x3 |
| `tube_arclength_coordinate` (1e-tube, 07-30) | 3 | `INS-1-001 SPEC.md` x3 |

The contrast confirms the loop itself is healthy: `born_carrier_bandsplit` ran
8 revision rounds and `lensing_build5` ran 7, both WITHOUT exhausting budget
and without a repeated finding — each round fixed something and surfaced
something new. The three failures are the ones where the loop had nothing it
could do.

**Mechanism — and the audit above UNDERCOUNTS it.** The revision loop does
dispatch trivial findings, to Foreman-Lite. Foreman-Lite has full edit tools,
`bypassPermissions` and a 75-turn budget, so it is not blocked by capability.
It DECLINES, correctly: `SPEC.md` is the Librarian's file and the finding is
explicitly tagged `→ Librarian:`. In 1e-tube it made exactly ONE tool call per
round — `write_memory` — and exited in 16 s.

What it wrote there is the whole story:

> INS-1-001 (this session, **recurrence 14x+**) ... Declined per role
> boundary — SPEC.md is Librarian-owned, not Foreman-Lite. No files touched.
> ... the orchestrator routing bug (dispatching "-> Librarian"-tagged findings
> to Foreman-Lite) **needs to be fixed upstream — recommend a pre-filter that
> strips these before they ever reach the Foreman-Lite queue.**

So the true recurrence is **14+, not 3** — the three budget exhaustions are
only the builds where the loop ran long enough to exhaust. The agent hitting
the bug root-caused it correctly and recommended the exact fix, in its own
memory, at least thirteen times, and nothing upstream ever acted on it. That
is the more damning half: the diagnosis was already in the system.

**Fixed** by a Tier 0.5 pre-filter that routes findings on `.claude/spec/**`
(or tagged `librarian`) to the LIBRARIAN before Foreman-Lite sees them, mirroring
the existing Tier 1.5 that routes test-authorship findings to the Test
Developer for exactly this reason. A Librarian failure there is caught: doc
prose must never abort a build whose code work is done and verified.

## F051 — three separate gates whose satisfying action lives OUTSIDE the loop that must satisfy them (2026-07-30)

**Where:** the build DAG's stage order and the pre-commit gate set.

F050 is one instance of a general shape. Build 1e-tube hit all three in a
single run:

1. **Librarian findings** (F050): the Inspector flags `SPEC.md`; only the
   Librarian may edit it; the Librarian is not in the revision loop. Fixed by
   routing.
2. **Changelog fragments**: the pre-commit hook demands a
   `contracts_changelog.d/` fragment when `DATA_CONTRACTS.yaml` changes.
   Fragments are the LIBRARIAN's artifact — the crew contract is explicit
   that the Inspector owns the ACCURACY of spec and contracts as checkable
   invariants while the Librarian owns SYNC, i.e. it is the role that WRITES
   every doc surface. And the Librarian ran one step AFTER the commit, so the
   owning role could not satisfy a gate that preceded it.

   The gap was papered over by `commit_preflight`, which auto-stubbed a
   fragment: hardcoded `bump: patch` regardless of the real change (1e-tube's
   schema addition was `minor`), a title scraped from the commit message
   subject, and an "(Auto-generated ... Librarian should refine)" note
   rendered straight into the CANONICAL changelog — with nothing tracking it
   for refinement. It also ran `git add -u` twice, re-introducing the blanket
   staging fixed elsewhere the same day.

   FIXED (2026-07-30): the doc stage moved ahead of the commit, so the owner
   writes its own fragments and the build produces ONE coherent commit
   instead of a code commit plus `docs: update documentation after build`
   plus `docs: render fragments after librarian`. `commit_preflight` stays as
   a backstop, because a build must never die at the commit for a doc-prose
   reason. Both fast paths are untouched (they run no Librarian).

   NOTE this gate did not fire in 1e-tube only because the hook runs
   correctness gates FIRST and drift blocked before it was reached — both
   were live.
3. **The gated-test-drift gate**: for a FALSE positive the remedy is an ack;
   for a GENUINE break it is updating the gated tests — the Test Developer's
   work. The Test Developer runs at Step 3; the gate fires at commit, after
   Step 5. In 1e-tube it finished at 09:32 and the gate fired at 10:09, 37
   minutes after the only agent who could have fixed it had finished, and
   nothing re-summons it. **So a build making a genuinely breaking change —
   e.g. correcting a past mistake — cannot converge**: it does all the work,
   passes Inspector and Professor, and dies at a gate whose remedy left two
   steps earlier.

   PARTLY FIXED (2026-07-30): the checker was imprecise as well as
   mis-scheduled. It compared signature fingerprints for EQUALITY, so a purely
   additive change — a new keyword-only parameter with a default, which no
   existing call site can observe — flagged exactly like a rename. And it
   matched a BARE METHOD NAME, so `TubeChart.from_values` changing flagged
   test classes that only ever touch `FarFieldChart.from_values`, which never
   changed. All 14 of 1e-tube's flags were false on both counts. It now
   classifies additive-vs-breaking and records the owning class. Verified
   against 16 classification cases and an end-to-end replay: silent on the
   real 1e-tube change, blocking on a rename, on a new required
   keyword-only parameter, and on a deleted method, each naming the exact
   parameter.

   STILL OPEN: surface drift as an INSPECTOR-phase finding, not only a commit
   gate. The revision loop already has a Tier 1.5 that routes test-authorship
   findings to a fresh Test Developer; a breaking drift finding routed there
   would be fixed in-loop and converge, leaving the commit gate as a backstop
   that should essentially never fire.

**Rule.** For every gate, name the actor who can satisfy it and check that
actor runs BEFORE the gate, inside the same loop. A gate whose remedy lives
downstream of itself is not a guard — it is a scheduled stall.

**RETRACTED (same day):** the first draft of this finding proposed letting the
pipeline run the gated tier for the classes a drift finding names, so a build
could ack from evidence. That is wrong and contradicts a standing law of this
repo — *slow tests NEVER run inside a build, no exceptions*. The law exists for
good reasons: an hour-scale in-build test deepens the transcript (which
measurably raises the permission classifier's fail-closed rate), and it is a
gate the in-build agents cannot actually run, so it certifies nothing.

The correct fixes are:
- **Make the checker precise instead of the pipeline slow.** The drift check
  matches a BARE METHOD NAME, so it cannot distinguish
  `TubeChart.from_values` from `FarFieldChart.from_values` (which never
  changed), nor a purely ADDITIVE optional-keyword change from a breaking one.
  Every one of 1e-tube's 14 flagged classes was a false positive on both
  counts. A signature-aware check — flag only removed, renamed or reordered
  parameters, and resolve the owning class — removes the whole class of
  stall without weakening the guard.
- **Keep slow verification with the driver, post-build**, which is where the
  law already puts it (`.claude/sdk/post_build_sweeps.sh`). If a drift finding
  survives a precise check, the build should COMMIT and record a loud driver
  task rather than strand — the gate's purpose is to prevent silent breakage,
  and a recorded task achieves that without leaving a verified build's work
  uncommitted in a working tree.

Still open and cheap: move the Librarian ahead of the main commit (or give the
commit a Librarian pass) so item 2 above stops needing its auto-stub
workaround.

**Rule (F050).** A revision loop can only fix what some agent owns. When
adding a gate that inspects a surface, check that the surface is inside
somebody's write scope — otherwise the gate is a guaranteed budget burn, not
a guard.

## F049 — marker strings were never checked against what the code actually emits, and a guard was placed where it could not run (2026-07-30)

**Where:** `.claude/sdk/cli.py` terminal markers; `.claude/sdk/orchestrator.py`
`_run_tidier_skill`; `scripts/tidy_mechanical.py`. All found by the gw driver
while porting these same fixes, and all real here.

**1. A phantom success marker.** The monitor's terminal set contained
`Build complete`. cogwheel emits that string **zero** times — 0 occurrences in
both real build logs; the success path prints the `  BUILD REPORT` banner. So
the monitor could exit on FAILURE but never on SUCCESS, leaking a 120-s polling
subshell after every clean build. The same orphan family as the seven-day hook
spin (F046).

**2. An anchor that could not match the most important line.** The anchor was
`^(\[[0-9:]+\])?[[:space:]]*`, but the watchdog emits
`[2026-07-30 08:15:22] === KILLED BY WATCHDOG (...)`: the bracket holds a date
and a space, and the marker sits behind `=== `. Both defeat that anchor, so a
watchdog-killed build would never stop its monitor. Fixed to
`^(\[[^]]*\])?[[:space:]]*(===[[:space:]]*)?` with the markers this repo
actually emits, verified line by line against real logs.

Together these are one mistake: **the marker set was written from memory of
what a build log ought to say, never checked against what the code prints.**
F048 was the same family seen from the other side — markers that matched
something they should not. Both are cured by the same discipline: before
keying a monitor on a string, grep the SOURCE for it and grep a REAL log for
it. A marker that appears zero times in either is not a marker.

**3. A guard placed where it could never run.** The mechanical style pass was
called inside `_run_tidier_skill` AFTER the early return that fires when the
in-DAG Tidier is skipped — and that skip is the DEFAULT. So the deterministic
pass, whose entire purpose was to stop style depending on whether an expensive
agent was scheduled, ran only when that agent was scheduled. Moved above the
opt-in check.

**4. A line-based normaliser inside string literals.** `_normalise` rstripped
every line, including lines inside a docstring, where a trailing space is part
of the string's VALUE. The AST guard caught it and aborted, so nothing was ever
corrupted — but the affected file was then never tidied at all. Now tokenises
first and protects the interior lines of multi-line strings, so surrounding
whitespace is still fixed while string bytes are preserved.

**Rule.** A defence is only as good as its trigger condition. Ask of every
guard: what exact string fires this, does the code emit it, and is the guard
reachable on the default path? Three of these four were failures of that
question, not of the logic behind it.

## F048 — a log monitor that greps for markers the log's own header ECHOES reads its own instructions as results (2026-07-30, twice)

**Where:** `.claude/sdk/cli.py`'s emitted Monitor command; earlier the same
day, an ad-hoc build-retry loop.

`cli.py` prints the suggested Monitor command INTO the log header. That command
string contains its own marker regex, so the words `Build complete`,
`Build failed`, `GATE FAILURE`, `KILLED` all appear on line 5 of every build
log before the build has done anything. A monitor grepping for them matches its
own echo on the FIRST poll: build 1e-tube was declared finished nine seconds
after launch and ran unmonitored until the discrepancy was noticed.

**This was the second instance in one session.** The first: a retry loop keyed
"did the build survive?" on the log containing `Plan written|plan_ready`, which
the same header echo also carries, so it reported SURVIVED for builds that had
already died. That one was fixed by keying on the `plan.json` ARTIFACT instead
— and then the identical mistake was made again in a different file, because
the fix had been treated as a one-off rather than as a rule.

**The rule.** A log is not a clean event stream: it contains instructions,
echoed commands, and quoted examples as well as results. Before grepping a log
for a marker, ask *what else in this file contains that word*. Two defences,
both cheap:
- ANCHOR to line structure. Real markers begin a line (optionally after a
  `[HH:MM:SS]` stamp); an echoed command carries them mid-line inside a quoted
  string. `^(\[[0-9:]+\])?[[:space:]]*(MARKER)` separates them completely.
- Prefer an ARTIFACT to a log line where one exists. `plan.json` on disk cannot
  be echoed into a log header; the string "plan_ready" can.

Verified with a synthetic log carrying a realistic header echo: unanchored
matches it, anchored ignores it, and the anchored pattern still catches a real
`[08:31:02]   Build complete`.

## F047 — a style pass wrote literal `\n` into operator.py, and nothing in the gate stack checked that staged Python parses (2026-07-30)

**Where:** `cogwheel/lensing/chang_refsdal/operator.py` line ~215;
`.claude/hooks/pre-commit`.

Three blank lines between `_CONTRACTION_TARGET = 1e-10` and
`@dataclass(frozen=True)` were replaced by the literal two-character sequence
`\n`, leaving `_CONTRACTION_TARGET = 1e-10\n\n\n@dataclass(frozen=True)` on one
line. `SyntaxError: unexpected character after line continuation character` —
the module does not parse and the whole package is un-importable.

**Attribution (circumstantial, not proven).** The leading suspect is an
interrupted Tidier run: `operator.py` is 8th of the 32 files in
`.claude/tidy_advisory.json`; collapsing blank lines around a top-level
definition is precisely rubric rules 1 and 3; the corruption is the signature
of a `replace_content` call with escaped rather than real newlines; the file's
mtime was 05:53:26, two minutes AFTER the slow sweeps finished green; and
`.serena/memories/tidy_short_term.md` does not exist, so whatever ran never
reached its final mandatory memory write. The driver had reported "the Tidier
never ran" on the strength of a harness rejection message, against the owner's
direct observation that it ran long and unfinished — the observation was right.

**The real finding is the gate gap.** Nothing in the pre-commit stack checked
that staged Python PARSES. This commit was blocked only by coincidence: the
corrupted region happened to swallow the name `geometric_amplification`, which
a SKIPPED test references, so `check_gated_test_drift.py` fired. Had the
mangled lines not mentioned a symbol some skipped test names, a non-parsing
module would have landed on `claude-dev` with every gate green.

**And the test suite could not have caught it.** The full suite (969) and the
slow sweeps (1016, 0 failed) had both run green minutes earlier — against the
file as it was BEFORE the edit. A green suite is evidence about the tree that
was tested, never about the tree being committed.

**Fixed:** a syntax gate now runs FIRST among the correctness gates. It reads
the staged blob (`git show :file`), not the worktree, so a broken index entry
cannot hide behind a clean working file — verified against that exact case.

**Rule.** `git add -A` is how an unrelated broken file gets swept into a
commit; stage explicitly. And the cheapest possible check — does it parse —
belongs before every expensive one.

## F046 — `${var/${BASH_REMATCH[0]}/x}` spins forever on a bracket class: the Bash gate hook hung a core for 7 days (2026-07-30)

**Where:** `.claude/hooks/use-serena.sh`, the three command-normalizer loops
in the `Bash)` branch (`$( )`, backticks, and the leading `VAR=` stripper).

`${var/PATTERN/repl}` and `${var#PATTERN}` treat PATTERN as a **glob**, but
`BASH_REMATCH[0]` is the raw matched **text**. When that text contains a
bracket character class the glob does not match itself — `[0-9]` as a glob
matches ONE digit, not the four literal characters — so the replacement is a
no-op, the `while` condition still matches, and the loop spins forever.

Found by the gw_detection driver after an orphaned copy of this hook
(`ppid 1`, state `RN`) burned 98.9% of a core for **7 days 7 hours**. Both
repos carry the same hook; cogwheel's was identical and unfixed.

**Why cogwheel was especially exposed:** AGENTS.md *mandates* the bracket
idiom for process checks (`pgrep -f "pytest [c]ogwheel"`, so the check does
not match itself). Reproduced locally, `echo $(pgrep -af "[s]dk/build.py")`
hung; so did `$(grep [0-9] f)` and `$(ls [ab])`. The trigger is narrower than
"any bracket" — the class must sit INSIDE a command substitution or backticks,
which is why bare `pgrep -f "[s]dk/..."` calls returned normally and the bug
stayed invisible for a week.

**Fix** (matching gw's, deliberately — one shared pipeline, one shared fix):
guard each loop on the string actually shrinking,
`[[ "$stripped" == "$_prev" ]] && break`. A pathological command is then left
un-normalized and judged as-is, which is refusal-conservative: it can only
produce a spurious DENY, never a spurious ALLOW.

**Two process lessons, both about verification, not bash.**
1. The patch went in via sidecar + atomic `mv`, never in place: this hook runs
   on EVERY tool call and bash reads scripts incrementally, so an in-place
   rewrite can corrupt a live instance.
2. `bash -n` is not enough. My first patch draft declared `local _prev` inside
   the top-level `case` branch — a RUNTIME error (`local` outside a function)
   that `bash -n` passes clean, and that would have broken every tool call.
   The gate must be a FUNCTIONAL probe: drive the candidate with real payloads
   under `timeout`. And read allow/deny from stdout, never from the exit code
   — `deny()` prints its JSON and still exits 0, so my first probe scored a
   correct DENY as an ALLOW. 20 payloads now cover the three former hangs plus
   every allow/deny decision.

## F045 — the HEAD-oracle antipattern spreads by REUSE, and a scan of ADDED lines cannot see it (2026-07-30)

**Where:** `test_lensing_surrogate_training.py::WedgeEdgeSelfFalsification`
(build 1d); `.claude/hooks/check_head_relative_tests.py`.

F043 was recorded on 2026-07-30 and a pre-commit guard shipped the same day
(`1f6a907`). The NEXT build reintroduced the antipattern and the guard let it
through. Both new self-falsification tests failed within the hour, exactly as
F043 predicts — `gap=0.000e+00` where the test demanded a gap, and
`2.9072886962267335 not less than 2.9072886962267335`, because "HEAD" was now
the post-change tree comparing against itself.

**Why the guard missed it.** It scans the ADDED lines of staged test files for
`git show HEAD:` and friends. That was a deliberate choice — it stops a
pre-existing oracle from re-firing on every unrelated edit to the same file.
But the only HEAD-relative line these tests ADDED was
`head = _head_training_module()`, a call to a helper committed weeks earlier.
The pattern never appeared in the diff. **A guard keyed on the introduction of
a construct is blind to its propagation.** Widened (2026-07-30): resolve the
file's HEAD-oracle helpers by AST first, then treat a CALL to one as the same
finding. Verified against a replay of the exact lines that got through.

**The second lesson is where it appeared.** The build's GATES were clean — it
froze `_WP1_INCUMBENT_CLOSURE_GAP` and `_WP1_INCUMBENT_SPAN` as literals and
its own docstring says "no live `git show` oracle in the gates". The rot was
in the META-TESTS written to prove those gates were reachable. Reaching for
the previous commit feels natural there in a way it does not in a gate,
because the claim genuinely is "the OLD code fails this". Discipline applied
to the primary assertion does not automatically reach the test that validates
it.

**The honest form of a self-falsification test** is to reconstruct the
counterfactual LOCALLY from today's engine, never to fetch a past commit. The
closure test now rebuilds the inset sweep inline (six lines of
`critical_point` calls) and shows it fails to close — which additionally
reproduces the frozen incumbent gap to five places, giving that literal its
provenance without git. Where the counterfactual is a whole pipeline and
cannot be reconstructed cheaply (the arc-span test drove
`detect_caustic_structure`), the residual claim is only the PROVENANCE of a
frozen number: retire the test and let the gate's own form carry the
reachability, since a gate asserting `span > incumbent_literal` goes red by
construction when the change is reverted.

**Rule.** Before adding an absolute standoff to keep a sampler off a
singularity, ask whether the singularity is in the OBJECT or in the
PARAMETER. A divergence whose rate is a clean power of the distance to a
branch point is the signature of the latter, and the fix is the coordinate
that removes it (here `s = sqrt(theta_max - theta)`, the same reparametrising
move as `u = sqrt(eta)` on the fold axis), not a margin. This is the wedge
edge's entry in [[lensing_collocation_from_local_scales]].

## F055 — the build watchdog failed OPEN for three days while the launcher printed "(watchdog 1200s)" (2026-07-30)

**Where:** `.claude/sdk/watchdog.sh:48`, `.claude/sdk/launch_build.sh`.

    ORCH_PID=$(pgrep -nf '\.claude/sdk/cli\.py build')   # the entrypoint is build.py

`build.py` is a 27-line shim that calls `sdk.cli.main()` **in-process**, so the
orchestrator's argv reads `.../.claude/sdk/build.py build` and never contains
`cli.py`. The watchdog therefore found nothing, logged `ERROR: Could not find
orchestrator process` into `<log>.watchdog.log`, and exited 1 — three seconds
after every launch.

Survey of `/tmp/*.watchdog.log`: **ok through 2026-07-20, dead on every build
from 2026-07-27 on** (`tiling_correctness`, `saddle_lobe_serve`,
`saddle_born_carrier`, both `wedge_standoff_and_tube_normal` runs,
`tube_arclength_coordinate`, `analytic_caustic_reach`). Six builds, including
two run concurrently, with no kill protection.

**Why it worked before and not after.** The pgrep line never changed (06-05);
`launch_build.sh` has invoked `build.py` since it was added (07-15). Builds
before 07-21 were HAND-launched as `python .claude/sdk/cli.py build`, which the
pattern did match. The hook that now mandates `launch_build.sh` closed off that
form — **the automation that replaced the manual step silently disarmed the
guard the manual step had been feeding.** Nothing announced the change because
nothing tied the two names together.

**Three defects, one line.**

1. *Wrong name.* `_retry_until_launch.sh:70` already greps the correct
   `[s]dk/build.py`. The entrypoint name was knowledge held in two places, and
   one copy rotted (DRY).
2. *Wrong process.* `pgrep -n` takes the NEWEST match, so two concurrent builds
   get two watchdogs both guarding the newer one. Measured worse than that:
   with two fake builds running, `-n` selected a THIRD pid — an unrelated
   shell whose command line merely quoted the pattern. The old discovery could
   have SIGKILLed a process that was not a build at all.
3. *Fails open, silently.* The launcher printed `launched: <log> (watchdog
   1200s)` unconditionally. The failure evidence went to a sidecar log nobody
   reads. **This is why it survived three days**, and it is the defect worth
   generalising: a guard that reports success it did not verify is worse than
   no guard, because it also suppresses the search for one.

**Fix.** Discovery is keyed on the LOG PATH: it is unique per build and already
on the orchestrator's own command line, so the match is exact, race-free, and
indifferent to how the interpreter was invoked. `launch_build.sh` also passes
`$!` (PYBIN is the absolute env python, so `$!` IS the orchestrator here) and
the watchdog treats a disagreement between the two as a reportable NOTE rather
than a silent choice — that mismatch is the signature of a launcher that
backgrounds a WRAPPER, where killing the PID leaves the build running.

The log-path key came from the gw pipeline, which needs it: its launcher goes
through `conda run`, so `$!` there is the conda wrapper, which (its own `cli.py`
documents) "survives a subtree kill and reads as alive". A verbatim port of the
`$!`-only fix would have given them a watchdog tracking a PID that never dies.
**The general lesson is about porting between the two repos: a fix travels, but
the assumption it rests on may not. Port the invariant, not the line.**

The launcher then WAITS for `Watching orchestrator PID <pid>` in the watchdog
log and prints a loud WARNING if it never appears, killing defect 3. `.claude/sdk/verify_watchdog.sh` is the
permanent probe (11 assertions, ~12 s): a stalled fake orchestrator must really
die (rc 137 + kill marker), a healthy one must exit 0 unkilled, a dead PID must
be refused, and — the F055 guard — the entrypoint name extracted from
`launch_build.sh` must match the pattern extracted from `watchdog.sh`, with a
non-vacuity assertion (an empty pattern makes `grep -E` match everything, so
the test would pass loudest exactly when the pgrep line was deleted) and a
retained contrast control asserting the old `cli.py` pattern does NOT match.

**Rule.** Any guard whose arming can fail must PROVE it armed before the thing
it guards is reported as protected, and any invariant spanning two files needs
a test that reads BOTH files — not one that re-states the invariant a third
time. Compare F049 (a guard placed where it could not run) and F052 (a tier no
routine job ran): same failure class, third instance this week — the automation
existed, was believed, and did nothing.

## F056 — four loop terminators, one empty list: an unreadable Inspector result was an infinite build (2026-07-30)

**Where:** `orchestrator.py::_parse_inspector_result` (text fallback) and
`_run_inspector_with_loop`; `gates.py::should_escalate`,
`revision_budget_spent`, `finding_signature`.

The text fallback returned `InspectorResult(verdict=ISSUES, findings=[])`
whenever the result text carried no JSON block and no bare `PASS` line. That
state cannot be escaped:

| terminator | why it cannot fire on `ISSUES` + `[]` |
|---|---|
| `check_inspector_gate` | verdict is ISSUES, so the loop is entered |
| `should_escalate` | requires an IMPLEMENTATION/DESIGN finding |
| `revision_budget_spent` | literally `bool(findings) and loop > MAX` |
| non-convergence | `if _signature and ...`; `frozenset()` is falsy |

Four independent exits, all keyed on the same list, and the parser's own
failure mode manufactures exactly the value that disables all four. The
revision counter runs past its budget with nothing to report and nothing to
fix: `0 trivial, 0 impl, 0 design`, forever.

**Observed:** `analytic_caustic_reach`, 2026-07-30. `inspector-12` hit a
transport wedge at 15:07, retried on the resumed session, and completed TWICE
(15:15:35 `$5.82`, 15:15:41 `$6.02`). The retry's final message arrived without
the JSON block. The loop logged `revision 3/2 (0 trivial, 0 impl, 0 design)`,
spawned `inspector-13`, and the build died with no terminal marker, no report,
and no commit — ~$60 of completed work left uncommitted on disk.

**The same bug, twice, and the fix carried the hole.** `gates.py:204` records
`revision 8/2` on 07-28 (Born carrier, ~26 min and ~$24 of Inspector cycles
with zero implementation findings). The remedy then was `revision_budget_spent`
— which was written as `bool(findings) and loop_count > MAX_REVISION_LOOPS`.
The new guard reproduced the exact precondition of the bug it was added to fix.
A budget that only applies when there is something to spend it on is not a
budget.

**Fix.** Name the parse failure AS a finding (IMPLEMENTATION severity, stable
round-invariant description) instead of adding a fifth guard: one honest
finding re-arms all four existing exits at once, and a fifth guard is a fifth
thing to forget. Severity matters — TRIVIAL would let `revision_budget_spent`
flip the verdict to PASS and ship work whose inspection was never readable.
`tests/test_inspector_parse_failure.py` pins it, including the contrast control
that none of the four fired on the old empty result.

**Rule.** When several guards all read the same value, they are ONE guard.
Count the *distinct inputs* your terminators consume, not the terminators. And
a retry that resumes a session can return a DIFFERENT SHAPE than the original
call — every parser downstream of a retry needs a defined answer for "the text
is real but the structure is missing", and that answer must never be a verdict.

## F057 — a test spec that names no file was free work with no budget (2026-07-30)

**Where:** `orchestrator.py::_group_test_specs`, `_test_dev_budget`.

Any domain-test description not literally quoting a `test_<x>.py` filename was
routed to `cross_suite` — appended to the agent's prompt as REAL WORK, counted
by neither the shard cap nor the `60 + 20*n` budget.

1e-farfield's approved plan carried 11 substantive descriptions; **10 named no
file.** The sharder logged `1 spec(s) in 1 shard(s) (cap 3/agent)` and budgeted
**80 turns for a load needing 250 across 4 shards**. `test_dev-6` and
`test_dev-7` each burned 80 turns and returned ZERO characters; the build died
with 1352 lines of production code written and not one test authored.

Measured from the logs: `test_dev-6` made its first WRITE at tool call **61 of
80** (76% of budget spent orienting), `test_dev-7` at call 62. They were starved
before they began. The shard cap exists precisely to prevent that death — its
own comment cites 7b, the 8a near-miss and 8b-levers — and never engaged,
because the quantity it keys on had been collapsed upstream.

**Fix.** `cross_suite` is for genuinely UNIVERSAL rules ("ALL SUITES: 79
columns"), cheap to repeat and needing to reach every shard. A substantive
description is one test's worth of work: with exactly one suite in play it can
only belong there, so it is assigned and COUNTED. Cross-suite specs now scale
every shard's budget. The first attempt folded ALL unscoped specs in, which
would have applied a universal style guard to one shard instead of all —
`test_test_dev_split` caught it, so the distinction is now an explicit
predicate with tests on both sides.

**Not just test_dev.** The same logs show `coder-4` (110 tool calls / 105
turns) and `coder-2` (156 / 95) exhausting while making their first edit at 8%
and 15% — they started fast and ran out doing real work. Coder budgets are a
free-form Architect estimate that does not scale with the WP's declared file
count, while `_test_dev_budget` scales with spec count. The mechanism exists on
one side of the DAG and not the other.

## F058 — every guard that failed today was disarmed by a quantity its producer controlled (2026-07-30)

Three guards failed in one day. Each was correctly written, each had a passing
test, and each was silently disabled by an upstream detail nobody had connected
to it.

| | the guard | keyed on | who could collapse it |
|---|---|---|---|
| F055 | watchdog kills a wedged build | the orchestrator's process NAME | `launch_build.sh` chose the entrypoint |
| F056 | 4 revision-loop exits | a non-empty `findings` list | the Inspector parser's fallback |
| F057 | shard cap + turn budget | a filename appearing in prose | the Architect writing the description |

    guard_engages = f(X)
    ...where X is produced upstream, incidentally, by something that does not
    know the guard exists.

No alarm fires when `X` reaches the disabling value, because staying quiet IS
the guard's success state. Silence means both "working" and "disarmed", and the
two are indistinguishable from outside.

**What made each expensive was the false assurance, not the defect.**
`launch_build.sh` PRINTED `(watchdog 1200s)`. The revision loop PRINTED
`revision 3/2` as though a budget were enforced. The sharder PRINTED
`1 spec(s) in 1 shard(s) (cap 3/agent)` — quoting the cap it was failing to
apply. Each told the reader the guard was working, in the moment it was not.

**Rule.** Do not key a guard on a quantity the upstream producer sets
incidentally. Prefer one it cannot avoid emitting (a log path is unique per
build and already on the command line; an absent finding is representable as a
synthesized finding). Where that is impossible, the guard must ANNOUNCE ITS OWN
ARMING with the value it armed on — `guarding PID 1494873`, not
`(watchdog 1200s)` — so the log carries evidence rather than assertion.

**Corollary.** Each of these had a test built from the author's IDEA of the
upstream input; none used the shape the real producer emits. A guard's test
needs one case constructed by the REAL upstream path. Compare F049 (a guard
placed where it could not run) and F052 (a tier no routine job ran).

## F059 — a build waiting on a human decision is indistinguishable from a wedge (2026-07-30)

**Where:** `gates.py::_file_based_approval`, `_file_based_escalation`.

Both gates polled for a decision file in SILENCE. The watchdog's only liveness
signal is log mtime, so a healthy build blocked on the driver looks exactly
like a hang. At 21:55:22 the watchdog killed `1e_farfield_port` mid-escalation,
after precisely 1200s — the time it took the driver to diagnose the finding the
build was asking about.

`launch_build.sh` had carried the warning in prose for weeks: "Respond
promptly: the watchdog staleness clock runs during the wait." That asks a human
to be fast; it does not resolve the conflict. It went unnoticed because the
watchdog had been dead since 07-27 (F055) — **repairing one guard exposed a
latent conflict with another, and the first healthy build the fixed watchdog
ever saw is the one it killed.**

**Fix.** `_gate_wait` owns both poll loops and emits a heartbeat every 4
minutes: `still waiting for <what> (Nm elapsed) — build is alive, not stale`.
The log advances, and the wait becomes VISIBLE with elapsed time instead of
reading as silence. `tests/test_gate_heartbeat.py` asserts what the watchdog
actually reads — that something reaches stdout well inside the 1200s window,
that beats repeat, that a promptly-answered gate stays quiet — with a contrast
control proving the old loop emitted nothing across the entire kill window.

**Rule.** Any state where the pipeline BLOCKS on an external decision must emit
a liveness beat, or every watchdog downstream will read it as death. Waiting is
not idling.

## F060 — normalizing the far-field `d` axis by curvature radius is wrong physics, wrong chart, and breaks separability (2026-08-03)

**Where:** `cogwheel/lensing/surrogate.py` — far-field chart `d`-axis grid design.

A Professor+Simplifier evaluation (build `eval_d_norm`, 2026-08-03) tested
whether normalizing the far-field `d` grid by the caustic curvature radius
`R_c(gamma, theta)` would reduce interpolation error. Rejected on five
independent grounds:

1. **Wrong physics.** The Airy transition scale is `ξ = (3wΔτ/4)^{2/3}`,
   not `d/R_c`. Normalizing by `R_c` alone cannot collapse the fold structure,
   which also depends on `w`, `b₃`, and `λ_h`.
2. **Wrong chart.** The far-field chart operates at `d >> R_c`; the regime
   where `d ~ R_c` is served by the tube chart. There is no interpolation
   benefit to relativizing `d` to a length scale the far-field chart never
   approaches.
3. **Breaks tensor-product separability.** `R_c(gamma, theta)` couples the
   spatial and parameter axes. The surrogate architecture requires a
   separable spline grid; mixing axes is architecturally forbidden.
4. **Numerical instability near cusps.** `R_c` diverges near cusp points,
   introducing instability in regions the chart already excludes via cusp
   windows.
5. **Non-problem.** The current 4 `d`-nodes achieve `eps < 1e-3`. The
   `~2×` `R_c` variation within a gamma band is smooth monotone drift
   absorbed by the `γ`-axis directly.

**Rule.** If accuracy tightens to `eps < 1e-4`, add 1–2 `d`-nodes (50%
training cost, zero serve cost, no architecture change) before considering
any axis redefinition. Never redefine a grid axis by a quantity that couples
to a perpendicular axis.

## F061 — `f_schwinger` above `w = 60` costs 250x more per call, and four tests walked into it (2026-08-06)

**Measured**, calling the shipping `f_schwinger` at `y_eig = (0.30, 0.15)`,
`gamma' = 0.42`:

| `w` | path | seconds/call |
|---|---|---|
| 10.0 | double-double | 0.172 |
| 40.0 | double-double | 0.187 |
| 59.0 | double-double | 0.336 |
| **61.0** | **mpmath** | **84.536** |
| **70.0** | **mpmath** | **111.352** |

The routing (`_schwinger.py:940`) is deliberate and documented: `w <= 60`
(`W_CEILING_SCHWINGER`) takes the double-double path, `60 < w <= 150`
(`W_CEILING_SCHWINGER_QD`) dispatches to `_f_schwinger_mpmath`, and `w > 150`
is an unconditional refuse whose stated reason is that "mpmath runtime
`O(w * dps^2)` exceeds training budget".

**The trap is the middle band, not the ceiling.** The ceiling at 150 implies
runtime below it is affordable. It is not: the cliff is at 60, and it is a
factor of ~250 at the very BOTTOM of the mpmath band. Nothing refuses in
`(60, 150]` — it serves, slowly, forever.

**How it bit.** `test_lensing_wedge_dd_arclength.py::DDWCeilingTestCase` says
in its own docstring:

> the DD cap gives `w_max ~ 121.6`, which is below the requested 500 but
> above the Schwinger ceiling (~60). Most refusals at the capped `w_max` are
> Schwinger-related (not DD) ...
> Cost: 4x4x4 = 64 nodes x ~13 w-points x 30ms ~ 25s.

Both halves are wrong in the same way: the author expected `w in (60, 150]`
to REFUSE (cheap) and budgeted 30 ms per evaluation. It SERVES at ~85-120 s
per evaluation, so a fixture budgeted at 25 s is hours of work. Three other
tests reached the same band by other routes
(`test_lensing_marginalized_likelihood.py:839`, `test_lensing_prior.py:1064`,
`test_lensing_saddle_likelihood.py:463`).

**Two thresholds, one gap.** The DD product cap (`w * |y| < 58`) and the
Schwinger DD ceiling (`w <= 60`) are different quantities. A geometry can
satisfy the DD cap at `w_max = 121.6` while every node above 60 falls into
mpmath. Any fixture that derives its `w` range from the DD cap alone will
walk into the expensive band without noticing.

**Rules.**
1. A fast-tier test must keep `w <= W_CEILING_SCHWINGER`. If a test's PURPOSE
   is the mpmath band, it belongs in a slow tier, and its cost must be
   budgeted at ~100 s per evaluation, not 30 ms.
2. Never infer "cheap" from "a ceiling exists above me". Check which side of
   the *dispatch* threshold you are on, not the refusal threshold.
3. Cost comments in fixtures are load-bearing and rot silently. `~30ms` here
   was off by 3-4 orders of magnitude and nothing checked it.

**Consequence.** Because the tree gate had no per-test timeout, these four
tests did not fail — they pinned workers until the gate burned its 3600 s
ceiling at ~88% and STRANDED a build that had already passed Inspector and
Professor, without naming a single test. Both gates now pass
`--timeout --timeout-method=signal` (`run_full_suite.sh`,
`orchestrator.py`'s tree gate). Verified: the previously-unbounded
`DDWCeilingTestCase` now errors in 92 s naming all six tests, while the other
14 tests in its file pass. A gate that cannot COMPLETE hides ordinary red as
effectively as it hides the hang — 11 unrelated failures were sitting behind
this one (see `todo.d/lensing_serving_ladder_guards_are_red.md`).

## F062 — careful staging never protected the commit message, because `git commit -m` takes the whole index (2026-08-07)

`git commit -m <msg>` with NO pathspec commits the entire INDEX, not the
paths a preceding `git add` named. Every mitigation in this repo had been
aimed at `git add`: F047 replaced the SDK's blanket `git add -u` with a
selective `_stage_build_output()`, and the driver's rule was "name paths,
never `-A`". Both address the wrong call. If anything is already staged when
the commit runs — by the operator, by an earlier agent, by a hook — it lands
under that message regardless.

**Measured, one session (2026-08-06/07).** Seven commits carried content their
messages disown:

- Five driver `spec:` commits swallowed a concurrent build's `surrogate.py`
  and `surrogate_training.py` work. `spec: retire the arc-length field names`
  carries +186 lines of cusp-axis implementation.
- `memory: consolidate short-term into long-term (Dreamer, 2026-08-07)`
  carried the driver's in-flight `DATA_CONTRACTS.yaml` major bump, `SPEC.md`,
  three rendered changelogs, and four new fragments. The Dreamer's `git add`
  named only `.serena/memories/*` paths — correctly — and it made no
  difference.
- `chore: update agent state after build` carried the build's ENTIRE
  production diff (`geometry.py`, `surrogate.py`, `surrogate_training.py`,
  five test files) — after the tree gate had BLOCKED that build's commit. A
  gate that blocks the commit step does not block a later step that commits.

Nothing was lost in any case; the cost is a history that cannot be read or
bisected honestly, and a contract bump attributed to memory housekeeping.

**Rules.**

1. Commit with an explicit pathspec: `git commit -m <msg> -- <paths>`. It
   ignores the rest of the index, so concurrent work cannot be captured. This
   is the only mechanical fix; discipline about `git add` is not one.
2. The index is a SHARED MUTABLE RESOURCE across the driver and every build
   agent running in the same tree. Treat a non-empty index at commit time as
   a conflict to report, not a state to inherit.
3. A blocked gate is not a blocked build. Audit every step that can reach
   `git commit`, not just the one the gate guards.

**Partially fixed, and the pathspec alone is NOT sufficient.**
`orchestrator.py::_git_commit` now snapshots the staged set before staging,
commits `-- <only its own paths>`, and logs any pre-existing staged paths it
excluded. In a scratch repo with no hooks this is exactly right: with `A` and
`B` both staged, `git commit -m msg -- B` commits only `B` and leaves `A`
staged.

**The second path in is the pre-commit hook.** This repo's hook runs
`sync_derived_docs.py --check` and then AUTO-FIXES AND AUTO-STAGES the
rendered surfaces (`TODO.md`, `SPEC_CHANGELOG.md`,
`DATA_CONTRACTS_CHANGELOG.md`, ...). Those `git add`s happen DURING the
commit, and for a partial (pathspec) commit git folds the hook's staged paths
into the result. Measured 2026-08-07: a commit explicitly scoped to
`-- .serena/memories/ .claude/agent_state/` produced 30 files, including the
whole `.claude/spec/` tree, because pending fragments made the rendered docs
stale and the hook synced them mid-commit.

So the reliable rule is about ORDER, not pathspec:

4. Commit `.claude/spec/` (fragments AND their rendered outputs) FIRST, in its
   own commit, so the hook has nothing left to sync when later commits run.
   A pathspec commit made while any fragment is unrendered will absorb every
   derived surface the hook touches, whatever paths you named.

## F063 — a non-deterministic failure cannot be confirmed fixed by a single passing run (2026-07-16, salvaged 2026-08-07)

Salvaged from `.claude/handoff/lensing/META_PLAN.md` before that journal was
deleted. Recorded here so the hypotheses are not re-derived.

**FALSIFIED — do not act on these.**

1. *"Builds cannot write files — a permission layer denies all writes."* FALSE.
   The grants in `settings.local.json` were expanded from 1 to 55 rules and the
   next build failed identically, because SDK agents use
   `setting_sources=["user"]` and NEVER read project or local settings. Agent
   permissions come from `AGENT_PERMISSION_MODES` (Phase 2+ is
   `bypassPermissions`); `settings.local.json` affects only the human's
   interactive session.
2. *"The sandbox denying out-of-workspace `/tmp` writes is the root cause, and
   `ignoreViolations` fixes it."* FALSE, and it was stated as verified off ONE
   observation. Tested properly the denial is NOT positional (4/4 sequential
   `/tmp` writes succeed), NOT content-dependent (the byte-for-byte denied
   command replays clean 3/3), NOT hooks, NOT deny-rules. It is transient and
   external — "The user doesn't want to take this action right now" is the
   harness's wording for a refused permission REQUEST — and it struck different
   coders at different call indices. NOT REPRODUCIBLE ON DEMAND. The
   `ignoreViolations` change was REVERTED in all three repos (cogwheel 8aa96c2,
   skill 426f29f, gw c4c4e354): it loosened the sandbox for no demonstrated
   benefit. Do not reinstate without measuring a denial RATE with and without
   it over many trials.

**THE LESSON, which generalises well beyond permissions.** If a fix cannot be
shown to change a RATE, it has not been shown to do anything. A single green
run after a change is not evidence when the failure was intermittent to begin
with. This is the same discipline the repo already applies to numerics —
report a distribution (p50/p90/max) and a worst-case locus, never a bare max —
applied to infrastructure.

**Operational facts worth keeping from the same journal.**

- Serena SSE orphans hold port 8322 and kill the next launch with "SSE server
  exited during startup (rc=3)". Diagnose with `lsof -tiTCP:8322`; kill the
  `--transport sse --port 8322` process, NEVER the session's own
  `--project-from-cwd` stdio server.
- SIBLING-GREP EVERY PATH FIX. `_run_hook_script` resolved its script path with
  two `dirname`s instead of three, yielding `<repo>/.claude/.claude/hooks/...`
  — a path that never exists — so hooks NEVER fired in cogwheel OR gw, and
  `hook_trace.log` had never been written anywhere. `gw`'s `_build_env` carried
  the identical defect (silently loading no `.env`, so `GW_*` vars never
  reached subagents) and was found only because gw's own agent looked, after
  one instance had been fixed without grepping for siblings.
- The zero-write builds were caused by the BRIEF, not the SDK: briefs demanding
  test-authoring work packages turned Coders into experimentalists who wrote
  `/tmp` probe scripts and stalled. Measured tool-call profile, cogwheel's
  failing build vs 24 gw build logs (149 coder calls): gw WRITE 26% / SHELL
  16% / zero `/tmp` probes, against build1b WRITE 1% / SHELL 60% / four `/tmp`
  probes. Fixed in `.claude/crew/architect.md`, which now forbids
  test-authoring WPs.

## F064 — normalising a chart radius by a caustic radius drags the cusp singularity to every radius (2026-08-06, generalised 2026-08-07)

Salvaged from the coordinate-program spine when that file was reduced to
ordering links. This is the defect the whole coordinate program exists to
cure, and it has now appeared twice in two different chart classes.

**The pattern.** A chart interpolates on a NORMALISED radius
`r = |y| / r_caustic(gamma, theta)`. Near a cusp the caustic reach behaves as
`r_caustic ~ const - c * d**(2/3)` in the angular distance `d` to that cusp.
Dividing by it therefore injects a `theta**(2/3)` non-smoothness into EVERY
radius at once, `w`-independently — the singular structure is no longer
confined to the neighbourhood of the cusp, it is smeared across the tile.

Curing it is a COORDINATE CHANGE, not more nodes. Splining the angular axis in
`u = d**(2/3)` measured 171x better than arc length on the wedge
(4.88e-2 -> 2.85e-4 on a 1-D transverse cut). Adding nodes in the bad
coordinate buys the usual 2-5x per halving and never closes the gap.

**Confirmed instances.** `InteriorWedgeChart` (`r = |y_eig| / r_caustic`,
four astroid cusps) — measured and cured 2026-08-06. `LobeInteriorChart`
(`rho_lobe = |y - centroid| / r_deltoid(theta_local)`, THREE deltoid cusps per
lobe) — predicted, not yet measured; see
`todo.d/lensing_saddle_forensics.md` item (a).

**The companion defect, which is why it stayed hidden.** A tiler with no eps
feedback cannot discover that it needs more tiles. Both instances were
invisible until someone read the eps DISTRIBUTION rather than its max — a
max-metric summary hid the wedge defect for a full day. Report p50/p90/max and
the worst-sample LOCUS in any chart diagnostic, never a bare max.
