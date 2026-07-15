# Coder short-term checkpoint

## Task: WP4 channels.py + _gauge.py (2026-07-16) — PARTIAL, BLOCKED on WP3

### WP4 IS BLOCKED — channels.py NOT written, deliberately
Plan declares `WP4 Dependencies: WP1, WP3`. Present in the tree:
geometry.py (WP1) + _dd.py. ABSENT: `_hyp1f1.py` (WP2) and `operator.py`
(WP3). The tracker's `exact` total comes from `_operator_total`, which the
plan has WP3 hoisting out of the prototype's topology_stable and re-exposing
"as an optional time-origin argument on the grid evaluator" — signature NOT
pinned anywhere (WP3 also adds W_cut, a `branch` flag, a refusal exception,
OperatorDiagnostics). Writing channels.py now = inventing an API + ZERO
runnable gates (reconstruction / continuity / path-independence / perf ALL
need exact_total). Did not guess. Delivered the genuinely unblocked half.

### DELIVERED: _gauge.py (5 primitives) + test_lensing_gauge.py (34 tests)
Zero-dependency (numpy only) — WP5 imports it too. Named the test file
test_lensing_gauge.py NOT test_lensing_channels.py (naming a suite after a
module that doesn't exist is a lie); channels gets its own suite at WP4 resume.

### KEY RESULT — channels.py will NOT need its own projection
`exact_transition_channels(w, total, tau_cluster=mean(delays), delays, H,
S_2d, weights)` is **BITWISE identical** (0.0 diff, 300 random configs) to the
prototype's `_exact_four_channel_projection`. I generalized `switch` to accept
(n_w,) [per-frequency] OR (n_w, n_members) [per-channel — the shape the tracker
builds]. So the tracker reuses the primitive; do NOT port a 4th copy.
NOTE the 1-D/2-D switch semantic is ambiguous iff n_w == n_members; documented
as per-frequency. Watch for that in a 4-w-node test.

### Numbers MEASURED (not assumed) — feed these to Inspector/FINDINGS
- Plan's F003 is CONFIRMED empirically: with |H| ~ 1e8 (near-fold sqrt|mu|),
  reconstruction err = 6.0e-8 — a flat 1e-12 gate (the brief's hard req 1b)
  fails by 4 orders of magnitude, while C=100 scale-aware bound holds with
  ~30x margin (ratio 0.034). `NearFoldScalingTestCase` asserts BOTH halves,
  incl. a test that goes red if the flat gate ever starts passing.
- Plan's "permutation invariance <= 1e-15" MEASURES 9.93e-16 for O(1)
  kernels — passes by 0.7%. It is a fixed-config number, NOT a robust bound
  (4 O(1) terms => reorder error ~ ulp(4) ~ 8.9e-16). Suite asserts the
  scale-aware bound FIRST, then pins 1e-15 as a frozen backstop.

### Real bug I found + fixed in my own first draft
`exact_transition_channels` normalized weights, then passed them to
`unresolved_member_channels` which normalized AGAIN — dividing an already-
normalized vector by its own float sum re-rounds it. Harmless to exactness
(projection is exact for ANY weights, so tests stayed green — this is exactly
the class of thing green tests cannot see) but it was the entire 2.5e-16 gap
vs the prototype. Fixed via private `_member_split` (normalize exactly once);
that is what made the agreement bitwise.

### Mutation check (WP2's lesson applied: green != evidence)
9 mutants, ALL RED: drop projection (grid + scalar), conj(carrier)->carrier,
weights unnormalized, smootherstep->smoothstep(C1), switch sign flip, cluster
demod sign, split loses relative phase, reconstructed_total carrier sign.
The C2 test discriminates by asserting the 2nd difference at the join vanishes
LINEARLY in h (C1 smoothstep would tend to 3/W^2 = 7e-2).

### Blockers / gaps for the orchestrator
- WP3 (and WP2 `_hyp1f1.py`) must land before WP4 resumes.
- **WP1 shipped geometry.py with NO test_lensing_geometry.py** — the plan's
  WP1 verification (168-row CSV regression, y=0 analytic, Morse census)
  appears never to have run. Whole cogwheel/lensing/ tree is still untracked.
- pylint AND pycodestyle absent from every conda env — "pylint clean" gate is
  NOT verifiable here; verified 79-col + ast.parse programmatically instead.
- Serena MCP server disconnected mid-task; used built-in tools per fallback.

## Task: WP2 double-double primitives `_dd.py` (2026-07-16) — DONE

- `_dd.py` + `test_lensing_dd.py` already existed untracked from a prior run.
  Reviewed rather than rewrote: the ARITHMETIC was correct (Knuth TwoSum,
  Veltkamp split @ 2**27+1, QD ieee_add / accurate_div, Smith complex div,
  four-flat-scalar dd-complex). Kept it. The TESTS were the problem.

### The Professor's warning was exactly right — and it was the tests, not the math
Ran the mandated mutation check (perturb `_SPLITTER`, confirm suite goes red).
The suite as found stayed **GREEN** for 2**27, 2**27+2 AND 2**28+1. Only
2**26+1 went red. Cause: exactness tests drew ~81 uniform random values, and a
mis-split only corrupts `_two_prod` for specific significand bit patterns.

### Key numerical finding (worth promoting to FINDINGS.md)
What is load-bearing is NOT the literal constant but the **split point**: both
halves must carry <= 26 bits so every cross product fits float64's 53-bit
significand.
- 2**27, 2**27+1, 2**27+2 -> halves 26/26 -> all FUNCTIONALLY VALID splitters.
- 2**26+1 -> 27-bit high word; 2**28+1 -> 27-bit low word -> BOTH silently
  destroy the error-free product.
So a suite staying green for 2**27 is CORRECT, not a gap. Don't "fix" that.

### Changes made
- Tests: adversarial full-53-bit significands (`_adversarial_significands`,
  low bit forced); exact integer `_significand_bits` (replaced `np.log2`
  exponent arithmetic that underflows to 0 -> vacuously true width check);
  `SplitterSensitivityTestCase` permanently encodes red-ability (+ verified
  falsifiable by mislabelling a healthy splitter as broken);
  `SignificandBitsTestCase` pins the load-bearing helper.
- Vacuity guard in `DdTestCase.tearDown` — **caught a real vacuous test**:
  `test_div_does_not_overflow_on_large_denominator` computed
  (1+i)/(1e300+1e300i)=1e-300, below `_DD_MIN_NORMAL`=2**-969, so its accuracy
  assertion was ALWAYS skipped; only `isfinite` ran. Split into a
  representable-quotient accuracy test + an explicit sub-normal-floor
  finiteness test.
- `dd_div`/`dd_complex_div` docstrings claimed "yields IEEE inf/nan, as for
  float64 — caller owns the guard". FALSE: they RAISE ZeroDivisionError (both
  in the interpreter and under njit's default error_model='python'). Fixed the
  docstrings + added tests pinning the raise. WP3 would have been misled.

Result: 37 tests pass; 2**28+1 and 2**26+1 now go RED; valid splitters green.

### For later WPs
- `@njit` still DEFERRED (per brief). When added: numba freezes module globals,
  so `_SPLITTER` stops being patchable. `_py()` helper resolves `py_func` but
  only one level deep — once `_two_prod` is jitted its call to `_split` binds
  the compiled version, and the mutation test needs `_dd` reloaded under patch.
- Leaf module confirmed: imports pull in no lal/numba/scipy.
- pylint NOT installed in any conda env — 79-col limit verified by hand (awk).
  Serena MCP tools unavailable this session; used built-in tools per fallback.
