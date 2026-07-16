# Microlensed-PE program — meta-plan and live status

Driver: the session agent (meta-planning only; each build's Architect owns its
internal plan). User directive 2026-07-16: drive autonomously to the finish line,
approve/reject plans via the file gate, give feedback, no further user input.

## Source materials
- Paper + tested prototype: `.claude/spec/lensing_paper/` (tex, pdf, code/, data/).
  UNPUBLISHED — stays in .claude/ (excluded from main sync).
- Professor topic memories: `professor/microlensing_chang_refsdal` (design manual),
  `professor/likelihood_and_inference`, `professor/marginalization`,
  `professor/priors_and_coordinates` (coordinate recipe + folding).

## Design decisions locked in with the user (2026-07-16 thread)
1. Multi-component RB with image-delay phases analytic; K_a interpolated.
2. Summary structure: NEVER product-of-summaries. Rapid×rapid stays inside the
   f-sum as delay-continuous summaries T^(p)_mn,b(δt) (and data-side A^(p)_m,b(δt)
   ≅ z_m(t) timeseries); slow fiducial K_a0*K_c0 Taylor-expanded within bins
   (costs one extra moment: p ≤ 3 for the norm term).
3. Hot path: NO FFTs (setup only). Sequential contraction — modes first
   (M² × few δt-grid nodes × bins), then images via envelope interpolation at
   10 pair delays (n_img² × bins). Additive M²+n_img², NOT multiplicative.
   Contraction must stay subdominant to the coarse-node waveform call.
4. δt vs Δt: fiducial absolute delays exact inside summaries; candidate residual
   δt handled by (v1) linear RB + lens-aware bin criterion + guard assert
   [π Δf_bin δt_max < tol], (general) delay-continuous evaluation at candidate
   delays. Common time shift via the BaseLinearFree idiom.
5. Degeneracies: kappa NEVER sampled (exact mass-sheet identity); beta NEVER
   sampled (circular point mass); sample reduced (gamma', y-in-shear-frame);
   overall amplitude -> apparent distance d_app (existing distance machinery);
   min-delay convention -> t_c. Astroid quadrant symmetries -> folding candidates.
6. Special functions: mpmath is ORACLE ONLY (tests). Production: double-precision
   complex-1F1 kernel (series + Kummer + k-ladder recurrences + large-|z| asymptotics),
   numba-compatible; stationary-phase regime needs no special functions.
7. Verification obligations (every build): tolerance tests vs exact/brute-force
   references (cogwheel value #1); timing assertions so contraction/K_a-eval
   regressions fail tests.

## Build sequence

| # | Build | Delivers | Status |
|---|-------|----------|--------|
| 1 | Lens engine | `cogwheel/lensing/chang_refsdal/` | **PARTIAL** — foundation only |
| 1b | Lens engine (corrective) | `_hyp1f1.py`, `operator.py`, `channels.py` + their tests | **REDO — brief needs rewrite** |
| 2 | Multi-component RB likelihood | `LensedWaveformGenerator` + RB likelihood (decisions 2–4) + brute-force lnL agreement | PENDING (brief after 1b) |
| 3 | Priors & validation | sampled lens coordinates + folding (decision 5) + injection-recovery | PENDING (brief after 2) |
| — | Optional research spike | lens-sector importance-sampling marginalization on the z(t) machinery | post-Build-3, judgement call |

### Build 1 — PARTIAL (2026-07-16)
ON DISK AND COMMITTED: `_dd.py` (37 tests), `_gauge.py` (34 tests) — both green;
`geometry.py` (872 lines) — committed but UNTESTED, treat as unreviewed.
MISSING: `_hyp1f1.py`, `operator.py`, `channels.py`, and all four new test modules.
SPEC.md's lensing row honestly says "IN PROGRESS — foundation only".
Plan deviations ACCEPTED at the gate (all argued, all improvements) — carry these:
  (1) double-double accumulation (`_dd.py`); cancellation law rel_err ~ eps*e^L,
      L = w(|y'|+gamma'/2); float64 tops out ~L<=15, headline config is L=29.6;
      dd certified to L<=48.
  (2) reconstruction + mass-sheet tolerances reframed as internal-consistency;
      accuracy gated by an mpmath oracle + a geometric-optics w^-3 slope test.
  (3) explicit branch gate {wave, geometric} — one event spans w ~50x in band.
  (4) ~450 lines of superseded prototype surface dropped.
  (5) no k-recurrence (unstable); shared-numerator ladder; no large-|z| branch;
      Kummer reparam makes the prefactor overflow-free in closed form.
BUILD-2 CONSEQUENCE TO CARRY: channels expose a `branch` flag; K-accuracy domain
is stated as L<=48, not a box.

### Build 1b — REDO. The brief must be rewritten before relaunch.
Four launches produced ZERO files. Root cause was MY BRIEF, not the SDK
(see Lessons). The next brief MUST obey the pipeline's role contract:
  - WPs deliver PRODUCTION CODE ONLY: `_hyp1f1.py`, `operator.py`, `channels.py`.
  - EVERY test goes in plan-level `domain_test_descriptions` for the Test
    Developer. No WP may name a test file as its deliverable.
  - No coder measurement campaigns. PRE-ANSWER every empirical fact (below).
MEASURED FACTS — pre-answer these, they are settled (verified against the
committed geometry.py; a coder independently reproduced and analytically derived
the census before being blocked):
  - Morse census: 4-image `(0,0,1,1)` — TWO minima + TWO saddles, NO maximum;
    2-image `(0,1)`. Holds at y=0 and general y inside the astroid, gamma
    0.05..0.4. `0,1,1,2` is WRONG (a point mass has -ln|x| -> +inf at the
    origin, so tau has no maximum; n_max = 0 in every regime). The invariant
    n_min - n_saddle + n_max = 0 holds for BOTH, so it cannot discriminate.
  - Fresh residual: all 168 CSV rows clear 1e-12 (max 1.93e-13; general
    1.93e-13, fold 1.69e-13, cusp 6.66e-16). No near-caustic exception needed.
    The solver's 3e-8 `residual_tolerance` default is acceptance-filter
    headroom, not achieved accuracy.
  - CSV fixture: 120 general + 24 fold + 24 cusp (rows 120-167). Any document
    saying "all 168 general" is wrong.
OPEN QUESTION worth answering in the brief (raised by a coder, and it is right):
an astroid winding-number check built from `geometry.critical_point` is a
consistency check, not an independent oracle — it lives inside the module under
test. Prefer an analytic astroid parametrization in the test.
KEEP: the non-circular crossing-fixture rule — fold/cusp scenario builders must
be constructed from geometry+operator+_gauge only, never from `channels.py`,
because they are the ground truth the label-continuity test judges channels
against. Circular tests pass.

## Driver protocol (for future context windows)
1. Launch: `.claude/sdk/launch_build.sh lensing_buildN .claude/handoff/lensing/buildN_brief.md`
   (file-gate approval; watchdog attaches; Monitor pattern printed in log header).
2. On plan-ready: read /tmp/lensing_buildN_approval/plan.json; REVIEW against this
   file's locked decisions; approve (touch plan_approved) or reject with concrete
   feedback (write plan_rejected). Respond promptly (watchdog clock runs).
   Reject NARROWLY: say what to KEEP as well as what to change. The Architect
   resumes in edit mode and amends rather than re-planning, so a narrow rejection
   costs ~4 min and preserves everything else.
3. On build end: review commit + report; run the build's tests myself; update this
   file's status; write next brief; launch next build.
4. If a build FAILS: diagnose from log; fix forward via a follow-up brief (or
   direct fix for infra-level issues only — physics/code goes through builds).
5. Between builds: /doc-sync if librarian backlog nears the hard block (>5).

### Brief-writing contract (learned the hard way — see Lessons)
- A WP's deliverable is PRODUCTION CODE. Never "write test X" — that is the Test
  Developer's job via `domain_test_descriptions`, and it is deliberate: code and
  the tests that bless it must not share an author.
  BUT DO NOT OVER-CORRECT: the prohibition is on a WP's DELIVERABLE, not on
  naming test files at all. The plan SHOULD recommend the suite layout — inside
  `domain_test_descriptions`, which is the ONLY channel the Test Developer sees
  (`_run_test_dev_agent` hands it the WP ids/titles plus these specs, and nothing
  else). Say which suites should exist and where, or it will guess: collapse four
  suites into one file, or invent names that collide with the AST import guard
  the committed test_lensing_gauge.py already enforces. Recommend the layout;
  just never make a test file a WP's deliverable.
- Never tell a coder to MEASURE and then decide. Pre-answer the fact, or make it
  a test. Ground truth in a discarded scratch file is unverifiable by construction.
- Coder verification is TARGETED (syntax/import + the one relevant test file).
  The Inspector runs the full suite afterwards.
- Plan field names are load-bearing: `has_domain_tests`, `domain_test_descriptions`.
  `stats_tests` / `has_stats_tests` / `stats_test_descriptions` are DEAD — the
  parser drops them silently and the Test Developer receives nothing.

- **STANDING RULES GO IN THE CREW PROMPT, ONCE. The brief carries only what is
  specific to THIS build.** The crew is a faithful servant: it does not need the
  same instruction repeated on every errand. If a rule is true for every build,
  it belongs in `.claude/crew/*.md`; if it is true for every WP of one build, it
  belongs in the brief's preamble, NOT stamped into each WP.
  Worked example, and the reason this rule exists: the 2026-07-16 brief mandated
  a ~90-word HEADLESS DISCIPLINE block be copied VERBATIM into every WP's `how`.
  It appeared 6x per plan, bloated every WP, and did not work. It was treating a
  symptom — coders ending with "let me know how to proceed" — whose actual cause
  was the role error (test-authoring WPs + "measure, don't guess"). gw has NO
  headless clause anywhere, in coder.md or its briefs, and never had the problem,
  because its coders are never put in a position to stop: the role contract keeps
  them writing code from pre-answered guidance.
  The block is RETIRED. Do not reintroduce it. Fixing the cause removed the need
  for the workaround — the correct prompt here is SHORTER, not longer. If a build
  ever does deadlock on etiquette again, fix it in `coder.md` once, or fix the
  brief that put the coder in that position; do not stamp a paragraph into
  every WP.

### Diagnostics that actually work
- A coder that reads/analyses then exits cheaply with no writes is NOT "the model
  choosing to stop". The build log records tool NAMES only — `tool_result`
  contents, including every denial, are invisible in it.
- READ THE SESSION TRANSCRIPT: `~/.claude/projects/<project-slug>/<session_id>.jsonl`.
  Map coder -> session by the `done` timestamp in the build log. The tool_result
  blocks carry the denials and the agent's final text/change-report verbatim.
  This is what finally cracked 2026-07-16 after hours of guessing.
- To reproduce one agent: `build_agent_options(...)` + `query()` in a small
  script. Use a NON-DEFAULT serena port and never start a second SSE on the same
  project while a build runs — it contends on the language server and wedges the
  interactive session's serena. Run detached to a logfile (a long agent blows past
  the shell tool's timeout).
- Health = log mtime advancing, not pgrep (the conda wrapper reads as alive; also
  `pgrep -f <pattern>` matches its own shell and reports phantom PIDs).

## Lessons and incident log (2026-07-16)

### THE root cause of the zero-write builds: my brief, not the SDK
I wrote briefs demanding test-authoring WPs ("a WP is complete ONLY when its
named tests exist") and "MEASURE, don't guess". That turned Coders into
experimentalists: they wrote /tmp probe scripts, hit a transient permission
denial, correctly refused to route around it, and ended BLOCKED with zero files.
Six WPs, four launches. gw's architect.md has forbidden test-authoring WPs for a
long time; I ported gw's SDK but not that rule — the most load-bearing wiring of
all. MEASURED, cogwheel's failing build vs gw's 24 build logs (149 coder calls):
    gw:       WRITE 26% (39 replace_content), SHELL 16%, /tmp probes: ZERO
    build1b:  WRITE  1% (1 call),             SHELL 60%, /tmp probes: 4
The profile is inverted. FIXED in .claude/crew/architect.md + the teja-force
template (cogwheel 0454d9e, skill 6f55220).

### FALSIFIED — do not act on these, they are recorded only so they are not re-derived
- "Builds cannot write files — a permission layer denies all writes." FALSE.
  The permission grants in settings.local.json were expanded 1 -> 55 rules and
  the next build failed identically, because SDK agents use
  `setting_sources=["user"]` and NEVER read project/local settings. Agent
  permissions come from AGENT_PERMISSION_MODES (Phase 2+ = bypassPermissions).
  settings.local.json only affects the human's interactive session.
- "The sandbox denying out-of-workspace /tmp writes is the root cause, and
  ignoreViolations fixes it." FALSE, and I stated it as verified off ONE
  observation. Tested properly: NOT positional (4/4 sequential /tmp writes
  succeed), NOT content (the byte-for-byte denied command replays clean 3/3),
  NOT hooks, NOT deny-rules. The denial is transient and external — "The user
  doesn't want to take this action right now" is the harness's wording for a
  refused permission REQUEST, and it struck different coders at different call
  indices. NOT REPRODUCIBLE ON DEMAND. The ignoreViolations change was REVERTED
  in all three repos (cogwheel 8aa96c2, skill 426f29f, gw c4c4e354) — it loosened
  the sandbox for no demonstrated benefit. Do not reinstate without measuring a
  denial RATE with and without it over many trials.
  THE LESSON: a non-deterministic failure cannot be confirmed fixed by a single
  passing run. If a fix cannot be shown to change a RATE, it has not been shown
  to do anything.

### Real infra fixes made (keep)
- SDK stream: DAG nodes and WP batches run SEQUENTIALLY; `_iter_query_with_timeout`
  queue-drains the stream in ONE dedicated task (anyio cancel-scope hazard in
  claude_agent_sdk 0.1.48); cancel-scope RuntimeErrors retry on a fresh stream,
  resuming the agent's own session. 18 SDK tests. PROPAGATE to the skill.
- `_run_hook_script` resolved its script path with TWO dirnames from
  .claude/sdk/agents.py, yielding `<repo>/.claude/.claude/hooks/...` — a path that
  never exists, so the hooks NEVER fired in cogwheel OR gw and hook_trace.log had
  never been written anywhere. Fixed (three dirnames) and VERIFIED: hook_trace.log
  writes, use-serena.sh returns correct decisions, hook_failures.log empty, coders
  still write via serena. cogwheel 8baa0e7, skill 8736c1c, gw 6a7cf371.
  gw's `_build_env` had the identical defect (silently loading no .env, so GW_*
  vars never reached subagents) — found by gw's own agent after I fixed one
  instance without grepping for siblings. Sibling-grep every path fix.
- Serena SSE orphans hold port 8322 and kill the next launch with
  "SSE server exited during startup (rc=3)". Diagnose with `lsof -tiTCP:8322`;
  kill the `--transport sse --port 8322` process, never the session's own
  `--project-from-cwd` stdio server.

### Professor gate: working as intended
Build 1's Professor review (first live firing) correctly returned CONCERN —
"~13 of 15 review specs describe code not yet written" — catching an
under-delivery the Inspector had passed. The commit was then blocked by the
SPEC-module pre-commit check. Both gates did their job.

### Credit where due
A coder refused to write the Morse-census assertion because it believed it was
false, and ended BLOCKED rather than encode it. It was right (`0,0,1,1`), and
that refusal is the only reason a wrong physics gate did not get baked into the
foundation and gate every downstream WP. Another coder caught the |C(w)|^2
prefactor tautology. The agents' judgement has been better than my briefs.

- CORRECTION TO THE CORRECTION (2026-07-16, later still). The entry above titled
  "THE SANDBOX WAS NEVER PROVEN TO BE THE CAUSE" is itself wrong, and the way it
  is wrong matters more than the conclusion.
  I "falsified" the sandbox fix with a bisect (4/4 /tmp writes OK) and a replay
  (3/3 OK) and concluded it did nothing. Timeline says otherwise:
      08:46:01  ignoreViolations APPLIED   (d5e55a4)
      ~09:5x    bisect 4/4 OK              <- ran WITH the fix applied
      ~10:0x    replay 3/3 OK              <- ran WITH the fix applied
      10:19:15  REVERTED on my advice      (8aa96c2)
      11:38     build: DENIED on the FIRST /tmp write
  I tested whether the treatment was necessary BY RUNNING WITH THE TREATMENT
  APPLIED. A control error, not a subtle one.
  Worse, the conditions were doubly confounded: every passing probe was BOTH
  with-fix AND attached (foreground, via the session's shell); every denial came
  from a DETACHED process (a real build, or probe_wp1.py launched with
  run_in_background). There is no attached-without-fix datum anywhere, so "fix
  vs no-fix" and "attached vs detached" cannot be separated from what I had.
  What the two clean observations actually say (n=1 each, so weak):
      with fix    (08:55 build): 1st /tmp write OK, 2nd DENIED
      without fix (11:38 build): 1st /tmp write DENIED
  Caveat against over-reading it the other way: a path glob that admits
  /tmp/probe.py and refuses /tmp/probe2.py in the SAME directory makes no
  mechanical sense, so this may still be noise.
  METHOD FOR ANYONE PICKING THIS UP: hold the context at DETACHED (that is what
  a build is), vary ONLY the allowlist, N trials per arm, compare DENIAL RATES.
  scratchpad/denial_rate.py does exactly that. Never judge this by a single
  passing run — that is the mistake that cost the whole afternoon, twice.

- "CODERS SHOULD NOT PROBE" WAS ALSO WRONG (2026-07-16). The brief was rewritten
  to remove every MEASURE order, and coder-2 wrote a /tmp probe anyway — for an
  excellent reason, and it was RIGHT to. It was checking a claim in my brief
  before building on it, and the claim was FALSE: I had asserted the
  (w/2)ln(w/2) phase "CANCELS against Im loggamma(1-iw/2)" and must be cancelled
  analytically. That cancellation is ASYMPTOTIC (Stirling), not an algebraic
  identity; "cancel it analytically" would mean a Bernoulli tail plus a small-x
  convergence switch, i.e. real correctness risk for nothing. The coder proposed
  the polar route instead — |C|*exp(i*theta) with |C|^2 = -pi*w/expm1(-pi*w) —
  which never needs loggamma's real part and so removes both overflow traps.
  VERIFIED against a 60-dps mpmath oracle before adopting: |C|^2 closed form is
  exact to 2.2e-16 flat over w in [1e-3,700]; the polar route lands at 3.4e-13
  at w=700, ~300x inside the 1e-10 gate. The coder's own "~0.75 digits lost"
  estimate was optimistic (it is ~3 digits), so verify even a correct-sounding
  agent claim — but its DIRECTION was right and my brief was the thing that was
  wrong. Brief corrected.
  So the rule is NOT "coders must not probe". Probing is correct engineering
  when the math is subtle. The rule is: pre-answer what you can, and make sure
  the probe PATH WORKS.
