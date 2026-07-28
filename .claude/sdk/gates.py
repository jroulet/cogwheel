"""Code-enforced gates that replace prompt-based gating instructions.

These are the hard requirements that were previously "vibes" in agent
prompts.  Now they're if/else in Python.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from .schemas import (
    BuildMode,
    EscalationLevel,
    Finding,
    InspectorResult,
    InspectorVerdict,
    Plan,
)


class GateFailure(Exception):
    """Raised when a code-enforced gate is not satisfied."""


class EscalationNeeded(Exception):
    """Raised when Inspector findings require user/design intervention."""

    def __init__(self, findings: list[Finding], loop_count: int):
        self.findings = findings
        self.loop_count = loop_count
        design_issues = [f for f in findings if f.severity == EscalationLevel.DESIGN]
        msg = (
            f"Inspector found {len(design_issues)} design-level issue(s) "
            f"after {loop_count} revision loop(s).  User intervention required."
        )
        super().__init__(msg)


# ── Plan verification gate ───────────────────────────────────────────────────


_TEST_SUITE_RE = re.compile(r"\b(test_\w+\.py)\b")


def _suite_write_conflicts(specs: list[str]) -> list[tuple[str, str]]:
    """Domain-test specs that route the same test file to two shards.

    The Test Developer phase shards ``domain_test_descriptions`` by the FIRST
    ``test_*.py`` each spec names (its owning suite — mirror of
    ``orchestrator._group_test_specs``). Write-ownership must be disjoint: a
    conflict is a spec whose owning shard ALSO names another spec's owning
    file, so two Test-Developer runs would both author that file. Returns
    ``(owner, shared)`` pairs; empty when ownership is disjoint. Defensive:
    any parse error yields ``[]`` (the check is skipped, never blocks a build).
    """
    try:
        primary: dict[int, str] = {}
        refs: dict[int, set[str]] = {}
        for i, spec in enumerate(specs or []):
            files = _TEST_SUITE_RE.findall(str(spec))
            if files:
                primary[i] = files[0]
                refs[i] = set(files)
        owners = set(primary.values())
        conflicts: list[tuple[str, str]] = []
        for i, owned in primary.items():
            for f in refs[i]:
                if f != owned and f in owners:
                    conflicts.append((owned, f))
        return conflicts
    except Exception:
        return []


def verify_plan(
        plan: Plan, require_professor: bool = False,
) -> tuple[list[str], list[str]]:
    """Check the plan against the verification checklist.

    Parameters
    ----------
    require_professor:
        Pass True when triage classified the task as complex and the
        Professor consultation was offered — the plan must then cite
        Professor inputs. Leave False for standard tasks (the Professor
        was skipped by design; requiring citations would reject every
        standard-task plan for not citing a consultant never spawned).

    Returns
    -------
    (failures, missing_turns)
        failures: list of hard failures (non-empty = plan is invalid).
        missing_turns: WP ids that need max_turns filled by Architect.
    """
    failures: list[str] = []
    missing_turns: list[str] = []

    if require_professor and not plan.professor_inputs:
        failures.append("Plan does not cite any Professor inputs.")

    if not plan.simplifier_inputs:
        failures.append("Plan does not cite any Simplifier inputs.")

    if not plan.work_packages:
        failures.append("Plan has no work packages.")

    # Test-suite write-ownership must be disjoint: no test file may be routed
    # to two Test-Developer shards (they would author it in parallel).
    for owner, shared in _suite_write_conflicts(plan.domain_test_descriptions):
        failures.append(
            f"Test-suite write-ownership conflict: the shard owning {owner} "
            f"also targets {shared}, which is another shard's suite. Each "
            f"test file must be authored by exactly one Test Developer — split "
            f"the spec so suites are disjoint.")

    for wp in plan.work_packages:
        if wp.who not in ("Coder", "Foreman-Lite"):
            failures.append(f"{wp.id}: 'who' must be 'Coder' or 'Foreman-Lite', got '{wp.who}'.")
        if not wp.where:
            failures.append(f"{wp.id}: missing 'where' (files/symbols affected).")
        # Flag missing max_turns — the orchestrator will resume the
        # Architect to fill them in rather than using a crude formula.
        if wp.max_turns is None:
            missing_turns.append(wp.id)

    return failures, missing_turns


# ── Fast-path eligibility gate ───────────────────────────────────────────────


def is_fast_path_eligible(plan: Plan) -> bool:
    """Determine if the plan qualifies for Foreman-Lite fast-path execution.

    Criteria (ALL must be true):
    - Affects <= 2 files
    - No new public API
    - No domain-specific tests needed
    - No domain-sensitive changes (a physics/numerics change must not skip
      the Coder + Inspector loop, even when it writes no new tests)
    """
    return (
        len(plan.files_affected) <= 2
        and not plan.has_new_public_api
        and not plan.has_domain_tests
        and not plan.has_domain_changes
    )


# ── Inspector gate ───────────────────────────────────────────────────────────


MAX_REVISION_LOOPS = 2
MAX_CLOSURE_RECHECKS = 2


def check_inspector_gate(result: InspectorResult) -> bool:
    """Return True if Inspector verdict is PASS."""
    return result.verdict == InspectorVerdict.PASS


def classify_findings(findings: list[Finding]) -> dict[EscalationLevel, list[Finding]]:
    """Group findings by escalation level for triage."""
    classified: dict[EscalationLevel, list[Finding]] = {
        EscalationLevel.TRIVIAL: [],
        EscalationLevel.IMPLEMENTATION: [],
        EscalationLevel.DESIGN: [],
    }
    for f in findings:
        classified[f.severity].append(f)
    return classified


def should_escalate(findings: list[Finding], loop_count: int) -> bool:
    """Determine if findings require escalation to the user.

    Only escalate when revision loops are exhausted AND actionable findings
    remain.  DESIGN findings are no longer instant-death — they go through
    the Architect → User chain of command first.

    TRIVIAL findings deliberately do NOT escalate: by their own definition
    ("style, missing tests, minor inefficiency — fix at convenience") they
    must never block a build or consume a human decision.  They must still
    TERMINATE the loop though — see `revision_budget_spent`, which is what
    the caller uses to decide whether to keep looping at all.
    """
    has_actionable = any(
        f.severity in (EscalationLevel.IMPLEMENTATION, EscalationLevel.DESIGN)
        for f in findings
    )
    return has_actionable and loop_count > MAX_REVISION_LOOPS


def revision_budget_spent(findings: list[Finding], loop_count: int) -> bool:
    """True when the revision loop must STOP, at ANY severity.

    `should_escalate` answers "does a human need to decide?"; this answers
    "may we go round again?".  They are different questions, and conflating
    them is what let a build run to `revision 8/2`: with only TRIVIAL
    findings outstanding `should_escalate` was permanently False, so the
    loop neither escalated nor exited and re-derived the same two findings
    eight times (2026-07-28, Born carrier build, ~26 min and ~$24 of
    Inspector + foreman_lite cycles with ZERO implementation findings).
    """
    return bool(findings) and loop_count > MAX_REVISION_LOOPS


def finding_signature(findings: list[Finding]) -> frozenset[str]:
    """Identity of a finding SET, for non-convergence detection.

    Two consecutive revisions producing the same signature means the loop is
    re-deriving findings the fixer cannot or will not clear — spending the
    remaining budget on it buys nothing.  Keyed on ``finding_id + severity +
    file`` ONLY: the Inspector already assigns a stable per-build id, so the
    description adds no identity, and including it DEFEATED the check —
    descriptions quote line numbers, and a partial fix that shifts the file
    re-words the same unfixed finding into a "new" signature, so the loop
    never detects that it is stuck.
    """
    return frozenset(
        f'{f.finding_id}|{f.severity}|{f.file}' for f in findings
    )


def has_design_findings(findings: list[Finding]) -> bool:
    """Check if any findings are DESIGN-level (triggers Architect triage)."""
    return any(f.severity == EscalationLevel.DESIGN for f in findings)


# ── Branch safety gate ───────────────────────────────────────────────────────

ALLOWED_BRANCHES: list[str] | None = ["claude-dev"]  # None = all branches allowed


def check_branch_safety(project_root: str) -> str:
    """Verify we're on an allowed branch.  Returns branch name or raises."""
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True, cwd=project_root,
    )
    branch = result.stdout.strip()
    if ALLOWED_BRANCHES is not None and branch not in ALLOWED_BRANCHES:
        raise GateFailure(
            f"Current branch '{branch}' is not in allowed list {ALLOWED_BRANCHES}. "
            f"Refusing to proceed."
        )
    return branch


# ── Commit gate ──────────────────────────────────────────────────────────────


def check_commit_allowed(
    inspector_result: Optional[InspectorResult],
    build_mode: BuildMode,
) -> bool:
    """Verify that committing is allowed.

    Rules:
    - Fast-path: no Inspector required, commit is allowed.
    - Full pipeline: Inspector must have returned PASS.
    """
    if build_mode == BuildMode.FAST_PATH:
        return True

    if inspector_result is None:
        raise GateFailure("Cannot commit: Inspector has not run yet.")

    if inspector_result.verdict != InspectorVerdict.PASS:
        raise GateFailure(
            f"Cannot commit: Inspector verdict is {inspector_result.verdict.value}. "
            f"Findings: {len(inspector_result.findings)}"
        )

    return True


# ── User approval gate ──────────────────────────────────────────────────────

_WP_HEADER_RE = re.compile(r"^#{2,3} WP\d+", re.MULTILINE)

# Bare-denial rate grows with transcript depth (0/106 in the first two tool
# calls, median at call 14; claude-code #74351, no upstream fix). Wide plans
# fan the whole context out to every coder, so depth is gated here.
MAX_WPS_BEFORE_WARNING = 3


def plan_depth_banner(plan_summary: str) -> str:
    """One-line depth report for the approval banner.

    Flags plans whose WP fan-out exceeds ``MAX_WPS_BEFORE_WARNING`` so the
    reviewer must consciously accept the transcript-depth risk (see
    CLAUDE.md, "SDK Build Briefs").
    """
    n_wps = len(_WP_HEADER_RE.findall(plan_summary))
    size_kb = len(plan_summary.encode()) / 1024
    line = f"Depth: {n_wps} WP(s), plan {size_kb:.1f} KB."
    if n_wps > MAX_WPS_BEFORE_WARNING:
        line += (
            f" WARNING: >{MAX_WPS_BEFORE_WARNING} WPs — transcript-depth"
            " risk (bare-denial rate grows with depth). Prefer splitting"
            " into sequential builds; approve only if the width is truly"
            " irreducible."
        )
    return line


def prompt_user_approval(
    plan_summary: str,
    auto_approve: bool = False,
    approval_dir: str | None = None,
) -> tuple[bool, str]:
    """Interactive prompt for plan approval.  Blocks until user responds.

    Three modes:
    - **stdin** (default): Interactive terminal prompt via ``input()``.
    - **auto** (``--yes``): Approve immediately, no prompt.
    - **file-based** (``--approval-dir``): Write plan to a directory and poll
      for a signal file.  Used by the in-session ``/build`` command so the
      session agent can review the plan asynchronously.
    """
    print("\n" + "=" * 70)
    print("PLAN FOR APPROVAL")
    print("=" * 70)
    print(plan_summary)
    print("=" * 70)
    print(plan_depth_banner(plan_summary))

    if auto_approve:
        print("\n[--yes] Plan auto-approved.")
        return True, ""

    if approval_dir:
        return _file_based_approval(plan_summary, Path(approval_dir))

    while True:
        response = input("\nApprove this plan? [y/n/q] ").strip().lower()
        if response in ("y", "yes"):
            return True, ""
        if response in ("n", "no"):
            feedback = input("Feedback (Enter to skip): ").strip()
            return False, feedback
        if response in ("q", "quit"):
            raise KeyboardInterrupt("User quit during plan approval.")
        print("Please enter 'y' to approve, 'n' to reject, or 'q' to quit.")


def _file_based_approval(plan_summary: str, dir_path: Path) -> tuple[bool, str]:
    """Write plan and poll for approval/rejection signal files."""
    dir_path.mkdir(parents=True, exist_ok=True)
    plan_file = dir_path / "plan.json"
    ready_file = dir_path / "plan_ready"
    approved_file = dir_path / "plan_approved"
    rejected_file = dir_path / "plan_rejected"

    plan_file.write_text(plan_summary)
    ready_file.touch()
    print(f"\n[file-based] Plan written to {plan_file}")
    print(f"[file-based] Waiting for approval signal in {dir_path} ...")

    while True:
        if approved_file.exists():
            approved_file.unlink()
            ready_file.unlink(missing_ok=True)
            print("[file-based] Plan approved.")
            return True, ""
        if rejected_file.exists():
            feedback = rejected_file.read_text().strip()
            rejected_file.unlink()
            ready_file.unlink(missing_ok=True)
            print(f"[file-based] Plan rejected. Feedback: {feedback or '(none)'}")
            return False, feedback
        time.sleep(5)


# ── Escalation decision gate ────────────────────────────────────────────────


def prompt_escalation_decision(
    findings: list[Finding],
    architect_rationale: str = "",
    approval_dir: str | None = None,
) -> tuple[str, str]:
    """Ask the user to disposition escalated Inspector findings.

    Returns (decision, feedback) where decision is one of:
    - "accept": proceed with commit despite findings
    - "fix": user provides feedback for another Coder revision
    - "abort": kill the build

    When ``approval_dir`` is set (a detached / file-gated build, same as the
    plan gate), this is FILE-BASED: findings are written to a file and the gate
    polls for a decision file. The interactive ``input()`` path only runs when
    there is a real terminal — calling it in a detached build raises EOFError
    (there is no stdin) and kills the whole run, which is exactly what happened
    on 2026-07-16 when the Inspector escalated the missing test suites.
    """
    if approval_dir:
        return _file_based_escalation(findings, architect_rationale,
                                      Path(approval_dir))
    print("\n" + "=" * 70)
    print("ESCALATION — Inspector findings require your decision")
    print("=" * 70)

    for f in findings:
        print(f"\n  [{f.severity.value}] {f.file}")
        print(f"    {f.description}")
        if f.suggested_fix:
            print(f"    Suggested: {f.suggested_fix}")

    if architect_rationale:
        print(f"\n  Architect assessment: {architect_rationale}")

    print("=" * 70)

    while True:
        response = input(
            "\n[a]ccept (commit anyway) / [f]ix (give instructions) / [q]uit? "
        ).strip().lower()
        if response in ("a", "accept"):
            return "accept", ""
        if response in ("f", "fix"):
            feedback = input("Instructions for Coder: ").strip()
            return "fix", feedback
        if response in ("q", "quit", "abort"):
            return "abort", ""
        print("Please enter 'a' to accept, 'f' to fix, or 'q' to quit.")


def _file_based_escalation(
    findings: list[Finding],
    architect_rationale: str,
    dir_path: Path,
) -> tuple[str, str]:
    """Write escalated findings and poll for a decision file.

    Mirrors ``_file_based_approval`` (the plan gate). The driver dispositions by
    creating one of:
      - ``escalation_accept``  -> ("accept", "")  commit despite the findings
      - ``escalation_fix``     -> ("fix", <file contents>)  another revision
      - ``escalation_abort``   -> ("abort", "")   stop the build
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    findings_file = dir_path / "escalation.json"
    ready_file = dir_path / "escalation_ready"
    accept_file = dir_path / "escalation_accept"
    fix_file = dir_path / "escalation_fix"
    abort_file = dir_path / "escalation_abort"

    payload = {
        "architect_rationale": architect_rationale,
        "findings": [
            {
                "finding_id": f.finding_id,
                "severity": f.severity.value,
                "file": f.file,
                "description": f.description,
                "suggested_fix": f.suggested_fix,
            }
            for f in findings
        ],
    }
    findings_file.write_text(json.dumps(payload, indent=2))
    ready_file.touch()
    print(f"\n[file-based] ESCALATION written to {findings_file}")
    print(f"[file-based] Waiting for a decision file in {dir_path} "
          f"(escalation_accept / escalation_fix / escalation_abort) ...")

    while True:
        if accept_file.exists():
            accept_file.unlink()
            ready_file.unlink(missing_ok=True)
            print("[file-based] Escalation accepted — proceeding.")
            return "accept", ""
        if fix_file.exists():
            feedback = fix_file.read_text().strip()
            fix_file.unlink()
            ready_file.unlink(missing_ok=True)
            print(f"[file-based] Escalation -> fix. Instructions: "
                  f"{feedback or '(none)'}")
            return "fix", feedback
        if abort_file.exists():
            abort_file.unlink()
            ready_file.unlink(missing_ok=True)
            print("[file-based] Escalation aborted.")
            return "abort", ""
        time.sleep(5)


# ── Outside Inspector merge ─────────────────────────────────────────────────


_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "in", "on", "at", "to", "of",
    "and", "or", "not", "it", "for", "be", "was", "are",
})


def _description_fingerprint(description: str) -> frozenset[str]:
    """Word-level fingerprint for near-duplicate finding detection."""
    words = description.lower().split()
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    return frozenset(content_words[:15])


def merge_inspector_results(
    primary: InspectorResult,
    outside: Optional[InspectorResult],
) -> InspectorResult:
    """Merge Claude (primary) and outside (e.g. Codex) inspector results.

    Rules:
    - If outside is None, return primary unchanged.
    - Verdict: ISSUES if either says ISSUES. PASS only if both PASS.
    - Findings: union, deduplicated by (file, description similarity).
    - Primary findings take precedence for near-duplicates.
    """
    if outside is None:
        return primary

    if primary.verdict == InspectorVerdict.ISSUES or outside.verdict == InspectorVerdict.ISSUES:
        merged_verdict = InspectorVerdict.ISSUES
    else:
        merged_verdict = InspectorVerdict.PASS

    merged_findings = list(primary.findings)
    primary_fingerprints = {
        (f.file, _description_fingerprint(f.description))
        for f in primary.findings
    }

    for outside_finding in outside.findings:
        outside_key = (outside_finding.file, _description_fingerprint(outside_finding.description))
        is_duplicate = False
        for pf, pfp in primary_fingerprints:
            if pf == outside_key[0] and len(pfp & outside_key[1]) > 0.6 * max(len(pfp), len(outside_key[1]), 1):
                is_duplicate = True
                break
        if not is_duplicate:
            merged_findings.append(outside_finding)

    summary_parts = []
    if primary.summary:
        summary_parts.append(f"Claude: {primary.summary[:200]}")
    if outside.summary:
        summary_parts.append(f"Outside: {outside.summary[:200]}")

    return InspectorResult(
        verdict=merged_verdict,
        findings=merged_findings,
        resolved_ids=primary.resolved_ids,
        summary=" | ".join(summary_parts),
        import_check_passed=primary.import_check_passed,
        smoke_test_passed=primary.smoke_test_passed,
    )
