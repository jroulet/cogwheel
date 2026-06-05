"""Structured types for the SDK pipeline orchestrator.

Every agent produces typed output that the orchestrator can
inspect programmatically — no parsing required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Optional


# ── Phase lifecycle ──────────────────────────────────────────────────────────


class Phase(Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    DREAMING = "dreaming"
    DONE = "done"


class InspectorVerdict(Enum):
    PASS = "PASS"
    ISSUES = "ISSUES"


class BuildMode(Enum):
    FULL = "full_pipeline"
    FAST_PATH = "fast_path"


class EscalationLevel(Enum):
    TRIVIAL = "trivial"        # Foreman-Lite fixes directly
    IMPLEMENTATION = "implementation"  # Coder fixes
    DESIGN = "design"          # Architect triages


class TriageVerdict(Enum):
    CODER_FIX = "coder_fix"       # Architect: Coder can handle this
    OVERRIDE = "override"          # Architect: dismiss finding
    ESCALATE = "escalate"          # Architect → User


class SimplifierVerdict(Enum):
    LEAN = "lean"
    WATCH = "watch"
    TRIM = "trim"


# ── Phase 1: Planning ────────────────────────────────────────────────────────


@dataclass
class WorkPackage:
    """Single unit of work in the approved plan."""
    id: str            # "WP-1"
    title: str
    what: str          # precise description
    where: list[str]   # files and symbols affected
    how: str           # implementation approach (includes directed domain guidance)
    who: str           # "Coder" or "Foreman-Lite"
    dependencies: list[str] = field(default_factory=list)  # WP ids
    verification: str = ""
    simplifier_verdict: Optional[SimplifierVerdict] = None
    simplifier_note: Optional[str] = None
    # Optional Coder turn budget override.  None → use the default 75.  Set
    # explicitly when the Architect estimates a WP needs more turns than the
    # default to complete (e.g. multi-function audits, large refactors).
    # See .claude/crew/architect.md for the budgeting rules.
    max_turns: Optional[int] = None


@dataclass
class Plan:
    """Output of Phase 1 — the Architect's approved plan.

    Domain-specific info flows through two paths:
    - Constraints/guidance → embedded in each WP's `how` field via
      the Architect's directed knowledge transfer (Coder reads these)
    - Test specifications → `domain_test_descriptions` (Test Dev reads these)
    """
    summary: str
    work_packages: list[WorkPackage]
    has_domain_tests: bool    # TestDev writes new domain-specific tests
    has_new_public_api: bool
    has_spec_update: bool     # requires SPEC.md + CHANGELOG.md update
    files_affected: list[str]
    # Domain test specs from the Architect (natural-language descriptions
    # of setup, operation, and expected results for domain-specific tests)
    domain_test_descriptions: list[str] = field(default_factory=list)
    # Traceability: which Simplifier inputs shaped the plan
    simplifier_inputs: list[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Serialize plan to JSON file for crash recovery / session agent context."""
        import json
        from dataclasses import asdict
        path.write_text(json.dumps(asdict(self), indent=2))


# ── Phase 2: Execution ───────────────────────────────────────────────────────


@dataclass
class CoderResult:
    """What the Coder reports after implementing work packages."""
    work_package_ids: list[str]
    files_changed: list[str]
    summary: str
    surprises: list[str] = field(default_factory=list)
    syntax_check_passed: bool = True
    import_check_passed: bool = True
    smoke_test_passed: bool = True


@dataclass
class TidierResult:
    """What the Tidier reports after structural cleanup."""
    files_touched: list[str]
    spacing_fixes: int = 0
    summary: str = ""


@dataclass
class TestDevResult:
    """What the Test Developer reports after writing tests."""
    test_files: list[str]
    tests_written: int
    tests_passed: int
    tests_failed: int
    domain_tests: list[str] = field(default_factory=list)  # test names
    diagnostic_plots: list[str] = field(default_factory=list)  # file paths
    summary: str = ""


@dataclass
class Finding:
    """A single issue found by the Inspector."""
    severity: EscalationLevel
    file: str
    description: str
    suggested_fix: Optional[str] = None
    is_preexisting: bool = False
    finding_id: str = ""  # stable identity across revision rounds (INS-{round}-{n:03d})
    source: str = "claude"  # "claude" or "codex" (for outside inspector)


@dataclass
class InspectorResult:
    """What the Inspector reports after reviewing."""
    verdict: InspectorVerdict
    findings: list[Finding] = field(default_factory=list)
    resolved_ids: list[str] = field(default_factory=list)
    import_check_passed: bool = True
    smoke_test_passed: bool = True
    summary: str = ""


@dataclass
class TriageEntry:
    """A single finding's triage verdict from Architect."""
    finding_id: str
    verdict: TriageVerdict
    rationale: str = ""
    coder_instructions: str = ""   # Only for CODER_FIX


@dataclass
class TriageResult:
    """Collected triage verdicts from Architect."""
    entries: list[TriageEntry] = field(default_factory=list)


@dataclass
class LibrarianResult:
    """What the Librarian reports after doc sync."""
    files_updated: list[str]
    spec_version_bumped: bool = False
    changelog_updated: bool = False
    rebuild_ran: bool = False
    summary: str = ""


@dataclass
class BuildReport:
    """Compiled by the orchestrator at the end of Phase 2."""
    mode: BuildMode
    work_packages_completed: int
    work_packages_total: int
    coder_result: Optional[CoderResult] = None
    tidier_result: Optional[TidierResult] = None
    test_dev_result: Optional[TestDevResult] = None
    inspector_result: Optional[InspectorResult] = None
    librarian_result: Optional[LibrarianResult] = None
    revision_loops: int = 0
    escalations: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)  # "sha message"
    total_cost: float = 0.0


# ── Phase 3: Dreaming ────────────────────────────────────────────────────────


class MemoryAction(Enum):
    PROMOTE = "promote"    # new pattern → long-term
    CORRECT = "correct"    # contradicts existing → update
    CONFIRM = "confirm"    # already known → discard
    DISCARD = "discard"    # session-specific → drop


@dataclass
class MemoryEntry:
    """A single entry the Dreamer classifies."""
    agent: str
    content: str
    action: MemoryAction
    target_memory: Optional[str] = None  # which long-term memory to update
    reason: str = ""


@dataclass
class DreamerResult:
    """What the Dreamer reports after memory consolidation."""
    entries_processed: int
    promoted: int
    corrected: int
    confirmed: int
    discarded: int
    participation_gaps: list[str] = field(default_factory=list)
    summary: str = ""


# ── DAG utilities ────────────────────────────────────────────────────────────


@dataclass
class DAGNode:
    """A node in the Phase 2 execution DAG.

    The ``run`` callable is a zero-arg async function that returns a result
    string.  Side-effects (updating ``report``, ``self._inspector_result``,
    etc.) are handled by the callable's closure.
    """
    name: str
    kind: str  # "agent" or "skill"
    depends_on: list[str]
    skip_when: Callable[["Plan", list[str]], bool]  # (plan, changed_files) → skip?
    run: Callable[[], "Awaitable[str]"]  # async () → result_text


def build_dependency_graph(work_packages: list[WorkPackage]) -> list[list[WorkPackage]]:
    """Topological sort of WPs into parallelizable batches.

    Returns a list of batches, where each batch contains WPs whose
    dependencies are all satisfied by previous batches.
    """
    id_to_wp = {wp.id: wp for wp in work_packages}
    completed: set[str] = set()
    batches: list[list[WorkPackage]] = []
    remaining = set(id_to_wp.keys())

    while remaining:
        ready = [
            wp_id for wp_id in remaining
            if all(dep in completed for dep in id_to_wp[wp_id].dependencies)
        ]
        if not ready:
            unresolved = {
                wp_id: [d for d in id_to_wp[wp_id].dependencies if d not in completed]
                for wp_id in remaining
            }
            raise ValueError(f"Circular or unresolvable dependencies: {unresolved}")
        batches.append([id_to_wp[wp_id] for wp_id in ready])
        completed.update(ready)
        remaining -= set(ready)

    return batches
