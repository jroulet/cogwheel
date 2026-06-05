"""Memory management via Serena MCP.

Reads/writes Serena memories for agent prompt assembly and Dreamer
consolidation.  Falls back to direct file I/O if Serena is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Memory assignments per agent.
AGENT_MEMORIES: dict[str, dict] = {
    "architect": {
        "reads": ["architect_short_term", "architect_knowledge"],
        "cross_reads": ["coder_knowledge", "inspector_knowledge"],
        "writes": "architect_short_term",
    },
    "simplifier": {
        "reads": [],
        "cross_reads": [],
        "writes": None,  # stateless
    },
    "coder": {
        "reads": ["coder_short_term", "coder_knowledge"],
        "cross_reads": ["inspector_knowledge"],
        "writes": "coder_short_term",
    },
    "foreman_lite": {
        "reads": ["foreman_short_term", "foreman_knowledge"],
        "cross_reads": ["inspector_knowledge", "coder_knowledge"],
        "writes": "foreman_short_term",
    },
    "tidier": {
        "reads": ["tidy_short_term", "tidy_knowledge"],
        "cross_reads": [],
        "writes": "tidy_short_term",
    },
    "test_dev": {
        "reads": ["test_dev_short_term", "test_dev_knowledge"],
        "cross_reads": ["coder_knowledge"],
        "writes": "test_dev_short_term",
    },
    "inspector": {
        "reads": ["inspector_short_term", "inspector_knowledge"],
        "cross_reads": ["coder_knowledge"],
        "writes": "inspector_short_term",
    },
    "librarian": {
        "reads": ["librarian_short_term", "librarian_knowledge"],
        "cross_reads": [],
        "writes": "librarian_short_term",
    },
    "dreamer": {
        "reads": [
            "architect_short_term", "architect_knowledge",
            "foreman_short_term", "foreman_knowledge",
            "coder_short_term", "coder_knowledge",
            "inspector_short_term", "inspector_knowledge",
            "tidy_short_term", "tidy_knowledge",
            "test_dev_short_term", "test_dev_knowledge",
            "librarian_short_term", "librarian_knowledge",
        ],
        "cross_reads": [],
        "writes": None,  # writes to many memories programmatically
    },
}

# Dreamer consolidation map
CONSOLIDATION_MAP: dict[str, dict[str, str]] = {
    "architect":    {"short_term": "architect_short_term",  "long_term": "architect_knowledge"},
    "foreman_lite": {"short_term": "foreman_short_term",    "long_term": "foreman_knowledge"},
    "coder":        {"short_term": "coder_short_term",      "long_term": "coder_knowledge"},
    "inspector":    {"short_term": "inspector_short_term",  "long_term": "inspector_knowledge"},
    "tidier":       {"short_term": "tidy_short_term",       "long_term": "tidy_knowledge"},
    "test_dev":     {"short_term": "test_dev_short_term",   "long_term": "test_dev_knowledge"},
    "librarian":    {"short_term": "librarian_short_term",  "long_term": "librarian_knowledge"},
}

# Serena memories directory (fallback path for direct file I/O)
SERENA_MEMORIES_DIR = Path(".serena/memories")


def get_memory_names_for_agent(agent_name: str) -> list[str]:
    """Return all memory names an agent should read at startup."""
    config = AGENT_MEMORIES.get(agent_name, {})
    return config.get("reads", []) + config.get("cross_reads", [])


def get_write_memory_for_agent(agent_name: str) -> Optional[str]:
    """Return the short-term memory name an agent writes to, or None."""
    config = AGENT_MEMORIES.get(agent_name, {})
    return config.get("writes")


async def load_memories_text(
    memory_names: list[str],
    project_root: str,
    serena_client=None,
) -> str:
    """Load and concatenate memory contents for inclusion in a system prompt."""
    parts: list[str] = []

    for name in memory_names:
        content = None

        if serena_client is not None:
            try:
                result = await serena_client.call_tool("read_memory", {"name": name})
                content = result.get("content", "") if isinstance(result, dict) else str(result)
            except Exception:
                content = None

        if content is None:
            mem_path = Path(project_root) / SERENA_MEMORIES_DIR / f"{name}.md"
            if mem_path.exists():
                content = mem_path.read_text(encoding="utf-8")
            else:
                content = f"(memory '{name}' not found)"

        parts.append(f"### Memory: {name}\n{content}")

    return "\n\n".join(parts)


def load_memories_text_sync(
    memory_names: list[str],
    project_root: str,
) -> str:
    """Synchronous version — reads directly from `.serena/memories/`."""
    parts: list[str] = []

    for name in memory_names:
        mem_path = Path(project_root) / SERENA_MEMORIES_DIR / f"{name}.md"
        if mem_path.exists():
            content = mem_path.read_text(encoding="utf-8")
        else:
            content = f"(memory '{name}' not found)"
        parts.append(f"### Memory: {name}\n{content}")

    return "\n\n".join(parts)


def write_memory_sync(
    name: str,
    content: str,
    project_root: str,
) -> Path:
    """Write a memory file directly (file-based, no Serena needed).

    Used by the orchestrator when use_serena=False, and as a fallback
    when Serena MCP is unavailable. Writes to the same `.serena/memories/`
    directory that load_memories_text reads from.
    """
    mem_dir = Path(project_root) / SERENA_MEMORIES_DIR
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_path = mem_dir / f"{name}.md"
    mem_path.write_text(content, encoding="utf-8")
    return mem_path
