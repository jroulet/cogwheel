#!/usr/bin/env bash
# bootstrap_claude_workflow.sh — set up Claude Code for a collaborator.
#
# Two tiers:
#
#   Default (non-agentic):
#     - Copy CLAUDE.md (stripping Serena-specific sections)
#     - Create CLAUDE.local.md.example
#     - Print quickstart
#     For collaborators who use Claude Code with CLAUDE.md only (no
#     Serena, no SDK pipeline).
#
#   --with-agentic (agentic tier):
#     - Everything above, PLUS:
#     - Install the `git mind-meld` alias (local scope) so the
#       collaborator can opt in to LLM-driven semantic merge of agent
#       memories across branches.
#     - Wipe .claude/agent_state/ so the collaborator's first /build
#       creates fresh state instead of inheriting the installer's.
#     - Print the mind-meld usage hint.
#     Optional --mind-meld sub-flag DISABLES the agent_state wipe, so
#     the collaborator inherits the installer's agent memories as a
#     starting point (use only for "graft from a known-good state").
#
# Usage modes:
#
#   In-place (no worktree creation):
#     bash scripts/bootstrap_claude_workflow.sh [--with-agentic] [--mind-meld]
#     Must be run from inside a checkout that already has the scaffolding
#     (.claude/, scripts/, ...). Mutates the current working tree.
#
#   With worktree creation:
#     bash scripts/bootstrap_claude_workflow.sh \
#         --branch dev-claude-alice \
#         --worktree ../myproject_alice \
#         [--source origin/dev] \
#         [--with-agentic] [--mind-meld]
#     Runs `git worktree add -b <branch> <worktree-path> <source>`, cd's
#     into the new worktree, then performs the in-place steps there.
#     Your existing checkout (e.g. master) is untouched. --source defaults
#     to origin/dev if it exists, else origin/master, else origin/main.
#     Requires that this script be accessible in the current directory —
#     if scaffolding is on a non-default branch, create a temp worktree
#     of the source branch first to get access to this script.
#
# Args:
#   --with-agentic    Install the SDK agent harness (git mind-meld alias,
#                     fresh agent_state/).
#   --mind-meld       Only with --with-agentic. Skip the agent_state wipe
#                     — graft from the source branch's accumulated state.
#   --branch <name>   Create a personal branch named <name>. Requires
#                     --worktree. Convention: dev-claude-NAME.
#   --worktree <path> Destination path for the new worktree. Requires
#                     --branch.
#   --source <ref>    Source branch to branch off (default: auto-detect
#                     origin/dev, then origin/master, then origin/main).
#   -h, --help        Print this help.

set -e

WITH_AGENTIC=0
MIND_MELD_GRAFT=0
BRANCH=""
WORKTREE=""
SOURCE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --with-agentic) WITH_AGENTIC=1; shift ;;
        --mind-meld)    MIND_MELD_GRAFT=1; shift ;;
        --branch)
            BRANCH="${2:-}"
            if [ -z "$BRANCH" ]; then
                echo "ERROR: --branch requires a value" >&2
                exit 1
            fi
            shift 2
            ;;
        --worktree)
            WORKTREE="${2:-}"
            if [ -z "$WORKTREE" ]; then
                echo "ERROR: --worktree requires a value" >&2
                exit 1
            fi
            shift 2
            ;;
        --source)
            SOURCE="${2:-}"
            if [ -z "$SOURCE" ]; then
                echo "ERROR: --source requires a value" >&2
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            sed -n '2,57p' "$0" | sed 's/^# //;s/^#//'
            exit 0
            ;;
        *)
            echo "WARNING: Unknown argument '$1' ignored" >&2
            shift
            ;;
    esac
done

# Validate --branch / --worktree pairing.
if [ -n "$BRANCH" ] && [ -z "$WORKTREE" ]; then
    echo "ERROR: --branch requires --worktree (where to create the worktree)" >&2
    exit 1
fi
if [ -n "$WORKTREE" ] && [ -z "$BRANCH" ]; then
    echo "ERROR: --worktree requires --branch (what branch to create)" >&2
    exit 1
fi

# If creating a worktree, resolve the source branch and run the git plumbing.
if [ -n "$BRANCH" ]; then
    echo "Preparing worktree + personal branch..."

    # Fetch so origin/* refs are up to date.
    git fetch origin 2>/dev/null || true

    # Auto-detect --source if not given.
    if [ -z "$SOURCE" ]; then
        for candidate in origin/dev origin/master origin/main; do
            if git rev-parse --verify "$candidate" >/dev/null 2>&1; then
                SOURCE="$candidate"
                break
            fi
        done
        if [ -z "$SOURCE" ]; then
            echo "ERROR: could not auto-detect source branch" >&2
            echo "       (tried origin/dev, origin/master, origin/main — none exist)" >&2
            echo "       pass --source explicitly" >&2
            exit 1
        fi
        echo "  Auto-detected source branch: $SOURCE"
    fi

    # Make sure the worktree destination doesn't already exist.
    if [ -e "$WORKTREE" ]; then
        echo "ERROR: --worktree path '$WORKTREE' already exists" >&2
        exit 1
    fi

    # Create the worktree on the new branch.
    git worktree add -b "$BRANCH" "$WORKTREE" "$SOURCE"
    echo "  Created worktree at $WORKTREE on branch $BRANCH (from $SOURCE)"

    # Hop into the new worktree for all subsequent steps.
    cd "$WORKTREE"
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Bootstrapping Claude Code workflow..."

# 1. Copy CLAUDE.md, stripping Serena sections
CLAUDE_SRC=".claude/CLAUDE.md"
CLAUDE_DST="CLAUDE.md"
if [ -f "$CLAUDE_SRC" ]; then
    if [ "$WITH_AGENTIC" -eq 1 ]; then
        # Agentic tier installs Serena + the agent infra, so KEEP the
        # sentinel-gated sections that document those capabilities.
        cp "$CLAUDE_SRC" "$CLAUDE_DST"
        echo "  Created $CLAUDE_DST (agentic tier: gated sections kept)"
    else
        # Non-agentic collaborators can't use Serena / agent-infra tools —
        # strip BOTH sentinel-gated sections (SERENA and AGENT INFRA).
        sed '/<!-- BEGIN SERENA SECTION/,/<!-- END SERENA SECTION -->/d;/<!-- BEGIN AGENT INFRA SECTION/,/<!-- END AGENT INFRA SECTION -->/d' \
            "$CLAUDE_SRC" > "$CLAUDE_DST"
        echo "  Created $CLAUDE_DST (gated sections stripped)"
    fi
elif [ -f "CLAUDE.md" ]; then
    echo "  CLAUDE.md already exists, skipping"
else
    echo "  WARNING: No CLAUDE.md source found"
fi

# 2. Create CLAUDE.local.md.example if it doesn't exist
LOCAL_EXAMPLE="CLAUDE.local.md.example"
if [ ! -f "$LOCAL_EXAMPLE" ]; then
    cat > "$LOCAL_EXAMPLE" << 'EXAMPLE_EOF'
# Local machine overrides (not tracked)

# Put machine-specific instructions here:
# - Conda environment name
# - Full Python interpreter path
# - Cluster access details
# - Local data paths

# Example:
# ## Python Path
# Always use: /opt/anaconda3/envs/myenv/bin/python
EXAMPLE_EOF
    echo "  Created $LOCAL_EXAMPLE"
fi

# 3. Agentic tier extras
if [ "$WITH_AGENTIC" -eq 1 ]; then
    echo ""
    echo "Setting up agentic tier..."

    # 3a. Install `git mind-meld` alias (requires scripts/mind_meld.sh to exist).
    MM_SCRIPT="$REPO_ROOT/scripts/mind_meld.sh"
    if [ -f "$MM_SCRIPT" ]; then
        # --local (not --worktree): git doesn't resolve aliases from worktree
        # config; --local is the working scope. Harmless in worktrees without
        # scripts/mind_meld.sh (alias just fails with "script not found").
        git config --local alias.mind-meld '!bash scripts/mind_meld.sh'
        echo "  Installed: git mind-meld <branch>"
    else
        echo "  WARNING: $MM_SCRIPT not found — mind-meld alias not installed."
        echo "  Run this script again after installing the skill's mind-meld support."
    fi

    # 3b. Memory policy: fresh start by default, inherit on --mind-meld.
    #
    # Default --with-agentic (fresh): wipe every path listed in
    # .claude/memory-blacklist.txt — both [soft] (per-agent memories) and
    # [hard] (per-machine state). The collaborator's workflow diverges from
    # the source branch from their first /build.
    #
    # --mind-meld (graft): keep [soft] paths intact — inherit the source
    # branch's accumulated agent memories wholesale. [hard] paths are still
    # wiped (per-machine state has no meaningful shared content).
    #
    # If .claude/memory-blacklist.txt doesn't exist (mind-meld not installed
    # for this project via Q7), fall back to wiping only .claude/agent_state/.
    BLACKLIST="$REPO_ROOT/.claude/memory-blacklist.txt"
    if [ "$MIND_MELD_GRAFT" -eq 1 ]; then
        echo "  --mind-meld: keeping source's soft-blacklisted agent memories intact."
        # Still wipe [hard] paths even under --mind-meld.
        if [ -f "$BLACKLIST" ]; then
            _wipe_section="hard"
        else
            _wipe_section=""
        fi
    elif [ -f "$BLACKLIST" ]; then
        echo "  Wiping soft+hard blacklisted paths for fresh-start workflow."
        _wipe_section="both"
    else
        # Fallback: no blacklist, just wipe agent_state.
        STATE_DIR="$REPO_ROOT/.claude/agent_state"
        if [ -d "$STATE_DIR" ]; then
            rm -rf "$STATE_DIR"
            echo "  Wiped $STATE_DIR (memory-blacklist.txt not present, fallback)."
        fi
        _wipe_section=""
    fi

    if [ -n "$_wipe_section" ] && [ -f "$BLACKLIST" ]; then
        _cur_tier=""
        while IFS= read -r line; do
            line="${line%%#*}"
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            [ -z "$line" ] && continue
            case "$line" in
                "[soft]") _cur_tier="soft"; continue ;;
                "[hard]") _cur_tier="hard"; continue ;;
            esac
            case "$_wipe_section" in
                hard) [ "$_cur_tier" = "hard" ] || continue ;;
                both) [ "$_cur_tier" = "soft" ] || [ "$_cur_tier" = "hard" ] || continue ;;
            esac
            _target="$REPO_ROOT/$line"
            if [ -e "$_target" ]; then
                rm -rf "$_target"
                echo "    wiped $line"
            fi
        done < "$BLACKLIST"
    fi

    # 3c. Ensure memory-blacklist.txt is in place (template from assets/).
    if [ ! -f "$BLACKLIST" ]; then
        echo "  NOTE: .claude/memory-blacklist.txt not found."
        echo "  Copy it from the skill's assets/memory-blacklist.txt and customize"
        echo "  for your project's agent memory layout."
    fi
fi

echo ""
echo "Done. To start using Claude Code:"
echo "  1. Copy $LOCAL_EXAMPLE to CLAUDE.local.md and customize"
if [ -n "$BRANCH" ]; then
    echo "  2. Open Claude Code in $WORKTREE/"
else
    echo "  2. Open Claude Code in this directory"
fi
echo "  3. The agent will read CLAUDE.md for project conventions"

if [ "$WITH_AGENTIC" -eq 1 ]; then
    cat <<'AGENTIC_TIPS'

Agentic tier enabled. Extra capabilities:
  - git mind-meld <branch>     LLM-assisted semantic merge of agent memories
                               (see references/mind-meld.md for details)
  - git mind-meld --git <br>   Plain 3-way merge, manual conflict resolution
  - Plain git pull / merge     Memory filter auto-reverts [soft] blacklist
                               paths so your agent memories aren't clobbered.

AGENTIC_TIPS
fi

echo "For the full agent pipeline (Serena, SDK builds, agent crew):"
echo "  Install TejaForce: /teja-force"
