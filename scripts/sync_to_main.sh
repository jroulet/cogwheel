#!/usr/bin/env bash
# sync_to_main.sh — sync code from dev branch to main, excluding agent files.
#
# Pushes code changes to main while keeping agent-only files
# (SDK, crew prompts, hooks, Serena config) on the dev branch only.
# Collaborators who don't use the agent pipeline get clean code
# without agent infrastructure cluttering their checkout.
#
# Usage:
#   bash scripts/sync_to_main.sh              # interactive (shows diff, asks for confirmation)
#   bash scripts/sync_to_main.sh --dry-run    # show what would change without doing it
#
# IMPORTANT: This script must be run manually — never by an agent.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

DEV_BRANCH="$(git branch --show-current)"
MAIN_BRANCH="main"

# Agent-only paths to exclude from sync
EXCLUDE_PATHS=(
    ".claude/"
    ".codex/"
    ".agents/"
    ".serena/"
    ".mcp.json"
    "AGENTS.md"
    "CLAUDE.md"
    "scripts/bootstrap_claude_workflow.sh"
    "scripts/sync_to_main.sh"
    "scripts/verify_installation.sh"
)

DRY_RUN=0
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=1
fi

# Safety checks
if [ "$DEV_BRANCH" = "$MAIN_BRANCH" ] || [ "$DEV_BRANCH" = "master" ]; then
    echo "ERROR: You are on $DEV_BRANCH. Switch to your dev branch first."
    exit 1
fi

if ! git diff --quiet; then
    echo "ERROR: Working tree has uncommitted changes. Commit or stash first."
    exit 1
fi

# Build the diff (excluding agent paths)
exclude_args=""
for path in "${EXCLUDE_PATHS[@]}"; do
    exclude_args="$exclude_args -- . ':!$path'"
done

echo "Comparing $DEV_BRANCH → $MAIN_BRANCH (excluding agent files)..."
echo ""

# Show what would change
eval git diff "$MAIN_BRANCH...$DEV_BRANCH" --stat $exclude_args

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "(dry run — no changes made)"
    exit 0
fi

echo ""
read -p "Proceed with sync to $MAIN_BRANCH? [y/n] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Checkout main, cherry-pick or merge selectively
git checkout "$MAIN_BRANCH"

# Use git checkout to selectively bring files from dev
# (excluding agent paths)
for path in "${EXCLUDE_PATHS[@]}"; do
    # Stash the main branch's version of excluded paths
    if [ -e "$path" ]; then
        git stash push -q -- "$path" 2>/dev/null || true
    fi
done

git merge "$DEV_BRANCH" --no-edit --no-ff -m "Merge $DEV_BRANCH into $MAIN_BRANCH (code only, agent files excluded)"

# Restore excluded paths to their main-branch state
for path in "${EXCLUDE_PATHS[@]}"; do
    git checkout "$MAIN_BRANCH~1" -- "$path" 2>/dev/null || true
done

# If any agent files leaked through, unstage them
for path in "${EXCLUDE_PATHS[@]}"; do
    git reset HEAD -- "$path" 2>/dev/null || true
    git checkout -- "$path" 2>/dev/null || true
done

if ! git diff --cached --quiet; then
    git commit --amend --no-edit
fi

echo ""
echo "Synced to $MAIN_BRANCH. Review with: git log --oneline -5"
echo "Push with: git push origin $MAIN_BRANCH"
echo ""

# Return to dev branch
git checkout "$DEV_BRANCH"
