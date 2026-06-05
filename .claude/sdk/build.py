#!/usr/bin/env python
"""Launch the SDK build pipeline.

Usage (from the project root):
    python .claude/sdk/build.py build "Add feature X to module Y"
    python .claude/sdk/build.py build --fast "Fix typo in module.py"
    python .claude/sdk/build.py build --plan-only "Refactor data loading"
    python .claude/sdk/build.py build -v "Add new processor"
    python .claude/sdk/build.py build --no-serena "Quick fix"

Verbosity:
    (default)   Phase transitions + agent tool calls
    -v          Full agent streaming (thinking, text, tool args)
    -q          Phase transitions + final report only
    --log FILE  Also write all output to a log file (tail -f from another terminal)
"""

import os
import sys

# Allow importing sdk.* from .claude/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sdk.cli import main

if __name__ == "__main__":
    main()
