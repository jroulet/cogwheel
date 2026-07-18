# Cogwheel repo / worktree layout (migrated from Claude auto-memory, 2026-07-18)

- Main checkout: /home/tejaswi/Work/cogwheel (origin
  git@github.com:jroulet/cogwheel.git); `main` stays checked out there.
  User has push access as GitHub user **ntveem**.
- Agentic work happens on the `claude-dev` branch, checked out as a git
  worktree at /home/tejaswi/Work/cogwheel-claude-dev (sibling folder).
- The same layout exists on the user's laptop. Set up 2026-07-17.
- Machine SSH quirk: GitHub over SSH on this box needs `CheckHostIP no`
  in ~/.ssh/config (detail in Claude auto-memory
  `github-ssh-checkhostip-fix`).
