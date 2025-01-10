#!/usr/bin/env bash
set -euxo pipefail

function check_branch_exists_on_remote() {
  local branch_name="$1"
  local remote="${2:-origin}" # Default to 'origin' if no remote is specified

  if git ls-remote --heads "$remote" "$branch_name" | grep -q "refs/heads/$branch_name"; then
    echo "Branch '$branch_name' exists on remote '$remote'."
    return 0
  else
    echo "Branch '$branch_name' does not exist on remote '$remote'."
    return 1
  fi
}

branch="$(git branch --show-current)"
proc_name="agentao-validator"

git fetch origin "$branch"

local_version=$(git rev-parse "$branch")
remote_version=$(git rev-parse origin/"$branch")
if $(check_branch_exists_on_remote "$branch") && [$local_version != $remote_version]; then
  echo "Local code is out of date (sha $local_version), updating to ($remote_version)..."
  git rebase FETCH_HEAD

  python3 -m pip install -e .
  pm2 restart "$proc_name"
fi

git rev-parse "$branch"
pm2 restart "$proc_name"
