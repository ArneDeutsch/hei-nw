#!/usr/bin/env bash
set -euo pipefail

# Fail the build if any stub markers are present in the repository.
if git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" -- \
    . ':!tests' ':!prompts' ':!planning' ':!reviews' ':!.github/workflows/ci.yml' \
    ':!scripts/grep_no_stubs.sh'; then
  echo "Stubs found."
  exit 1
else
  echo "No stubs."
fi
