#!/usr/bin/env bash
set -euo pipefail

git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" -- \
  . ':!tests' ':!prompts' ':!planning' ':!reviews' ':!.github/workflows/ci.yml' \
  ':!scripts/grep_no_stubs.sh' \
  || echo "No stubs."
