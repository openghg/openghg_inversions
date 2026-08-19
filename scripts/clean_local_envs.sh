#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)

for environment in "$repo_root/.venv" "$repo_root/.tox"; do
    if [[ -e "$environment" ]]; then
        printf 'Removing %s\n' "$environment"
        rm -rf -- "$environment"
    fi
done
