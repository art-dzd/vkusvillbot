#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/macmini/dev-server/vkusvillbot"
BRANCH="main"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Repo not found: $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"

echo "[deploy] fetch"
git fetch --all --prune

echo "[deploy] reset"
git reset --hard "origin/$BRANCH"

echo "[deploy] clean"
# keep local secrets and data
if [ -f .env ]; then
  git clean -ffdx -e .env -e data -e logs
else
  git clean -ffdx -e data -e logs
fi

if [ ! -f .env ]; then
  echo "[deploy] WARNING: .env not found in repo dir" >&2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[deploy] docker not found" >&2
  exit 1
fi

echo "[deploy] docker compose up"
docker compose up -d --build

sleep 1

docker compose ps
