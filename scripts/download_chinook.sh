#!/usr/bin/env bash
set -euo pipefail

# Downloads the Chinook SQLite sample database used by the agent.
# Source: https://github.com/lerocha/chinook-database

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
DEST_FILE="${DEST_DIR}/chinook.db"
URL="https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"

mkdir -p "${DEST_DIR}"
curl -fL "${URL}" -o "${DEST_FILE}"

echo "Chinook database downloaded to ${DEST_FILE}"
