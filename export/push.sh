#!/bin/bash
# Upload shards with the hub's storage client, as root with the hub's environment (keys never leave it). Usage:
#   export/push.sh STAGE MANIFEST PROJECT [--delete-local] > pushed.jsonl
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec sudo -n systemd-run --quiet --wait --pipe --collect -p EnvironmentFile=/etc/scrape-hub/env -p WorkingDirectory=/opt/scrape-hub \
  -p Nice=15 -p IOSchedulingClass=best-effort -p IOSchedulingPriority=7 \
  /opt/scrape-hub/.venv/bin/python "$HERE/push.py" "$@"
