#!/usr/bin/env bash
set -Eeuo pipefail

for attempt in $(seq 1 60); do
  if curl --fail --silent --show-error --max-time 5 http://127.0.0.1/health >/dev/null; then
    docker ps --filter "name=^/gurgaon-estate$" --format '{{.Names}} {{.Status}}'
    exit 0
  fi
  sleep 2
done

echo "Gurgaon Estate failed its local /health check." >&2
docker ps -a --filter "name=^/gurgaon-estate$" || true
docker logs --tail 200 gurgaon-estate || true
exit 1
