#!/usr/bin/env bash
set -Eeuo pipefail

AWS_REGION="ap-southeast-2"
AWS_ACCOUNT_ID="970734043001"
ECR_REPOSITORY="gugaon_estate_ecr"
CONTAINER_NAME="gurgaon-estate"
LEGACY_CONTAINER_NAME="my-app"
LEGACY_IMAGE="jubito27/sentiment-2:latest"
REVISION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_URI="$(cat "${REVISION_ROOT}/image_uri.txt")"
EXPECTED_PREFIX="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:"

if [[ "${IMAGE_URI}" != "${EXPECTED_PREFIX}"* ]]; then
  echo "Unexpected Gurgaon Estate image URI in deployment revision: ${IMAGE_URI}" >&2
  exit 1
fi

systemctl enable --now docker
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
      "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Pull the immutable image before stopping the currently running service.
docker pull "${IMAGE_URI}"

# This instance previously ran the unrelated Sentiment Predictor container
# under the generic name "my-app". Remove it only when its image matches the
# exact known legacy image; never remove an unknown workload automatically.
if docker inspect "${LEGACY_CONTAINER_NAME}" >/dev/null 2>&1; then
  LEGACY_RUNNING_IMAGE="$(docker inspect --format '{{.Config.Image}}' "${LEGACY_CONTAINER_NAME}")"
  if [[ "${LEGACY_RUNNING_IMAGE}" == "${LEGACY_IMAGE}" ]]; then
    echo "Removing stale Sentiment Predictor container from the Gurgaon host."
    docker rm --force "${LEGACY_CONTAINER_NAME}"
  else
    echo "Refusing to remove unexpected container '${LEGACY_CONTAINER_NAME}' using image '${LEGACY_RUNNING_IMAGE}'." >&2
    exit 1
  fi
fi

docker rm --force "${CONTAINER_NAME}" >/dev/null 2>&1 || true

REMAINING_PORT_80="$(docker ps --filter publish=80 --format '{{.Names}}')"
if [[ -n "${REMAINING_PORT_80}" ]]; then
  echo "Port 80 is still occupied by an unexpected container: ${REMAINING_PORT_80}" >&2
  exit 1
fi

docker run --detach \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --init \
  --publish 80:5000 \
  --memory 768m \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  "${IMAGE_URI}"

echo "Started ${CONTAINER_NAME} using ${IMAGE_URI}"
