#!/usr/bin/env bash
set -Eeuo pipefail

AWS_REGION="ap-southeast-2"
AWS_ACCOUNT_ID="970734043001"
ECR_REPOSITORY="gugaon_estate_ecr"
CONTAINER_NAME="gurgaon-estate"
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

docker pull "${IMAGE_URI}"
docker rm --force "${CONTAINER_NAME}" >/dev/null 2>&1 || true

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
