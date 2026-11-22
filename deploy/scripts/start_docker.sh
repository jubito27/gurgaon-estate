#!/usr/bin/env bash
set -Eeuo pipefail

AWS_REGION="ap-southeast-2"
AWS_ACCOUNT_ID="970734043001"
ECR_REPOSITORY="gugaon_estate_ecr"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest"
CONTAINER_NAME="gurgaon-estate"

systemctl enable --now docker
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
      "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker pull "$IMAGE_URI"
docker rm --force "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --detach \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --init \
  --publish 80:5000 \
  --memory 1700m \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  "$IMAGE_URI"

echo "Started $CONTAINER_NAME using $IMAGE_URI"
