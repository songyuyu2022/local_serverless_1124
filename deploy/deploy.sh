#!/usr/bin/env bash
set -euo pipefail
: "${REGISTRY:?Set REGISTRY=registry.cn-xxx.aliyuncs.com/namespace}"
docker build -f docker/Dockerfile.pre    -t "$REGISTRY/moe-pre:latest" .
docker build -f docker/Dockerfile.post   -t "$REGISTRY/moe-post:latest" .
docker build -f docker/Dockerfile.expert -t "$REGISTRY/moe-expert:latest" .
docker push "$REGISTRY/moe-pre:latest"
docker push "$REGISTRY/moe-post:latest"
docker push "$REGISTRY/moe-expert:latest"
s deploy -y
