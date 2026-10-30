#!/bin/sh
# Usage: ./build-docker-aisim.sh [base] [tag]
#   (defaults vllm-sim:cpu, built by ./build-docker-vllm.sh, and vllm-sim:cpu-aisim)
# Layers the pinned AISimulate wheel (docker/aisimulate-requirements.txt) on
# the base image for `timing: aisimulate`.
set -eu

BASE=${1:-vllm-sim:cpu}
TAG=${2:-vllm-sim:cpu-aisim}
ROOT=$(cd "$(dirname "$0")" && pwd)

docker build -f "$ROOT/docker/Dockerfile.aisim" --build-arg BASE="$BASE" -t "$TAG" "$ROOT/docker"
echo "base $BASE $(docker image inspect -f '{{.Id}}' "$BASE")"
echo "image $TAG $(docker image inspect -f '{{.Id}}' "$TAG")"
