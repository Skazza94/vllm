#!/bin/sh
# Usage: ./build-docker-vllm.sh [tag]   (default vllm-sim:cpu)
# Environment: MAX_JOBS (default 4), extra `docker build` options in DOCKER_BUILD_ARGS.
set -eu

TAG=${1:-vllm-sim:cpu}
ROOT=$(cd "$(dirname "$0")" && pwd)
CONTEXT=$(mktemp -d)
trap 'rm -rf "$CONTEXT"' EXIT

git clone -q --depth 1 "file://$ROOT/vllm" "$CONTEXT/vllm"
echo "Building $TAG from vllm $(git -C "$CONTEXT/vllm" log --oneline -1)"
if [ -n "$(git -C "$ROOT/vllm" status --porcelain)" ]; then
    echo "Note: vllm/ has uncommitted changes, which the image leaves out."
fi

docker build -f "$CONTEXT/vllm/docker/Dockerfile.cpu" --target vllm-openai --build-arg VLLM_CPU_X86=true --build-arg ENABLE_AMX_FP8=0 --build-arg max_jobs="${MAX_JOBS:-4}" ${DOCKER_BUILD_ARGS:-} -t "$TAG" "$CONTEXT/vllm"
