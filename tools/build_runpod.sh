#!/bin/bash
# Build the linux/amd64 image (e.g. on Apple Silicon) and push it for RunPod.
# Usage: tools/build_runpod.sh <dockerhub_user>/supermimic[:tag]
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${1:?usage: tools/build_runpod.sh <dockerhub_user>/supermimic[:tag]}
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
if [ ! -d IsaacLab ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git
fi
git -C IsaacLab checkout "$ISAACLAB_COMMIT"
# active Docker Desktop builder (docker driver), so the second build sees the locally loaded base image
(cd IsaacLab && docker buildx build --platform linux/amd64 --load -f docker/Dockerfile.kitless -t isaac-lab-kitless:main-46d6cc3 .)
docker buildx build --platform linux/amd64 --load -f docker/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
