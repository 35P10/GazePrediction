#!/usr/bin/env bash
# Host entry point (Linux + NVIDIA GPU + Docker + NVIDIA Container Toolkit).
#   ./run.sh                  download, verify, prepare, train 6 epochs, evaluate
#   ./run.sh train --epochs 1 any single stage; extra arguments go to that stage
#   ./run.sh export --images /work/my_charts   maps for new charts (put them under $UMSS_WORK)
# Outputs go to $UMSS_WORK (default: <repo>/artifacts/runs/umss/massvis_md_eam_validation).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$HERE/../../.." && pwd)
WORK=${UMSS_WORK:-$PROJECT_ROOT/artifacts/runs/umss/massvis_md_eam_validation}
IMAGE=${UMSS_IMAGE:-umss-train}
mkdir -p "$WORK"

docker build -f "$HERE/Dockerfile" ${TF1_TAG:+--build-arg TF1_TAG=$TF1_TAG} -t "$IMAGE" "$HERE/.."

# --shm-size and ulimits follow NVIDIA's recommendations for their TF containers.
exec docker run --rm --gpus all --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" -e HOME=/work -e KERAS_HOME=/work/.keras \
    -v "$WORK":/work "$IMAGE" "$@"
