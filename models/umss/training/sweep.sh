#!/usr/bin/env bash
# How much S1 scores move with the training choices the authors left open: the seed (the notebook sets
# none) and the epoch count (6 in the paper, 9 in the notebook). Every epoch of every run is evaluated
# against Table 2. Run after download and prepare, on the host:
#   ./sweep.sh               MICRO_BATCH=8 ./sweep.sh on GPUs with room for 8 images
# Each run is kept in $UMSS_WORK/sweep/<run>/: ckpt/ and eval_epNN.json (Table 2 means). A run already
# there is skipped. An existing $UMSS_WORK/ckpt is moved in first as run seed0_6ep.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$HERE/../../.." && pwd)
export UMSS_WORK=${UMSS_WORK:-$PROJECT_ROOT/artifacts/runs/umss/massvis_md_eam_validation}
WORK=$UMSS_WORK
MICRO=${MICRO_BATCH:-4}
mkdir -p "$WORK/sweep"

if [ -d "$WORK/ckpt" ] && [ ! -e "$WORK/sweep/seed0_6ep" ]; then
    mkdir "$WORK/sweep/seed0_6ep" && mv "$WORK/ckpt" "$WORK/sweep/seed0_6ep/ckpt"
fi

train_run() {  # name, then train arguments
    local name=$1; shift
    [ -d "$WORK/sweep/$name/ckpt" ] && return
    [ -e "$WORK/ckpt" ] && { echo "$WORK/ckpt exists; move it away first" >&2; exit 1; }
    "$HERE/run.sh" train --micro-batch "$MICRO" "$@"
    mkdir -p "$WORK/sweep/$name"
    mv "$WORK/ckpt" "$WORK/sweep/$name/ckpt"
}

evaluate_run() {
    local name=$1 checkpoint epoch
    for checkpoint in "$WORK/sweep/$name"/ckpt/*_ep*.hdf5; do
        epoch=$(basename "$checkpoint" | sed -E 's/.*_ep([0-9]+)_.*/\1/')
        [ -e "$WORK/sweep/$name/eval_ep$epoch.json" ] && continue
        "$HERE/run.sh" evaluate --checkpoint "/work/sweep/$name/ckpt/$(basename "$checkpoint")" | tail -6
        cp "$WORK/predictions/paper_saliency_summary.json" "$WORK/sweep/$name/eval_ep$epoch.json"
    done
}

train_run seed1_6ep --seed 1
train_run seed2_6ep --seed 2
train_run seed3_6ep --seed 3
train_run seed0_9ep --seed 0 --epochs 9
for run in seed0_6ep seed1_6ep seed2_6ep seed3_6ep seed0_9ep; do
    [ -d "$WORK/sweep/$run/ckpt" ] && evaluate_run "$run"
done

python3 - "$WORK/sweep" <<'EOF'
import json, sys
from pathlib import Path
print('Table 2 means (paper: 3 s NSS 1.406 CC 0.586 KL 0.754 SIM 0.516 | 5 s NSS 1.024 CC 0.514 KL 0.689 SIM 0.530)')
print('%-10s %5s | %6s %6s %6s %6s | %6s %6s %6s %6s' % ('run', 'epoch', *('NSS CC KL SIM'.split() * 2)))
for path in sorted(Path(sys.argv[1]).glob('*/eval_ep*.json')):
    means = json.loads(path.read_text())['means']
    row = [means[t]['ours'][m] for t in ('3000', '5000') for m in ('NSS', 'CC', 'KL', 'SIM')]
    print('%-10s %5s | %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f %6.3f'
          % (path.parent.name, path.stem[len('eval_ep'):], *row))
EOF
