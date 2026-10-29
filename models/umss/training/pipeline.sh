#!/usr/bin/env bash
# Stages run inside the container; everything is written to /work (a mounted host folder).
set -euo pipefail
cd /umss/training/pipeline
WORK=${WORK:-/work}

stage=${1:-all}
shift || true

download() { python download.py --work "$WORK"; }
verify()   { python ../../reference/verify.py --work "$WORK"; }
check()    { python prepare_data.py --work "$WORK" --check; }
prepare()  { python prepare_data.py --work "$WORK"; }
train()    { python train.py --work "$WORK" "$@"; }
evaluate() { python predict.py --work "$WORK" "$@" evaluate; }
export_()  { python predict.py --work "$WORK" export "$@"; }

case "$stage" in
  download|verify|check|prepare) "$stage" ;;
  train)  train "$@" ;;
  evaluate) evaluate "$@" ;;
  export) export_ "$@" ;;
  all)    download; check; prepare; train "$@"; evaluate ;;
  shell)  exec bash ;;
  *) echo "usage: pipeline.sh [all|download|verify|check|prepare|train|evaluate [--checkpoint FILE]|export --images DIR|shell]" >&2; exit 2 ;;
esac
