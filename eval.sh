#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}

PYTHON=${PYTHON:-python}
MODEL=${MODEL:-weights/5000_fd-kitti.pth}
DATASET=${DATASET:-kitti}
TASK=${TASK:-validation}
OUTPUT=${OUTPUT:-kitti_submission}
BASE_CKPT=${BASE_CKPT:-weights/100000_raft-sintel.pth}
BASE_DEVICE=${BASE_DEVICE:-cpu}
KITTI_ROOT=${KITTI_ROOT:-data/KITTI}

${PYTHON} -m flow_painter.cli.evaluate \
  --model ${MODEL} \
  --dataset ${DATASET} \
  --task ${TASK} \
  --output_path ${OUTPUT} \
  --base_model_ckpt ${BASE_CKPT} \
  --base_model_device ${BASE_DEVICE} \
  --kitti_root ${KITTI_ROOT}
