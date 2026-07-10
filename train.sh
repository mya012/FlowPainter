#!/usr/bin/env bash
set -e

mkdir -p checkpoints

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PYTHON=${PYTHON:-python}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
BASE_CKPT=${BASE_CKPT:-weights/100000_raft-sintel.pth}
COMMON="--gpus ${GPUS} --checkpoint_dir checkpoints --base_model_ckpt ${BASE_CKPT} --accumulation_steps 3"

${PYTHON} -m flow_painter.cli.train ${COMMON} --name fd-chairs --stage chairs --validation chairs --num_steps 20000 --batch_size 12 --lr 0.00045 --image_size 384 512 --wdecay 0.0001
${PYTHON} -m flow_painter.cli.train ${COMMON} --name fd-things --stage things --validation sintel --restore_ckpt checkpoints/fd-chairs.pth --num_steps 50000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.0001
${PYTHON} -m flow_painter.cli.train ${COMMON} --name fd-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/fd-things.pth --num_steps 50000 --batch_size 6 --lr 0.000175 --image_size 432 960 --wdecay 0.00001 --gamma 0.85
${PYTHON} -m flow_painter.cli.train ${COMMON} --name fd-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/fd-sintel.pth --num_steps 15000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma 0.85

