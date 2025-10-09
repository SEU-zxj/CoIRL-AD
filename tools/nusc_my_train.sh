#!/usr/bin/env bash
CONFIG=projects/configs/$1.py
# echo $CONFIG
GPUS=$2
MODEL_ID=$3
PORT=${PORT:-59231}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --work-dir work_dirs/$3 \
    --launcher pytorch ${@:4} \
    --deterministic \
    --cfg-options evaluation.jsonfile_prefix=work_dirs/$3/eval/results 
    # evaluation.classwise=True
