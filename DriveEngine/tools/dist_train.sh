#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
RATE=$3
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4} --deterministic 
# --limited-rate $RATE
