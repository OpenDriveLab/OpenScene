#!/usr/bin/env bash
bash ./tools/dist_train.sh ./projects/configs/bevformer/bev_tiny_occ_v2-99_nuplan.py 8
# ./tools/dist_train.sh ./projects/configs/bevformer/bevformer_tiny.py 2 
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python $(dirname "$0")/train.py ./projects/configs/bevformer/bev_tiny_det_occ.py --gpu-ids 0 --deterministic --limited-rate 0.2
# ./tools/dist_train.sh ./projects/configs/bevformer/bev_tiny_det_occ.py 2 