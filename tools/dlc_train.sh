# conda initialize
__conda_setup="$('/cpfs01/user/zhouyunsong/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cpfs01/user/zhouyunsong/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/cpfs01/user/zhouyunsong/opt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/cpfs01/user/zhouyunsong/opt/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate open-mmlab

# NCCL配置，为阿里云调优过的，不需要改动，直接copy进去就行
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO

# 这里的WORK_DIR为log输出文件夹
# 默认设定为此shell脚本所在路径，如在网页中调用或想设置成其他文件夹，请手动指定
WORK_DIR=/cpfs01/user/zhouyunsong/zhouys/Git_repos/OccupancyNetwork_nuplan/work_dirs
CODE_HOME=/cpfs01/user/zhouyunsong/zhouys/Git_repos/OccupancyNetwork_nuplan
CONFIG=projects/configs/bevformer/bev_tiny_occ_v2-99_nuplan.py

export PYTHONPATH=$CODE_HOME:$PYTHONPATH

# cd到bevformer根目录
cd $CODE_HOME

python -m torch.distributed.run \
    --nproc_per_node=${KUBERNETES_CONTAINER_RESOURCE_GPU} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR}
# 使用tee记录log时，多个节点会同时记录log，可能产生冲突