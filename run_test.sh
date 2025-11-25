#!/bin/bash

# NCCL测试启动脚本 - 使用conda环境

if [ $# -ne 1 ]; then
    echo "使用方法: $0 <rank>"
    echo "在主节点(192.168.100.10): $0 0"
    echo "在从节点(192.168.100.11): $0 1"
    exit 1
fi

RANK=$1

# 激活conda环境
source /home/wlt2025/anaconda3/bin/activate evo2

# 设置环境变量
export NCCL_SOCKET_IFNAME="enp1s0f0np0"
export NCCL_DEBUG="INFO"
export MASTER_ADDR="192.168.100.10"
export MASTER_PORT="12355"
export WORLD_SIZE="2"
export RANK="$RANK"

echo "启动NCCL测试..."
echo "Rank: $RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Interface: $NCCL_SOCKET_IFNAME"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

python3 simple_nccl_test.py $RANK
