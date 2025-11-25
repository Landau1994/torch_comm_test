#!/bin/bash

# NCCL分布式通信启动脚本

set -e

RANK=${1:-0}
MASTER_IP=${2:-192.168.100.10}

echo "启动NCCL分布式通信..."
echo "Rank: $RANK"
echo "Master IP: $MASTER_IP"

export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT="12355"
export WORLD_SIZE="2"
export RANK="$RANK"
export NCCL_SOCKET_IFNAME="enp1s0f0np0"
export NCCL_DEBUG="INFO"

python3 nccl_distributed.py
