#!/usr/bin/env python3
"""
PyTorch NCCL分布式通信实现
适用于两个节点之间的通信
网卡: enp1s0f0np0
主节点: 192.168.100.10
从节点: 192.168.100.11
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta

def setup_nccl_env():
    """设置NCCL环境变量"""
    # 指定使用的网卡
    os.environ["NCCL_SOCKET_IFNAME"] = "enp1s0f0np0"
    
    # NCCL优化设置
    os.environ["NCCL_DEBUG"] = "INFO"  # 开启NCCL调试信息
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    os.environ["NCCL_IB_DISABLE"] = "0"  # 启用InfiniBand
    os.environ["NCCL_P2P_DISABLE"] = "0"  # 启用P2P传输
    
    # 超时设置
    os.environ["NCCL_TIMEOUT"] = "600000"  # 10分钟超时
    
    # 主节点设置
    os.environ['MASTER_ADDR'] = '192.168.100.10'
    os.environ['MASTER_PORT'] = '12355'  # 使用非默认端口避免冲突
    
    print("NCCL环境变量设置完成")
    print(f"NCCL_SOCKET_IFNAME: {os.environ.get('NCCL_SOCKET_IFNAME')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

def test_basic_communication(rank, world_size):
    """测试基本的分布式通信"""
    print(f"Rank {rank}: 开始测试基本通信...")
    
    # 创建张量
    if rank == 0:
        # 主节点发送数据
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda()
        print(f"Rank {rank}: 发送张量 {tensor}")
        dist.send(tensor, dst=1)
        print(f"Rank {rank}: 发送完成")
    else:
        # 从节点接收数据
        tensor = torch.zeros(4).cuda()
        print(f"Rank {rank}: 准备接收张量")
        dist.recv(tensor, src=0)
        print(f"Rank {rank}: 接收到张量 {tensor}")

def test_all_reduce(rank, world_size):
    """测试All-Reduce操作"""
    print(f"Rank {rank}: 开始测试All-Reduce...")
    
    # 每个rank创建不同的张量
    tensor = torch.tensor([rank + 1.0, rank + 2.0]).cuda()
    print(f"Rank {rank}: 初始张量 {tensor}")
    
    # 执行All-Reduce (求和)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: All-Reduce结果 {tensor}")
    
    # 验证结果
    expected = torch.tensor([1.0 + 2.0, 2.0 + 3.0]).cuda()
    if torch.allclose(tensor, expected):
        print(f"Rank {rank}: All-Reduce测试通过")
    else:
        print(f"Rank {rank}: All-Reduce测试失败")

def test_broadcast(rank, world_size):
    """测试Broadcast操作"""
    print(f"Rank {rank}: 开始测试Broadcast...")
    
    if rank == 0:
        # 主节点创建数据
        tensor = torch.tensor([10.0, 20.0, 30.0]).cuda()
        print(f"Rank {rank}: 广播张量 {tensor}")
    else:
        # 从节点创建空张量
        tensor = torch.zeros(3).cuda()
        print(f"Rank {rank}: 准备接收广播")
    
    # 从rank 0广播到所有rank
    dist.broadcast(tensor, src=0)
    print(f"Rank {rank}: 接收到广播 {tensor}")

def test_gather(rank, world_size):
    """测试Gather操作"""
    print(f"Rank {rank}: 开始测试Gather...")
    
    # 每个rank创建不同的数据
    local_data = torch.tensor([rank + 1.0]).cuda()
    print(f"Rank {rank}: 本地数据 {local_data}")
    
    if rank == 0:
        # 主节点收集所有数据
        gathered_data = [torch.zeros(1).cuda() for _ in range(world_size)]
        dist.gather(local_data, gather_list=gathered_data, dst=0)
        print(f"Rank {rank}: 收集到的数据 {[t.item() for t in gathered_data]}")
    else:
        # 从节点发送数据
        dist.gather(local_data, dst=0)
        print(f"Rank {rank}: 数据发送完成")

def run_distributed_task(rank, world_size, task_func):
    """运行分布式任务"""
    try:
        # 初始化进程组
        print(f"Rank {rank}: 正在初始化进程组...")
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10)
        )
        
        # 设置当前CUDA设备
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: 进程组初始化完成，使用GPU {rank}")
        
        # 执行测试任务
        task_func(rank, world_size)
        
        # 同步所有进程
        dist.barrier()
        print(f"Rank {rank}: 任务完成")
        
    except Exception as e:
        print(f"Rank {rank}: 错误 - {str(e)}")
        raise
    finally:
        # 清理
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"Rank {rank}: 进程组已销毁")

def main():
    """主函数"""
    # 设置环境
    setup_nccl_env()
    
    # 获取当前节点信息
    current_ip = os.popen('hostname -I').read().strip().split()[0]
    print(f"当前节点IP: {current_ip}")
    
    # 确定当前节点的rank
    if current_ip == '192.168.100.10':
        rank = 0
    elif current_ip == '192.168.100.11':
        rank = 1
    else:
        print(f"警告: 未知IP {current_ip}，默认使用rank 0")
        rank = 0
    
    world_size = 2
    
    print(f"开始分布式训练 - Rank: {rank}, World Size: {world_size}")
    
    # 运行测试
    test_functions = [
        test_basic_communication,
        test_all_reduce,
        test_broadcast,
        test_gather
    ]
    
    for test_func in test_functions:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_func.__name__}")
        print(f"{'='*50}")
        
        run_distributed_task(rank, world_size, test_func)
        
        # 等待一段时间确保清理完成
        time.sleep(2)

if __name__ == "__main__":
    main()
