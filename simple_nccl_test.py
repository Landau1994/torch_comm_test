#!/usr/bin/env python3
"""
简化的NCCL分布式通信测试
适用于手动在两个节点上分别运行
"""

import os
import torch
import torch.distributed as dist
from datetime import timedelta

def setup_environment(rank, master_ip='192.168.100.10'):
    """设置环境变量"""
    os.environ["NCCL_SOCKET_IFNAME"] = "enp1s0f0np0"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = '12355'
    
    print(f"环境设置完成 - Rank: {rank}")
    print(f"NCCL_SOCKET_IFNAME: {os.environ.get('NCCL_SOCKET_IFNAME')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

def test_nccl_communication(rank, world_size=2):
    """测试NCCL通信"""
    print(f"\n开始测试NCCL通信 - Rank: {rank}")
    
    try:
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5)
        )
        
        # 设置CUDA设备
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        
        print(f"Rank {rank}: 进程组初始化成功")
        
        # 测试1: All-Reduce
        print(f"\nRank {rank}: 测试All-Reduce...")
        tensor = torch.tensor([rank + 1.0, rank + 2.0]).to(device)
        print(f"Rank {rank}: 初始张量 {tensor}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank}: All-Reduce结果 {tensor}")
        
        # 测试2: Broadcast
        print(f"\nRank {rank}: 测试Broadcast...")
        if rank == 0:
            broadcast_tensor = torch.tensor([100.0, 200.0, 300.0]).to(device)
        else:
            broadcast_tensor = torch.zeros(3).to(device)
        
        print(f"Rank {rank}: 广播前 {broadcast_tensor}")
        dist.broadcast(broadcast_tensor, src=0)
        print(f"Rank {rank}: 广播后 {broadcast_tensor}")
        
        # 测试3: 点对点通信
        print(f"\nRank {rank}: 测试点对点通信...")
        if rank == 0:
            send_tensor = torch.tensor([1.5, 2.5, 3.5]).to(device)
            dist.send(send_tensor, dst=1)
            print(f"Rank {rank}: 发送张量 {send_tensor}")
        else:
            recv_tensor = torch.zeros(3).to(device)
            dist.recv(recv_tensor, src=0)
            print(f"Rank {rank}: 接收张量 {recv_tensor}")
        
        # 同步
        dist.barrier()
        print(f"Rank {rank}: 所有测试完成")
        
    except Exception as e:
        print(f"Rank {rank}: 错误 - {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"Rank {rank}: 进程组已销毁")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python simple_nccl_test.py <rank>")
        print("在主节点(192.168.100.10): python simple_nccl_test.py 0")
        print("在从节点(192.168.100.11): python simple_nccl_test.py 1")
        sys.exit(1)
    
    rank = int(sys.argv[1])
    if rank not in [0, 1]:
        print("错误: rank必须是0或1")
        sys.exit(1)
    
    # 根据rank确定主节点IP
    master_ip = '192.168.100.10'
    
    setup_environment(rank, master_ip)
    test_nccl_communication(rank)

if __name__ == "__main__":
    main()
