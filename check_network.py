#!/usr/bin/env python3
"""
网络配置检查脚本
用于验证NCCL分布式通信的网络环境
"""

import os
import socket
import subprocess
import sys

def get_local_ip():
    """获取本地IP地址"""
    try:
        # 获取所有网络接口的IP
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            # 查找192.168.100.x的IP
            for ip in ips:
                if ip.startswith('192.168.100.'):
                    return ip
            # 如果没有找到，返回第一个非回环地址
            for ip in ips:
                if not ip.startswith('127.'):
                    return ip
        return None
    except Exception:
        return None

def check_network_interface():
    """检查指定的网络接口是否存在"""
    interface = "enp1s0f0np0"
    try:
        result = subprocess.run(['ip', 'link', 'show', interface], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ 网络接口 {interface} 存在")
            return True
        else:
            print(f"✗ 网络接口 {interface} 不存在")
            print("可用的网络接口:")
            subprocess.run(['ip', 'link', 'show'], check=False)
            return False
    except Exception as e:
        print(f"✗ 检查网络接口时出错: {e}")
        return False

def check_cuda():
    """检查CUDA环境"""
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU{i}: {torch.cuda.get_device_name(i)}")
        return torch.cuda.is_available()
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    except Exception as e:
        print(f"✗ CUDA检查失败: {e}")
        return False

def main():
    """主检查函数"""
    print("=" * 50)
    print("NCCL分布式通信环境检查")
    print("=" * 50)
    
    # 获取本地IP
    local_ip = get_local_ip()
    if local_ip:
        print(f"本地IP: {local_ip}")
    else:
        print("✗ 无法获取本地IP")
        return
    
    # 检查网络接口
    print("\n1. 网络接口检查")
    interface_ok = check_network_interface()
    
    # 检查CUDA
    print("\n2. CUDA环境检查")
    cuda_ok = check_cuda()
    
    # 确定节点角色
    print("\n3. 节点角色确定")
    if local_ip == "192.168.100.10":
        print("✓ 当前节点: 主节点 (Rank 0)")
    elif local_ip == "192.168.100.11":
        print("✓ 当前节点: 从节点 (Rank 1)")
    else:
        print(f"⚠ 当前IP {local_ip} 不在预期范围内")
        print("  预期IP: 192.168.100.10 或 192.168.100.11")
    
    # 总结
    print("\n" + "=" * 50)
    print("检查结果总结")
    print("=" * 50)
    
    if interface_ok and cuda_ok:
        print("✓ 环境检查通过，可以运行NCCL分布式通信")
        print("\n运行命令:")
        if local_ip == "192.168.100.10":
            print("  主节点: python3 simple_nccl_test.py 0")
        elif local_ip == "192.168.100.11":
            print("  从节点: python3 simple_nccl_test.py 1")
        else:
            print("  请手动设置正确的IP地址")
    else:
        print("✗ 环境检查未通过，请解决上述问题")

if __name__ == "__main__":
    main()
