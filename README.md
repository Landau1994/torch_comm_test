# torch_comm_test
Communication of two dgx-spark by torch
=======
# PyTorch NCCL分布式通信实现

## 项目概述
本项目实现了基于NCCL后端的PyTorch分布式通信，用于两个节点之间的高效通信。

## 网络配置
- **网卡**: enp1s0f0np0
- **主节点**: 192.168.100.10 (Rank 0)
- **从节点**: 192.168.100.11 (Rank 1)

## 文件说明

### 1. simple_nccl_test.py
简化的NCCL通信测试脚本，支持以下功能：
- All-Reduce操作测试
- Broadcast操作测试
- 点对点通信测试
- 自动环境配置

### 2. run_test.sh
启动脚本，简化运行流程

## 使用方法

### 步骤1: 检查网络配置
在两个节点上分别检查网卡是否存在：
```bash
ip link show enp1s0f0np0
```

### 步骤2: 在主节点启动 (192.168.100.10)
```bash
./run_test.sh 0
```

### 步骤3: 在从节点启动 (192.168.100.11)
```bash
./run_test.sh 1
```

或者手动运行：
```bash
# 主节点
python3 simple_nccl_test.py 0

# 从节点
python3 simple_nccl_test.py 1
```

## 环境变量说明

| 变量名 | 说明 | 值 |
|--------|------|---|
| NCCL_SOCKET_IFNAME | 指定通信网卡 | enp1s0f0np0 |
| MASTER_ADDR | 主节点IP地址 | 192.168.100.10 |
| MASTER_PORT | 通信端口 | 12355 |
| WORLD_SIZE | 总进程数 | 2 |
| RANK | 当前进程ID | 0或1 |
| NCCL_DEBUG | 调试信息级别 | INFO |

## 测试内容

### 1. All-Reduce测试
- 每个节点创建不同的张量
- 执行全局求和操作
- 验证结果正确性

### 2. Broadcast测试
- 主节点向所有节点广播数据
- 验证数据一致性

### 3. 点对点通信测试
- 主节点向从节点发送数据
- 从节点接收并验证数据

## 故障排除

### 常见问题

1. **NCCL初始化失败**
   - 检查网络连通性: `ping 192.168.100.11`
   - 检查端口是否开放: `nc -zv 192.168.100.10 12355`

2. **网卡不存在**
   - 使用 `ip link show` 查看可用网卡
   - 修改 `NCCL_SOCKET_IFNAME` 为正确的网卡名

3. **CUDA不可用**
   - 检查CUDA安装: `nvidia-smi`
   - 检查PyTorch CUDA支持: `python -c "import torch; print(torch.cuda.is_available())"`

### 调试命令

```bash
# 检查NCCL版本
python -c "import torch; print(torch.cuda.nccl.version())"

# 检查网络接口
ifconfig -a

# 检查端口占用
netstat -tulnp | grep 12355
```

## 性能优化建议

1. **网络优化**
   - 确保使用高速网络接口
   - 关闭网络节能模式

2. **NCCL优化**
   - 调整 `NCCL_NTHREADS` 参数
   - 启用 `NCCL_IB_DISABLE=0` 使用InfiniBand

3. **CUDA优化**
   - 确保GPU驱动版本兼容
   - 使用相同型号的GPU

## 扩展使用

### 多节点扩展
要扩展到更多节点，修改以下参数：
- `WORLD_SIZE`: 总节点数
- `RANK`: 每个节点的唯一ID
- 更新 `simple_nccl_test.py` 中的IP判断逻辑

### 自定义通信逻辑
在 `simple_nccl_test.py` 的 `test_nccl_communication` 函数中添加自定义的通信逻辑。

## 注意事项

1. 确保所有节点的PyTorch版本一致
2. 确保所有节点的CUDA版本一致
3. 确保防火墙允许指定端口的通信
4. 建议在相同的硬件环境下运行
