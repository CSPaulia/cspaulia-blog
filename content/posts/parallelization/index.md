---
title: "并行化"
date: 2026-07-18T10:00:00+08:00
series:
  main: "大语言模型"
  subseries: "系统与硬件"
draft: false
categories: ["大语言模型", "系统"]
tags: ["GPU", "CUDA", "并行计算", "训练"]
author: "CSPaulia"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "CS336 Lecture 7 & 8 学习笔记。"
disableHLJS: true
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
disableShare: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

上一讲讨论的是单个 GPU 内部的并行化；这一讲进一步讨论多个 GPU 之间的并行化。

<figure>
  <img src="gpu-node-overview.png" alt="多 GPU 节点的层级结构">
  <figcaption>多 GPU 节点的层级结构：GPU 内部包含 SM、L2 和 HBM，GPU 之间通过 NVLink / NVSwitch 连接，节点之间通过 InfiniBand / Ethernet 连接。</figcaption>
</figure>

## 1. 为什么需要多个 GPU

从访问速度来看，常见的硬件层级如下：

- 单节点、单 GPU：L1 缓存（L1 Cache）或共享内存（Shared Memory），速度最快；
- 单节点、单 GPU：高带宽内存（High Bandwidth Memory，HBM）；
- 单节点、多 GPU：通过 NVLink 或 NVSwitch 连接；
- 多节点、多 GPU：通过 InfiniBand 或以太网（Ethernet）连接，速度最慢。

层级越远，数据传输通常越慢，通信成本也越高。单个 GPU 内部可以通过算子融合（Operator Fusion）和分块（Tiling）减少内存访问；扩展到多个 GPU 或多个节点后，则需要通过复制（Replication）和分片（Sharding）减少 GPU 之间的通信。

为什么需要多 GPU？主要有两个原因：

1. 模型训练所需的参数、优化器状态、梯度和激活值无法放入单个 GPU。
2. 希望使用更多 GPU（更多浮点运算次数，Floating Point Operations，FLOPs）来更快训练模型。

## 2. 集合通信原语

集合通信（Collective Operations）是分布式编程中使用的概念性原语，是 20 世纪 80 年代并行编程文献中的经典概念。“集合”意味着：我们描述的是跨越多个设备的通用通信模式，而不是逐一管理点对点通信。这样通常能够获得更好的性能，也能让底层实现进一步优化通信过程。

<figure>
  <img src="ranks.png" alt="Rank 示意图">
  <figcaption>每个参与通信的设备都有一个 Rank，所有设备的数量称为 World Size。</figcaption>
</figure>

### 基本设置

在分布式环境中，每个设备或 GPU 称为一个秩（Rank），例如 Rank 0、Rank 1、Rank 2 和 Rank 3。所有设备的总数称为世界大小（World Size），这里的 World Size 是 4。

下面的代码沿用课程原文，以 `tensor(...)` 展示通信前后各个 Rank 持有的张量。它用于理解数据如何流动，而不是完整的分布式运行代码。

集合通信操作可以分为三类：

- Broadcast、Scatter、Gather 和 Reduce 是基础操作；
- All-gather、Reduce-scatter 和 All-reduce 是实际使用中的主要操作；
- All-to-all 则常用于混合专家模型（Mixture of Experts，MoE）。

### 广播（Broadcast）

Broadcast 将 Rank 0 上的数据复制到所有 Rank。例如，Rank 0 的输入为 `[0, 1, 2, 3]`，操作完成后，每个 Rank 都会得到 `[0, 1, 2, 3]`。

一个简单的使用场景是：Rank 0 加载初始检查点（Checkpoint），然后将其广播给所有 Rank。

```python
# Input
rank0 = tensor([0., 1, 2, 3])

# Output
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
```

### 分发（Scatter）

Scatter 将 Rank 0 上的张量切分后发送给所有 Rank。例如，Rank 0 上的 `[0, 1, 2, 3]` 会被分发为：

- Rank 0：`[0]`；
- Rank 1：`[1]`；
- Rank 2：`[2]`；
- Rank 3：`[3]`。

Scatter 是理解 Reduce-scatter 的基础。

```python
# Input
rank0 = tensor([0., 1, 2, 3])

# Output
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])
```

### 收集（Gather）

Gather 将所有 Rank 上的数据收集到 Rank 0，是 Scatter 的逆操作。例如，Rank 0、Rank 1、Rank 2 和 Rank 3 分别持有 `[0]`、`[1]`、`[2]` 和 `[3]`，Gather 后 Rank 0 得到 `[0, 1, 2, 3]`。

Gather 是理解 All-gather 的基础。

```python
# Input
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# Output
rank0 = tensor([0., 1, 2, 3])
```

### 归约（Reduce）

Reduce 将所有 Rank 上的数据收集到 Rank 0，并执行某种操作，例如求和、取最小值或取最大值。若四个 Rank 分别持有 `[0]`、`[1]`、`[2]` 和 `[3]`，使用求和操作后，Rank 0 得到 `[6]`。

Reduce 是理解 All-reduce 的基础。

```python
# Input
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# Output
rank0 = tensor([6.])  # Sum of all ranks (0 + 1 + 2 + 3)
```

### 全收集（All-gather）

All-gather 与 Gather 类似，但它会把收集结果发送给所有 Rank，而不只是 Rank 0。若每个 Rank 持有一个参数分片，All-gather 可以让每个 Rank 都获得完整参数，以便执行前向传播。

```python
# Input
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# Output
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
```

### 归约-分发（Reduce-scatter）

Reduce-scatter 会沿每个维度执行 Reduce，再将结果分发给各个 Rank。例如：

- Rank 0 输入 `[0, 1, 2, 3]`，得到 `[6]`；
- Rank 1 输入 `[1, 2, 3, 4]`，得到 `[10]`；
- Rank 2 输入 `[2, 3, 4, 5]`，得到 `[14]`；
- Rank 3 输入 `[3, 4, 5, 6]`，得到 `[18]`。

一个使用场景是：反向传播后，将不同数据分片产生的梯度相加，同时把梯度的存储分散到不同 Rank。

```python
# Input
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])

# Output
rank0 = tensor([6.])   # Sum along dim 0 (0 + 1 + 2 + 3)
rank1 = tensor([10.])  # Sum along dim 1 (1 + 2 + 3 + 4)
rank2 = tensor([14.])  # Sum along dim 2 (2 + 3 + 4 + 5)
rank3 = tensor([18.])  # Sum along dim 3 (3 + 4 + 5 + 6)
```

### 全归约（All-reduce）

```
All-reduce = Reduce-scatter + All-gather
```

All-reduce 等价于 Reduce-scatter 加 All-gather。对于上一节的输入，操作完成后每个 Rank 都会得到 `[6, 10, 14, 18]`。

All-reduce 可以在反向传播后汇总不同数据分片产生的梯度，同时复制完整参数。将 All-reduce 拆分为 Reduce-scatter 和 All-gather，还能提供更大的灵活性，例如支持 ZeRO 和完全分片数据并行（Fully Sharded Data Parallel，FSDP）。

```python
# Input
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])

# Output
rank0 = tensor([6., 10, 14, 18])
rank1 = tensor([6., 10, 14, 18])
rank2 = tensor([6., 10, 14, 18])
rank3 = tensor([6., 10, 14, 18])
```

### 全互换（All-to-all）

All-to-all 是最通用的集合通信操作：每个 Rank 都向其他 Rank 发送一部分张量。例如，四个 Rank 的输入分别为：

- Rank 0：`[0, 1, 2, 3]`；
- Rank 1：`[4, 5, 6, 7]`；
- Rank 2：`[8, 9, 10, 11]`；
- Rank 3：`[12, 13, 14, 15]`。

每个 Rank 按位置发送数据后，输出变为：

- Rank 0：`[0, 4, 8, 12]`；
- Rank 1：`[1, 5, 9, 13]`；
- Rank 2：`[2, 6, 10, 14]`；
- Rank 3：`[3, 7, 11, 15]`。

All-to-all 对混合专家模型很有用：每个 Rank 只持有一部分数据和一部分专家，需要通过通信将数据路由到对应专家。对于均衡切分，All-to-all 看起来类似转置；它也可以处理不均衡切分，但实际中应尽量保持各部分大小均衡。

```python
# Input
rank0 = tensor([0., 1, 2, 3])      # send  0 to rank 0,  1 to rank 1,  2 to rank 2,  3 to rank 3
rank1 = tensor([4., 5, 6, 7])      # send  4 to rank 0,  5 to rank 1,  6 to rank 2,  7 to rank 3
rank2 = tensor([8., 9, 10, 11])    # send  8 to rank 0,  9 to rank 1, 10 to rank 2, 11 to rank 3
rank3 = tensor([12., 13, 14, 15])  # send 12 to rank 0, 13 to rank 1, 14 to rank 2, 15 to rank 3

# Output
rank0 = tensor([0, 4, 8, 12])
rank1 = tensor([1, 5, 9, 13])
rank2 = tensor([2, 6, 10, 14])
rank3 = tensor([3, 7, 11, 15])
```

### 术语记忆

- Reduce 表示执行某种满足结合律和交换律的操作，例如求和、取最小值或取最大值；
- Scatter 是 Gather 的逆操作；
- All 表示目标是所有设备。

## 3. GPU 互连硬件

### 3.1. 传统硬件

<figure>
  <img src="gpu-node-classic.webp" alt="传统多节点 GPU 互连拓扑">
  <figcaption>传统硬件拓扑：同一服务器内的 GPU 通过 PCIe 连接，不同服务器通过 Ethernet 连接。</figcaption>
</figure>

- 同一节点内的 GPU 通过外围组件互连（Peripheral Component Interconnect Express，PCIe）总线通信。以 PCIe 7.0 的 16 条通道为例，带宽约为 242 GB/s；
- 不同节点内的 GPU 通过以太网（Ethernet）通信，带宽约为 200 MB/s。

### 3.2. 现代硬件（数据中心）

<figure>
  <img src="gpu-node-overview.png" alt="多 GPU 节点的层级结构">
  <figcaption>多 GPU 节点的层级结构：GPU 内部包含 SM、L2 和 HBM，GPU 之间通过 NVLink / NVSwitch 连接，节点之间通过 InfiniBand / Ethernet 连接。</figcaption>
</figure>

上图展示了数据中心中的典型层级：

- 每个节点（node）通常有 8 个 GPU，通过 NVLink 连接到 NVSwitch；
  - B200 的 NVLink 5.0 带宽为 1.8 TB/s
  - 高带宽内存（High Bandwidth Memory，HBM）的带宽约为 8 TB/s；
- 一个 Pod 通常有 256 个节点，通过 InfiniBand 互连：
  - PCIe → 主机通道适配器（Host Channel Adapter，HCA）/ InfiniBand 网卡 → InfiniBand 线缆，带宽约为 0.05 TB/s；
- 集群或数据中心中的多个 Pod 通过以太网连接。
  - 通信路径为 PCIe → CPU。

#### 3.2.1. 绕过 CPU

- 标准以太网通信需要经过 CPU：将数据复制到内核套接字缓冲区、构造 TCP 数据包，再复制到网卡的环形缓冲区；
- 远程直接内存访问（Remote Direct Memory Access，RDMA）允许一个 GPU 直接读写另一个 GPU 的内存，无需 CPU 参与；
- InfiniBand 支持 RDMA，而标准以太网不支持。

#### 3.2.2. 新进展

- GB200/GB300 NVL72：每个托盘有 8 个 GPU，每个机架有 9 个托盘，因此 72 个 GPU 位于同一个 NVLink 域中；
- 融合以太网远程直接内存访问（RDMA over Converged Ethernet，RoCE）让以太网也能绕过 CPU。它与 InfiniBand 类似，但成本更低、能力也较弱；Meta 正在使用这种方案。

### 3.3. NVIDIA 集体通信库（NVIDIA Collective Communication Library，NCCL）

NCCL 会将集合通信操作转换为在 GPU 之间传输的底层数据包，并负责：

- 探测硬件拓扑，例如节点数、交换机数量、NVLink 和 PCIe 的连接情况；
- 优化 GPU 之间的通信路径；
- 启动 GPU 内核来发送和接收数据。

## 4. PyTorch 分布式库（`torch.distributed`）

PyTorch 分布式库（`torch.distributed`）为集合通信提供了统一接口，例如 `all_gather_into_tensor`。它支持针对不同硬件的多种后端：
- Gloo 用于 CPU；
- NCCL 用于 GPU；

还提供完全分片数据并行（Fully Sharded Data Parallel，FSDP）等高层算法，本课程暂不使用。

### All-reduce、Reduce-scatter 与 All-gather

cs336 lecture 7 通过 `setup` 初始化进程组，通过 `cleanup` 销毁进程组；`spawn` 为每个 Rank 启动一个进程。以下代码保留课程源码；其中 `DisableDistributed` 是源码中用于生成可执行讲义追踪结果的上下文管理器。

```python
def setup(rank: int, world_size: int):
    """Initializes the distributed environment (called at start of process)."""
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Cleans up the distributed environment (called at end of process)."""
    torch.distributed.destroy_process_group()


def spawn(func: Callable, world_size: int, *args, **kwargs):
    """
    Launches `world_size` processes that each calls `func` on world_size, args, kwargs.
    Note: if we are being traced (inside edtrace), we just run the function directly without multiprocessing and disable distributed functions.
    """
    # Note: assume kwargs are in the same order as what main needs
    if not sys.gettrace():
        # This is the normal code path for multiprocessing
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)
    else:
        # If we're being traced (inside edtrace), just run the function directly.
        with DisableDistributed():
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)
```

下面是课程中的通信示例。函数会由四个进程异步执行，`rank` 取值为 0 到 3。

```python
def collective_operations_main(rank: int, world_size: int):
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
    setup(rank, world_size)

    # All-reduce (dist = torch.distributed)
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)

    data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank  # Both input and output

    print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce]: {data}", flush=True)

    # Reduce-scatter
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=cuda_if_available(rank)) + rank  # Input
    output = torch.empty(1, device=cuda_if_available(rank))  # Allocate output

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    # All-gather
    dist.barrier()

    input = output  # Input is the output of reduce-scatter
    output = torch.empty(world_size, device=cuda_if_available(rank))  # Allocate output

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    cleanup()
```

这段代码先直接执行 All-reduce；随后执行 Reduce-scatter，再把其输出作为 All-gather 的输入。最终可以验证：All-reduce = Reduce-scatter + All-gather。

## 5. 通信性能基准测试（Benchmarking）

分布式通信的基准测试要测量集合通信真正完成后的耗时，而不是只测量 CPU 发起调用的时间。cs336 lecture 7 采用的基本流程是：分配输入输出张量 → 预热一次 → GPU 同步 → 所有 Rank 对齐 → 计时执行通信 → 再次同步和对齐 → 计算有效带宽。

- 预热（Warmup）可以避免首次初始化带来的额外开销；
- `torch.cuda.synchronize()` 等待 GPU 内核完成，否则 CPU 计时会低估通信时间；
- `dist.barrier()` 让各 Rank 在相同位置开始或结束，避免单个慢 Rank 被忽略；
- 有效带宽（Effective Bandwidth）衡量实际传输量除以总耗时，并不等于硬件链路的峰值带宽。

cs336 lecture 7 设置 `num_elements = 100 * 1024**2 = 104857600`。在 `float32` 下，All-reduce 的单个张量约为 400 MiB；Reduce-scatter 的输入形状为 `[world_size, num_elements]`，四卡时约为 1.56 GiB。显存不足时应先减小 `num_elements`。

```python
def benchmarking():
    # All-reduce
    spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)

    # Reduce-scatter
    spawn(reduce_scatter, world_size=4, num_elements=100 * 1024**2)


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create tensor
    data = torch.randn(num_elements, device=cuda_if_available(rank))

    # Warmup
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here

    # Perform all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    size_bytes = data.element_size() * data.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send + receive, world_size-1 steps in all-reduce
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create input and outputs
    input = torch.randn(world_size, num_elements, device=cuda_if_available(rank))  # Each rank has a matrix
    output = torch.empty(num_elements, device=cuda_if_available(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here

    # Perform reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # How much data in the input
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent (no 2x here)
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()
```

设 `W = world_size`，`T = duration`。All-reduce 的有效带宽为：

\[
B_{\text{all-reduce}} = \frac{\text{size\_bytes} \times 2 \times (W - 1)}{W \times T}
\]

其中 `2` 表示发送和接收两个方向，`W - 1` 表示每个 Rank 要与其余 Rank 交换数据。

Reduce-scatter 也有有效带宽公式：

\[
B_{\text{reduce-scatter}} = \frac{\text{data\_bytes} \times (W - 1)}{W \times T}
\]

这里的 `data_bytes` 是每个 Rank 输入张量的总字节数。课程代码中，输入形状为 `[W, num_elements]`，因此 `data_bytes = W × size_bytes`，上式可化简为：

\[
B_{\text{reduce-scatter}} = \frac{\text{size\_bytes} \times (W - 1)}{T}
\]

Reduce-scatter 没有 All-reduce 中的 `2`，因为它只执行归约和分发；All-reduce 还需要额外执行一次 All-gather。

实际测试时，应重复多次并取平均值或中位数，再比较不同消息大小、GPU 数量和互连拓扑的结果。cs336 lecture 7 代码用于展示计时与带宽计算的基本方法。

```text
[all_reduce] Rank 0: all_reduce(world_size=4, num_elements=104857600) took 1.60ms
[all_reduce] Rank 2: all_reduce(world_size=4, num_elements=104857600) took 1.38ms
[all_reduce] Rank 1: all_reduce(world_size=4, num_elements=104857600) took 1.50ms
[all_reduce] Rank 3: all_reduce(world_size=4, num_elements=104857600) took 1.38ms
[all_reduce] Rank 1: all_reduce measured bandwidth = 390 GB/s
[all_reduce] Rank 2: all_reduce measured bandwidth = 426 GB/s
[all_reduce] Rank 0: all_reduce measured bandwidth = 366 GB/s
[all_reduce] Rank 3: all_reduce measured bandwidth = 425 GB/s
[reduce_scatter] Rank 0: reduce_scatter(world_size=4, num_elements=104857600) took 2.61ms
[reduce_scatter] Rank 1: reduce_scatter(world_size=4, num_elements=104857600) took 2.47ms
[reduce_scatter] Rank 2: reduce_scatter(world_size=4, num_elements=104857600) took 2.39ms
[reduce_scatter] Rank 3: reduce_scatter(world_size=4, num_elements=104857600) took 2.39ms
[reduce_scatter] Rank 1: reduce_scatter measured bandwidth = 475 GB/s
[reduce_scatter] Rank 0: reduce_scatter measured bandwidth = 450 GB/s
[reduce_scatter] Rank 2: reduce_scatter measured bandwidth = 490 GB/s
[reduce_scatter] Rank 3: reduce_scatter measured bandwidth = 490 GB/s
```

## 6. 分布式训练

### 6.1. 数据并行（Data Parallelism，DP）

数据并行的分片策略是：每个 Rank 获得一部分数据。模型的每一层在所有 Rank 上完整复制，只有 Batch 被切分。

<figure>
  <img src="data-parallelism.png" alt="数据并行示意图" width="50%">
  <figcaption>数据并行：所有 Rank 复制完整模型，但分别处理不同的数据切片。</figcaption>
</figure>

cs336 lecture 7 用一个 Batch 为 128、特征维度为 1024 的样例，启动 4 个 Rank、训练 4 层多层感知机（Multilayer Perceptron，MLP）一个 step：

```python
def data_parallelism():
    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data
```

每个 Rank 先切出自己的数据，再创建完整模型参数和各自的 AdamW 优化器状态。前向、反向都在本地进行；与单卡训练相比，唯一的额外步骤是对每一层的梯度执行 All-reduce 平均：

```python
def data_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # Get the slice of data for this rank
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size) # Each rank gets a portion of the batch
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    data = data[start_index:end_index].to(cuda_if_available(rank))

    # Each rank has all parameters and its own optimizer state.
    params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()

        # Backward pass
        loss.backward()

        # The only difference from standard training: synchronize gradients.
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[layer]) for layer in range(num_layers)]}", flush=True)

    cleanup()
```

注意：各 Rank 的损失不同，因为它们处理的是本地数据；梯度经过 All-reduce 后相同，因此更新后的参数在各 Rank 上保持一致。

### 6.2. 张量并行（Tensor Parallelism，TP）

张量并行的分片策略是：每个 Rank 持有**每一层的一部分参数**，并在层与层之间传输完整数据或激活值。它沿模型宽度切分，而不是沿 Batch 切分。

<figure>
  <img src="tensor-parallelism.png" alt="张量并行示意图" width="50%">
  <figcaption>张量并行：所有 Rank 保留完整数据，但每层参数沿宽度维度切分。</figcaption>
</figure>

cs336 lecture 7 用 4 个 Rank 将每层宽度均分为四份。所有 Rank 都持有形状为 `batch_size × num_dim` 的输入，每个 Rank 只计算 `batch_size × local_num_dim` 的局部激活值；随后通过 All-gather 收集并拼接回完整激活值：

```python
def tensor_parallelism():
    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int):
    setup(rank, world_size)

    # All ranks get the data (batch_size x num_dim)
    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_num_dim = int_divide(num_dim, world_size)

    # Each rank gets 1 / world_size of the parameters in every layer.
    params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]

    # Forward pass
    x = data
    for layer in range(num_layers):
        # Compute activations (batch_size x local_num_dim)
        x = x @ params[layer]
        x = F.gelu(x)

        # Allocate memory for activations from every Rank.
        activations = [
            torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank))
            for _ in range(world_size)
        ]

        # Send activations via all-gather, then concatenate them.
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)

    # Backward pass: homework exercise
    cleanup()
```

因此，张量并行减少了每个 GPU 的参数量，但每层前向传播都需要一次 All-gather。这种频繁通信使它更适合 NVLink 等高速互连环境。

### 6.3. 流水线并行（Pipeline Parallelism，PP）

流水线并行的分片策略是：每个 Rank 持有连续的一部分层，并在相邻 Rank 之间传输完整数据或激活值。它沿模型深度切分。

<figure>
  <img src="pipeline-parallelism.png" alt="流水线并行示意图" width="50%">
  <figcaption>流水线并行：模型层沿深度切分，相邻 Rank 之间传递激活值。</figcaption>
</figure>

cs336 lecture 7 示例将 4 层模型切到 2 个 Rank 上，并把一个 Batch 切成 4 个微批次（Micro-batch）。微批次可以减少流水线气泡（Pipeline Bubble）：不同 Rank 同时处理处于不同阶段的微批次。

```python
def pipeline_parallelism():
    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # Use all the data.
    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)

    # Split up layers: each Rank gets a consecutive subset.
    local_num_layers = int_divide(num_layers, world_size)
    local_params = [get_init_params(num_dim, num_dim, rank) for layer in range(local_num_layers)]

    # Break up into micro batches to minimize the bubble.
    micro_batch_size = int_divide(batch_size, num_micro_batches)
    if rank == 0:
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        micro_batches = [
            torch.empty(micro_batch_size, num_dim, device=cuda_if_available(rank))
            for _ in range(num_micro_batches)
        ]

    # Forward pass
    for x in micro_batches:
        # Get activations from the previous Rank.
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Compute layers assigned to this Rank.
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Send activations to the next Rank.
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    # Backward pass: homework exercise
    cleanup()
```



---

## 参考文献

[1] Stanford CS336, "Lecture 7: Parallelism." [Online]. Available: https://cs336.stanford.edu/lectures/?trace=lecture_07.

[2] NVIDIA, "How to reason about collective operations." [Online]. Available: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce.

[3] Stas Bekman, "Sample benchmarking code." [Online]. Available: https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py.
