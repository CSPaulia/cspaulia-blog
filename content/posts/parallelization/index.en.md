---
title: "Parallelization"
date: 2026-07-18T10:00:00+08:00
series:
  main: "Large Language Model"
  subseries: "Systems and Hardware"
draft: false
categories: ["大语言模型", "系统"]
tags: ["GPU", "CUDA", "并行计算", "训练"]
author: "CSPaulia"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "CS336 Lecture 7 & 8 study notes."
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
cover:
    image: "cover.png"
    alt: "cover"
    caption: "cover"
    relative: true
    hidden: false
    hiddenInList: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

The previous lecture covered parallelism within a single GPU. This lecture moves on to parallelism across multiple GPUs.

<figure>
  <img src="gpu-node-overview.png" alt="Hierarchy of a multi-GPU node">
  <figcaption>Hierarchy of a multi-GPU node: a GPU contains SMs, L2, and HBM; GPUs connect through NVLink / NVSwitch, while nodes connect through InfiniBand / Ethernet.</figcaption>
</figure>

## 1. Why Multiple GPUs Are Needed

From fastest to slowest, a typical hardware hierarchy is:

- One node, one GPU: L1 cache or shared memory, the fastest level;
- One node, one GPU: High Bandwidth Memory (HBM);
- One node, multiple GPUs: connected through NVLink or NVSwitch;
- Multiple nodes, multiple GPUs: connected through InfiniBand or Ethernet, the slowest level.

The farther apart two levels are, the slower their data transfer is usually and the higher the communication cost becomes. Within one GPU, operator fusion and tiling reduce memory accesses. Across GPUs or nodes, replication and sharding reduce inter-GPU communication.

There are two main reasons to use multiple GPUs:

1. The parameters, optimizer state, gradients, and activations required for training do not fit on one GPU.
2. More GPUs provide more floating-point operations (FLOPs), allowing training to finish faster.

Scaling a single GPU is constrained by both compute and memory. Even though supercomputers have reached the exaFLOPS scale, the compute capability and memory capacity of an individual GPU cannot keep pace with continued model growth. The goal of multi-GPU and multi-node parallelism is therefore to distribute a model's memory and compute requirements across devices and nodes.

## 2. Collective Communication Primitives

Collective operations are conceptual primitives used in distributed programming and are a classic idea from parallel-programming literature of the 1980s. “Collective” means describing a common communication pattern across devices rather than manually managing individual point-to-point transfers. This usually improves performance and gives the underlying implementation more opportunities to optimize communication.

<figure>
  <img src="ranks.png" alt="Illustration of ranks">
  <figcaption>Each device participating in communication has a rank; their total number is the world size.</figcaption>
</figure>

### Basic Setup

In a distributed environment, each device or GPU is called a rank, such as Rank 0, Rank 1, Rank 2, and Rank 3. The total number of devices is called the world size; here, the world size is 4.

The code below follows the original lecture and uses `tensor(...)` to show the tensor held by each rank before and after communication. It illustrates data flow rather than providing complete distributed code.

Collective operations fall into three groups:

- Broadcast, Scatter, Gather, and Reduce are basic operations;
- All-gather, Reduce-scatter, and All-reduce are the primary operations used in practice;
- All-to-all is often used for Mixture of Experts (MoE) models.

<figure>
  <img src="collective-operations-overview.png" alt="Overview of collective communication primitives">
  <figcaption>Collective communication primitives: the data flow of All-reduce, Broadcast, Reduce, All-gather, and Reduce-scatter.</figcaption>
</figure>

### Broadcast

Broadcast copies data from Rank 0 to every rank. For example, if Rank 0 starts with `[0, 1, 2, 3]`, every rank receives `[0, 1, 2, 3]` after the operation.

A simple use case is for Rank 0 to load an initial checkpoint and then broadcast it to all ranks.

```python
# Input
rank0 = tensor([0., 1, 2, 3])

# Output
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
```

### Scatter

Scatter splits a tensor on Rank 0 and sends a shard to every rank. For example, `[0, 1, 2, 3]` on Rank 0 is distributed as:

- Rank 0: `[0]`;
- Rank 1: `[1]`;
- Rank 2: `[2]`;
- Rank 3: `[3]`.

Scatter is the foundation for understanding Reduce-scatter.

```python
# Input
rank0 = tensor([0., 1, 2, 3])

# Output
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])
```

### Gather

Gather collects data from every rank onto Rank 0, the inverse of Scatter. For example, if Ranks 0, 1, 2, and 3 hold `[0]`, `[1]`, `[2]`, and `[3]`, respectively, Rank 0 obtains `[0, 1, 2, 3]` after Gather.

Gather is the foundation for understanding All-gather.

```python
# Input
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# Output
rank0 = tensor([0., 1, 2, 3])
```

### Reduce

Reduce collects data from every rank onto Rank 0 and applies an operation, such as sum, minimum, or maximum. If four ranks hold `[0]`, `[1]`, `[2]`, and `[3]`, summation leaves Rank 0 with `[6]`.

Reduce is the foundation for understanding All-reduce.

```python
# Input
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# Output
rank0 = tensor([6.])  # Sum of all ranks (0 + 1 + 2 + 3)
```

### All-gather

All-gather resembles Gather, but sends the collected result to every rank instead of only Rank 0. If each rank holds a parameter shard, All-gather lets every rank obtain the full parameters for the forward pass.

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

### Reduce-scatter

Reduce-scatter performs Reduce along each dimension and then distributes the results among ranks. For example:

- Rank 0 inputs `[0, 1, 2, 3]` and receives `[6]`;
- Rank 1 inputs `[1, 2, 3, 4]` and receives `[10]`;
- Rank 2 inputs `[2, 3, 4, 5]` and receives `[14]`;
- Rank 3 inputs `[3, 4, 5, 6]` and receives `[18]`.

One use case is to sum gradients produced by different data shards after backpropagation while distributing gradient storage among ranks.

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

### All-reduce

```
All-reduce = Reduce-scatter + All-gather
```

All-reduce is equivalent to Reduce-scatter followed by All-gather. For the input in the preceding section, every rank obtains `[6, 10, 14, 18]`.

All-reduce can aggregate gradients generated by different data shards after backpropagation while replicating full parameters. Splitting All-reduce into Reduce-scatter and All-gather also offers more flexibility, for example by supporting the Zero Redundancy Optimizer (ZeRO) and Fully Sharded Data Parallel (FSDP).

<figure>
  <img src="all-reduce-decomposition.png" alt="All-reduce decomposed into Reduce-scatter and All-gather">
  <figcaption>All-reduce can be decomposed into Reduce-scatter and All-gather. Under bandwidth constraints, this decomposition achieves the optimal data-transfer volume.</figcaption>
</figure>

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

### All-to-all

All-to-all is the most general collective communication operation: every rank sends part of its tensor to every other rank. The inputs of four ranks are:

- Rank 0: `[0, 1, 2, 3]`;
- Rank 1: `[4, 5, 6, 7]`;
- Rank 2: `[8, 9, 10, 11]`;
- Rank 3: `[12, 13, 14, 15]`.

After every rank sends data by position, the outputs are:

- Rank 0: `[0, 4, 8, 12]`;
- Rank 1: `[1, 5, 9, 13]`;
- Rank 2: `[2, 6, 10, 14]`;
- Rank 3: `[3, 7, 11, 15]`.

All-to-all is useful for MoE: each rank holds only part of the data and part of the experts, so communication routes data to the appropriate experts. With balanced splits, All-to-all resembles a transpose. It can also handle uneven splits, but the parts should be kept as balanced as possible in practice.

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

### Terminology to Remember

- Reduce applies an operation that is associative and commutative, such as sum, minimum, or maximum;
- Scatter is the inverse of Gather;
- All means that the destination is every device.

## 3. GPU Interconnect Hardware

### 3.1. Traditional Hardware

<figure>
  <img src="gpu-node-classic.webp" alt="Traditional multi-node GPU interconnect topology">
  <figcaption>Traditional hardware topology: GPUs within a server connect through PCIe, while servers connect through Ethernet.</figcaption>
</figure>

- GPUs in the same node communicate over the Peripheral Component Interconnect Express (PCIe) bus. For example, PCIe 7.0 with 16 lanes provides about 242 GB/s of bandwidth;
- GPUs in different nodes communicate through Ethernet, with bandwidth of about 200 MB/s.

### 3.2. Modern Hardware (Data Centers)

<figure>
  <img src="gpu-node-overview.png" alt="Hierarchy of a multi-GPU node">
  <figcaption>Hierarchy of a multi-GPU node: a GPU contains SMs, L2, and HBM; GPUs connect through NVLink / NVSwitch, while nodes connect through InfiniBand / Ethernet.</figcaption>
</figure>

The figure shows a typical data-center hierarchy:

- Each node usually has 8 GPUs connected to an NVSwitch through NVLink;
  - B200 NVLink 5.0 provides 1.8 TB/s of bandwidth;
  - HBM provides about 8 TB/s of bandwidth;
- A pod usually has 256 nodes connected through InfiniBand:
  - PCIe → Host Channel Adapter (HCA) / InfiniBand NIC → InfiniBand cable, with about 0.05 TB/s of bandwidth;
- Multiple pods in a cluster or data center connect through Ethernet.
  - The communication path is PCIe → CPU.

#### 3.2.1. Bypassing the CPU

- Standard Ethernet communication passes through the CPU: data is copied into a kernel socket buffer, packaged as TCP packets, and copied again into the NIC ring buffer;
- Remote Direct Memory Access (RDMA) allows one GPU to directly read or write another GPU's memory without involving the CPU;
- InfiniBand supports RDMA, while standard Ethernet does not.

#### 3.2.2. Recent Developments

- GB200/GB300 NVL72: each tray has 8 GPUs and each rack has 9 trays, placing 72 GPUs in one NVLink domain;
- RDMA over Converged Ethernet (RoCE) lets Ethernet bypass the CPU as well. It is similar to InfiniBand but less capable and less expensive; Meta is using this approach.

### 3.3. NVIDIA Collective Communication Library (NCCL)

NCCL turns collective operations into low-level packets transmitted between GPUs. It is responsible for:

- Discovering the hardware topology, such as the number of nodes and switches and the NVLink and PCIe connections;
- Optimizing communication paths between GPUs;
- Launching GPU kernels to send and receive data.

### 3.4. Network Topologies and Scaling Domains

<figure>
  <img src="tpu-gpu-networking.png" alt="Comparison of a TPU torus mesh and a switched GPU network">
  <figcaption>A TPU torus mesh and a switched GPU-cluster network: they target different communication patterns.</figcaption>
</figure>

#### 3.4.1. TPU Meshes and GPU Tree Networks

TPUs often use a toroidal mesh:

- Each chip connects directly only to neighboring chips, so data travels hop by hop through the mesh;
- This fixed-neighbor wiring is simpler and less expensive;
- For structured communication such as tensor parallelism, frequently exchanged data can be mapped to neighboring chips to make local communication faster.

GPU clusters more commonly use tree or switched networks:

- Devices form a hierarchy through switches and support more flexible **All-to-all** communication within a high-speed domain;
- Compared with a mesh, this hardware and its operation cost more, but it handles **irregular communication** better, such as the expert-parallel pattern used by MoE.

#### 3.4.2. TPUs Are Also Adopting Switched Designs

TPU networking is evolving as well. TPU8i uses an interconnect closer to a tree topology, possibly to better support complex communication such as MoE. TPU8t uses a switched cross-domain network called Virgo. In other words, while retaining the efficiency of meshes for regular communication, TPUs are also adopting switched designs for more complex topologies and traffic patterns.

#### 3.4.4. Boundaries of Scaling Domains

Not every accelerator can be directly interconnected: wiring, power, cost, and network scale all grow quickly. Real systems therefore distinguish a scale-up domain within a node or rack from a scale-out network across domains.

The goal of multi-node scaling is for both the number of parameters a model can hold and its compute capability to grow roughly linearly with the number of GPUs. This relies on simple and efficient collective communication primitives.

## 4. PyTorch Distributed (`torch.distributed`)

PyTorch Distributed (`torch.distributed`) provides a unified interface for collective communication, such as `all_gather_into_tensor`. It supports multiple backends for different hardware:

- Gloo for CPUs;
- NCCL for GPUs.

It also provides higher-level algorithms such as Fully Sharded Data Parallel (FSDP), which are not used in this lecture.

### All-reduce, Reduce-scatter, and All-gather

CS336 Lecture 7 uses `setup` to initialize a process group, `cleanup` to destroy it, and `spawn` to launch one process for each rank. The code below is retained from the lecture source. `DisableDistributed` is a context manager used in the source to generate executable-lecture traces.

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

The following communication example from the lecture runs asynchronously in four processes, with `rank` ranging from 0 to 3.

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

The code first runs All-reduce directly, then runs Reduce-scatter and uses its output as the input to All-gather. It therefore verifies that `All-reduce = Reduce-scatter + All-gather`.

## 5. Communication Performance Benchmarking

Distributed-communication benchmarks must measure the time after a collective actually completes, rather than only the time for the CPU to launch the call. CS336 Lecture 7 uses the following basic sequence: allocate input and output tensors → warm up once → synchronize the GPU → align all ranks → time the collective → synchronize and align again → compute effective bandwidth.

- A warmup avoids overhead from first-time initialization;
- `torch.cuda.synchronize()` waits for GPU kernels to finish; without it, CPU timing underestimates communication time;
- `dist.barrier()` makes all ranks start or finish at the same point, preventing a slow rank from being ignored;
- Effective bandwidth measures actual transferred data divided by total time; it is not the peak bandwidth of the hardware link.

CS336 Lecture 7 sets `num_elements = 100 * 1024**2 = 104857600`. With `float32`, a single All-reduce tensor is about 400 MiB; the Reduce-scatter input has shape `[world_size, num_elements]` and occupies about 1.56 GiB on four GPUs. Reduce `num_elements` first if memory is insufficient.

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

Let `W = world_size` and `T = duration`. The effective bandwidth of All-reduce is:

\[
B_{\text{all-reduce}} = \frac{\text{size\_bytes} \times 2 \times (W - 1)}{W \times T}
\]

Here, `2` represents the send and receive directions, and `W - 1` means that every rank exchanges data with all other ranks.

Reduce-scatter also has an effective-bandwidth formula:

\[
B_{\text{reduce-scatter}} = \frac{\text{data\_bytes} \times (W - 1)}{W \times T}
\]

Here, `data_bytes` is the total number of bytes in each rank's input tensor. In the lecture code, the input shape is `[W, num_elements]`, so `data_bytes = W × size_bytes`; the formula simplifies to:

\[
B_{\text{reduce-scatter}} = \frac{\text{size\_bytes} \times (W - 1)}{T}
\]

Reduce-scatter does not have the `2` in All-reduce because it performs only reduction and distribution; All-reduce additionally performs All-gather.

In a real benchmark, repeat measurements and use an average or median, then compare results for different message sizes, GPU counts, and interconnect topologies. The CS336 Lecture 7 code demonstrates the basic timing and bandwidth calculations.

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

## 6. Distributed Training

### 6.1. Data Parallelism (DP)

The sharding strategy of data parallelism is to give each rank a portion of the data. Every model layer is fully replicated on all ranks; only the batch is split.

<figure>
  <img src="data-parallelism.png" alt="Illustration of data parallelism" width="50%">
  <figcaption>Data parallelism: every rank replicates the full model but processes a different data shard.</figcaption>
</figure>

CS336 Lecture 7 uses an example with a batch size of 128 and feature dimension of 1024. It launches 4 ranks and trains a 4-layer Multilayer Perceptron (MLP) for one step:

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

Each rank first takes its data slice, then creates full model parameters and its own AdamW optimizer state. The forward and backward passes are local. Compared with single-GPU training, the only extra step is to average the gradient of every layer with All-reduce:

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

Note that the losses differ across ranks because they process local data. Their gradients are identical after All-reduce, so their updated parameters remain consistent.

### 6.2. Zero Redundancy Optimizer (ZeRO)

> ZeRO can be understood as a data-parallel optimization that shards training state. Its purpose is to **reduce the memory use of data parallelism**.

#### Memory Cost of Naive Data Parallelism

Naive data parallelism does not scale model-state memory: every rank keeps a full copy of the parameters, gradients, and optimizer state. With mixed-precision Adam, a parameter typically corresponds to five kinds of training state:

- BF16/FP16 model parameter: 2 bytes;
- BF16/FP16 gradient: 2 bytes;
- FP32 master weight used to accumulate updates: 4 bytes;
- Adam first moment: 4 bytes;
- Adam second moment: 4 bytes.

Thus, each parameter needs about 16 bytes across five related states; the last three are collectively called optimizer state. Adding GPUs does not reduce per-GPU memory because these states are still fully replicated. This is the redundancy that ZeRO eliminates.

The intuition behind ZeRO is that data parallelism still lets every rank compute full gradients for its local batch, but no longer makes every rank retain expensive, redundant training state. It distributes state among ranks and rewrites the original All-reduce as Reduce-scatter plus All-gather.

<figure>
  <img src="zero-state-sharding.png" alt="The three ZeRO stages shard parameters, gradients, and optimizer state differently">
  <figcaption>ZeRO state sharding: blue represents parameters, orange gradients, and green optimizer state. Higher stages progressively reduce the state stored on each GPU.</figcaption>
</figure>

Let a model have \(\Psi\) parameters and use \(N\) GPUs. Parameters and gradients each take 2 bytes, while optimizer-related state takes \(K\) bytes. Model-state memory per GPU can be summarized as:

| Scheme | Parameters | Gradients | Optimizer state | Model-state memory per GPU |
| --- | --- | --- | --- | --- |
| Naive data parallelism | Full replica | Full replica | Full replica | \((2 + 2 + K) \times \Psi\) |
| ZeRO Stage 1 | Full replica | Full replica | Sharded | \((2 + 2 + K / N) \times \Psi\) |
| ZeRO Stage 2 | Full replica | Sharded | Sharded | \((2 + (2 + K) / N) \times \Psi\) |
| ZeRO Stage 3 / FSDP | Sharded | Sharded | Sharded | \(((2 + 2 + K) / N) \times \Psi\) |

#### 6.2.1. Stage 1: Shard Optimizer State

- Stage 1 shards only optimizer state, including FP32 master weights and Adam's first and second moments;
- Every rank still has full parameters and full gradients, but updates only its assigned parameter shard.

The difference between ordinary data parallelism and ZeRO Stage 1 in one training step is:

| Step | Ordinary data parallelism | ZeRO Stage 1 |
| --- | --- | --- |
| 1. Backpropagation | Every rank computes full gradients on its local data shard. | Every rank computes full gradients on its local data shard. |
| 2. Synchronize gradients | All-reduce full gradients; every rank obtains **full global gradients**. | Reduce-scatter gradients; every rank obtains only the global gradient for its assigned parameter shard. |
| 3. Update parameters | Every rank has full optimizer state and updates **all parameters**. | Every rank uses local optimizer state to update **its assigned parameter shard**. |
| 4. Prepare the next step | Ranks already hold identical full parameters; no extra communication is needed. | All-gather the updated parameter shards so every rank again has full parameters. |

> Note: Reduce-scatter shard boundaries must align with optimizer-state shard boundaries. Frameworks flatten all parameters in a fixed order and split them into `N` parts, such as `P₀, P₁, ..., Pₙ₋₁`. Rank `i` persistently stores the master weights and Adam state for `Pᵢ`. After backpropagation, Reduce-scatter reduces gradients using the same boundaries, so Rank `i` receives exactly the global gradient for `Pᵢ` and can update it directly.

| Communication phase | Ordinary data parallelism | ZeRO Stage 1 |
| --- | --- | --- |
| Operation and data | All-reduce (= Reduce-scatter + All-gather) produces **global gradients**. | Reduce-scatter produces the global gradient for each rank's parameter shard; All-gather collects updated parameter shards into **full model parameters**. |
| Total transfer volume | About `2 × parameter size`. | About `2 × parameter size`. |
| Result | Every rank retains full gradients, full parameters, and full optimizer state. | Every rank retains full parameters, but only `1 / N` of optimizer state. |

> **Why is the transfer volume `2 × parameter size`?**
>
> Let the full parameter tensor, equivalently the full gradient, be \(P\) bytes.
>
> For ordinary data parallelism, gradient All-reduce can be decomposed into one Reduce-scatter and one All-gather:
>
> - Reduce-scatter reduces gradients and distributes gradient shards, moving about \(P\);
> - All-gather collects all gradient shards so every rank receives the full global gradient, moving about \(P\).
>
> Thus, gradient synchronization in ordinary data parallelism transfers about `2 × P` per step.
>
> For ZeRO Stage 1:
>
> - Reduce-scatter reduces gradients and gives every rank the global gradient for its parameter shard, moving about \(P\);
> - After every rank updates its local parameter shard, All-gather collects the updated shards so every rank again receives the full model parameters, moving about \(P\).
>
> Therefore, ZeRO Stage 1 also transfers about `2 × P` in total. The difference is that ordinary data parallelism All-gathers **gradients**, whereas ZeRO Stage 1 All-gathers **updated parameters**.
>
> More precisely, with \(N\) ranks, ring collectives transfer approximately
>
> \[
> 2 \times \frac{N - 1}{N} \times P
> \]
>
> per rank. As \(N\) grows, this approaches `2 × P`.

#### 6.2.2. Stage 2: Continue Sharding Gradients

Stage 2 continues from Stage 1 by sharding gradients. Parameters remain fully replicated on every rank, while gradients and optimizer state are retained only as `1 / N` shards. The key is not to avoid computing full gradients, but to **avoid retaining full gradients**.

Compared with Stage 1, the key change is to embed gradient communication in layer-by-layer backpropagation instead of handling full gradients only after all backpropagation finishes.

| Step | ZeRO Stage 1 | ZeRO Stage 2 |
| --- | --- | --- |
| 1. Backpropagation | Computes local gradients layer by layer and accumulates all layers; a full gradient buffer exists at the end. | Computes local gradients layer by layer; as soon as one layer is ready, it performs Reduce-scatter and does not accumulate a full gradient buffer. |
| 2. Synchronize gradients | Performs one Reduce-scatter on full gradients after backpropagation. | Embeds Reduce-scatter in backpropagation and runs it layer by layer. |
| 3. Release gradients | The full gradient buffer is retained until synchronization completes. | Releases a layer's full local gradient immediately after Reduce-scatter, retaining only its local gradient shard. |
| 4. Update parameters | Every rank uses its local gradient shard and optimizer state to update its assigned parameter shard. | Same. |
| 5. Prepare the next step | All-gathers updated parameter shards so every rank again has full parameters. | Same. |

| Communication phase | ZeRO Stage 1 | ZeRO Stage 2 |
| --- | --- | --- |
| Reduce-scatter | Reduces full gradients after backpropagation; every rank obtains the global gradient for its parameter shard. | Reduces a layer as soon as its gradient is ready; every rank obtains the corresponding shard. |
| All-gather | After updating parameter shards, collects them into full model parameters. | Same. |
| Total transfer volume | About `2 × parameter size`. | About `2 × parameter size`. |
| Gradient state | Requires a full gradient buffer during backpropagation. | Retains only local gradient shards long term. |

This also increases implementation complexity: gradient, optimizer-state, and parameter shards must keep the same boundaries, and a layer can be released only after it is no longer used by the backward graph. Implementations also usually overlap communication with backpropagation of later layers to reduce waiting.

#### 6.2.3. Stage 3: Fully Sharded Data Parallel

Stage 3, also called Fully Sharded Data Parallel (FSDP):

- Further shards **model parameters** on top of Stage 2, so parameters, gradients, and optimizer state each retain only `1 / N` on every rank;
- A rank no longer holds the full model when it is not computing. This is why Stage 3 can scale model-state memory roughly with the number of GPUs.

| Comparison | ZeRO Stage 2 | ZeRO Stage 3 / FSDP |
| --- | --- | --- |
| Static parameters | Every rank stores full parameters. | Every rank stores only parameter shards. |
| Before computing a unit | No parameter collection is needed. | All-gather the complete parameters of the current FSDP unit on demand. |
| After computing a unit | Full parameters remain resident. | Immediately release temporary full parameters and retain only shards. |
| After updating parameters | All-gather updated parameter shards to restore full parameters. | Keep updated parameter shards and All-gather again only when needed. |

#### 6.2.3.1. Complete Lifecycle of an FSDP Unit

An FSDP unit can be understood as a wrapped module, such as a Transformer block.

<figure>
  <img src="fsdp-operation-flow.png" alt="An FSDP unit aggregates parameters, computes, reduces gradients, and releases memory during forward and backward passes">
  <figcaption>Lifecycle of an FSDP unit: parameters are gathered once for each forward and backward pass, gradients are reduced after backpropagation, and temporary full parameters are released as soon as they are no longer needed.</figcaption>
</figure>

For unit `i`, the operations in one training step are:

1. **Load local model shards**: every rank already stores its shard of unit `i`. If CPU offload is enabled, first transfer this shard from CPU to GPU; this is not GPU-to-GPU collective communication.
2. **Forward pass**: All-gather the unit's full parameters temporarily; after computing the forward pass on local data, release the full parameters immediately, keeping only activations needed for the backward pass.
3. **Backward pass**: when backpropagation reaches the unit, All-gather again because full parameters are still needed to compute its gradients. After local backpropagation, use Reduce-scatter to reduce global gradients and hand them to the rank responsible for the parameter shard.
4. **Release and update**: release the full local gradients and temporary full parameters, retaining gradient shards. After all units finish backpropagation, each rank updates its parameter shard using its gradient shard and optimizer state.

Thus, every parameter participates in two All-gathers per step, one for the forward pass and one for the backward pass; its gradient participates in one Reduce-scatter. Total communication is about `3 × parameter size`. Unlike Stage 2, Stage 3 does not collect updated parameters into a full model at the end; the next forward pass gathers parameters for the current unit on demand.

#### 6.2.3.2. Incremental Communication and Compute Overlap

Page 26 expands this process into a timeline. Blue and green blocks are forward and backward computation, red blocks are All-gather (AG), purple blocks are Reduce-scatter (RS), and yellow blocks indicate parameter release. The CPU timeline schedules the work, while the GPU compute and communication streams appear below it.

<figure>
  <img src="fsdp-overlap-timeline.png" alt="FSDP timeline showing All-gather and Reduce-scatter overlapped with forward and backward computation">
  <figcaption>FSDP timeline: AG and RS in the communication stream interleave with forward and backward computation in the compute stream; parameters are released as soon as their unit finishes computing.</figcaption>
</figure>

For forward units `0`, `1`, and `2`, `FWD0` can begin after `AG0` finishes. While `FWD0` runs, the communication stream prefetches unit `1` parameters with `AG1`. Thus, when `FWD0` ends, `FWD1` usually does not need to wait for parameter communication. The backward pass is similar: after one unit's backward computation completes, its gradient is sent through `RS` while the compute stream proceeds with an adjacent unit.

Without overlap, a unit waits for approximately “communication time + compute time.” In the steady state, communication and computation run in parallel, so the unit time is closer to the slower of the two:

\[
T_{\text{steady}} \approx \max(T_{\text{compute}}, T_{\text{communication}})
\]

This does not make communication free. The first unit's All-gather and the last unit's Reduce-scatter are difficult to hide, and collectives in the communication stream queue behind one another. Communication can still become the bottleneck when a model is too small, FSDP units are too fine-grained, network latency is high, or CPU offload is enabled.

#### 6.2.4. Capacity Gains and Remaining Limits

In an example that counts only model state, 8 A100 GPUs with 80 GB each use BF16 for everything except FP32 master weights, for about 12 bytes per parameter. The approximate maximum model sizes are:

| Scheme | Maximum parameter count |
| --- | --- |
| Naive data parallelism | 6.66B |
| ZeRO Stage 1 | 16B |
| ZeRO Stage 2 | 24.62B |
| ZeRO Stage 3 / FSDP | 53.33B |

This estimates only state memory and excludes activations, temporary tensors, and communication buffers. In particular, Stages 1 and 2 still require every GPU to hold full parameters. Although Stage 3 shards parameters, it does not automatically reduce activation memory.

Data-parallel compute scaling is still limited by batch size: the number of devices must be smaller than the batch, and gains diminish as the batch grows. ZeRO removes model-state redundancy, not every training bottleneck. When parameters or activations still do not fit, combine it with tensor parallelism, pipeline parallelism, or recomputation.

### 6.3. Tensor Parallelism (TP)

The sharding strategy of tensor parallelism is for every rank to hold **part of the parameters of each layer** and exchange full data or activations between layers. It splits along model width rather than the batch dimension.

<figure>
  <img src="tensor-parallelism.png" alt="Illustration of tensor parallelism" width="50%">
  <figcaption>Tensor parallelism: every rank retains full data, while each layer's parameters are split along the width dimension.</figcaption>
</figure>

The CS336 Lecture 7 example splits each layer's width evenly across 4 ranks. Every rank holds an input of shape `batch_size × num_dim` and computes only a local activation of shape `batch_size × local_num_dim`; it then uses All-gather to collect and concatenate full activations:

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

Tensor parallelism reduces parameters per GPU, but every forward pass through a layer needs one All-gather. This frequent communication makes it more suitable for high-speed interconnects such as NVLink.

#### 6.3.1. Column Parallelism and Row Parallelism

The preceding example performs All-gather after every layer to make sharding intuitive and restore full activations. Actual Transformers pair adjacent linear layers: first split by columns, then by rows, avoiding full-activation collection between the two layers.

Let the two linear layers of an MLP be:

\[
Y = \operatorname{GeLU}(XA), \qquad Z = \operatorname{Dropout}(YB)
\]

Split the first weight matrix by columns and the second by rows:

\[
A = [A_1, A_2, \ldots, A_t], \qquad
B =
\begin{bmatrix}
B_1 \\
B_2 \\
\vdots \\
B_t
\end{bmatrix}
\]

Rank \(i\) independently computes the local activation \(Y_i = \operatorname{GeLU}(XA_i)\). Because \(Y_i\) depends only on \(A_i\), no communication is needed here. It then computes the local partial sum \(Z_i = Y_iB_i\) and uses All-reduce to obtain \(Z = \sum_i Z_i\). The forward pass therefore synchronizes only at the end of the row-parallel layer.

Backpropagation is the reverse: input gradients for a row-parallel layer naturally correspond to local shards and need not be collected; a column-parallel layer must sum every rank's contribution to the same input, so it uses All-reduce at its boundary. These communication locations are often marked `f` and `g`: during the forward pass, `f` is identity and `g` is All-reduce; during the backward pass, they swap roles.

<figure>
  <img src="tensor-parallel-transformer-block.png" alt="Column and row parallelism in an MLP and self-attention" width="100%">
  <figcaption>The MLP first splits the up-projection by columns and then the down-projection by rows; self-attention splits Q, K, and V by heads and merges at the output projection.</figcaption>
</figure>

In a Transformer block, a common arrangement is:

- Column parallel: Q, K, and V projections, plus the MLP up-projection;
- Row parallel: the attention output projection and the MLP down-projection;
- Replicated: small or unsuitable-to-shard operations such as LayerNorm and the router.

Attention heads can be computed independently, so assigning different heads to ranks is natural. The output projection is where the results must actually be merged. The point of this design is not to remove communication entirely, but to have one All-reduce cover a larger span of computation.

#### 6.3.2. When to Use Tensor Parallelism

Tensor parallelism performs collective communication in every Transformer block, so it is usually used within one node. GPU servers often connect about 8 GPUs through NVLink or NVSwitch. This is not a hard TP limit, but latency and bandwidth worsen across nodes, so communication can easily outweigh the compute benefit of sharding.

<figure>
  <img src="tensor-parallel-scaling.png" alt="Tensor parallel degree versus per-GPU throughput for a 3B model" width="70%">
  <figcaption>Lecture 8 measurements for a 3B model: compared with TP 2, TP 4 and 8 reduce per-GPU throughput by about 10.8% and 12.2%; TP 16 and 32 reduce it by about 42.7% and 65.6%. Exact values depend on the model and hardware, but the trend shows that TP degree cannot grow indefinitely.</figcaption>
</figure>

Compared with pipeline parallelism, TP:

- Has no pipeline bubbles, is relatively straightforward to package, and does not require a very large batch;
- Pays for this with more frequent communication.

For micro-batch activations of shape `b × s × h`:

- Pipeline parallelism transmits only about `bsh` data point-to-point between adjacent stages;
- Lecture 8 approximates TP communication for one Transformer block as:

\[
8bsh \frac{t - 1}{t}
\]

> **How is this formula derived?**
>
> It estimates the communication volume, in elements, caused by tensor parallelism **per rank** for one Transformer block's forward and backward pass. Let an activation tensor have size \(T = bsh\).
>
> In ring All-reduce, Reduce-scatter and All-gather each transfer \((t - 1)\) times, with \(T / t\) elements per transfer. Thus, one All-reduce transfers per rank:
>
> \[
> 2(t - 1)\frac{T}{t}
> = 2bsh\frac{t - 1}{t}
> \]
>
> The `2` represents Reduce-scatter and All-gather. A TP Transformer block typically performs four such All-reduces: one each at the attention output projection and MLP down-projection during the forward pass, and one each at the QKV projection and MLP up-projection during the backward pass. The total is therefore:
>
> \[
> 4 \times 2bsh\frac{t - 1}{t}
> = 8bsh\frac{t - 1}{t}
> \]
>
> This estimate ignores bytes per data type; with BF16, every element is 2 bytes, so actual bytes must be multiplied by 2. If an implementation fuses communication, uses sequence parallelism, or changes the compute graph, communication counts and tensor shapes can differ.

Here, `t` is the tensor-parallel degree. The constant depends on the block's communication arrangement, but the conclusion remains: TP uses **per-layer** collectives and is best served by **low-latency, high-bandwidth interconnects**.

#### 6.3.3. Activation Memory: A Bottleneck Even After Parameter Sharding

Parameters, gradients, and optimizer state are relatively static model state. Activations are produced during a forward pass, retained for backpropagation, and vary with batch and sequence length. TP and PP reduce parameter memory per GPU, but do not automatically eliminate activation memory.

When all intermediate activations are retained, the approximate activation memory of one Transformer layer is:

\[
M_{\text{act, layer}} = sbh\left(34 + \frac{5as}{h}\right).
\]

Here, \(b\) is micro-batch size, \(s\) sequence length, \(h\) hidden size, and \(a\) the number of attention heads.

> **How is this formula derived?**
>
> This is not an empirical constant; it is an itemized accounting of tensors required for backpropagation [5]. It assumes BF16 activations (2 bytes per element) and 1-byte dropout masks, while ignoring small buffers such as LayerNorm means and variances and GEMM bias. Its unit is **bytes**.
>
> - Attention block: the output-projection input uses \(2sbh\), the attention dropout mask \(sbh\), the shared QKV input \(2sbh\), Q and K retained for \(QK^\mathsf{T}\) \(4sbh\), and V for the attention output \(2sbh\), totaling \(11sbh\). Softmax output, softmax dropout mask, and attention dropout output use \(2as^2b\), \(as^2b\), and \(2as^2b\), totaling \(5as^2b\). The attention block therefore totals \(11sbh + 5as^2b\).
> - MLP: the first linear input uses \(2sbh\); the second linear input and GeLU input use \(8sbh\) each; and the dropout mask uses \(sbh\), totaling \(19sbh\).
> - Two LayerNorms each retain one input, totaling \(4sbh\).
>
> Therefore, total activation memory is \(11sbh + 5as^2b + 19sbh + 4sbh = 34sbh + 5as^2b = sbh\left(34 + \frac{5as}{h}\right)\). The \(5as^2b\) term comes from attention matrices and related dropout, which explains the quadratic growth for long sequences.

With tensor parallelism alone, activation memory is approximately:

\[
  sbh\left(10 + \frac{24}{t} + \frac{5as}{ht}\right).
\]

Although \(24 / t\) and the quadratic attention term shrink with TP degree, the \(10\), representing \(10sbh\), remains on every rank: about \(4sbh\) for LayerNorm, \(2sbh\) for Dropout, and \(4sbh\) for the attention and MLP inputs. These operations are per-token and cannot be further reduced just by splitting the hidden dimension.

##### More Ways to Reduce Intermediate Activations

1. **Selective Activation Recomputation** recomputes intermediate attention results during backpropagation instead of retaining them, trading additional compute for memory and removing the \(s^2\) term above;
2. Implementations also often use kernels such as **FlashAttention** to reduce the storage of attention intermediates.

#### 6.3.4. Sequence Parallelism

Sequence Parallelism (SP) targets the \(10sbh\) term that does not shrink with \(t\):

- It splits token-independent operations such as LayerNorm and Dropout along the sequence dimension, so every rank retains activations for only \(s / t\) tokens;
- Before attention or an MLP, which need full hidden computation, it gathers the data temporarily.

<figure>
  <img src="sequence-parallelism.png" alt="Sequence parallelism and tensor parallelism alternate within a Transformer block" width="100%">
  <figcaption>LayerNorm and Dropout use sequence parallelism; self-attention and the MLP use tensor parallelism. The two sharding schemes alternate within a block.</figcaption>
</figure>

During the forward pass, \(g\) in the figure is All-gather, collecting sequence shards into the input needed by tensor parallelism; \(\bar{g}\) is Reduce-scatter, splitting the result along the sequence again. Their order and roles swap during the backward pass. SP does not change TP parameter sharding; it also distributes per-token activations that would otherwise be replicated on every GPU.

#### 6.3.5. Summary

Activation memory per layer under different strategies can be written as:

- No parallelism: \(sbh(34 + 5as / h)\).
- TP only: \(sbh(10 + 24 / t + 5as / (ht))\).
- TP + SP: \(sbh(34 / t + 5as / (ht))\).
- TP + selective activation recomputation: \(sbh(10 + 24 / t)\).
- TP + SP + selective activation recomputation: \(sbh(34 / t)\).

TP mainly distributes parameters and part of the activations along width; SP additionally removes replication of per-token activations; recomputation handles the quadratic attention term. They are complementary, but all introduce additional communication or compute and must be balanced against sequence length, available memory, and interconnect capability.

### 6.4. Context Parallelism (CP)

Context parallelism targets long sequences. It splits activations along the sequence dimension, so **each rank retains and processes only one context segment**. Unlike sequence parallelism, which mainly handles per-token operations, CP also distributes self-attention across multiple GPUs.

In Ring Attention:

1. **Local operation**: every rank keeps its query block fixed and computes part of the attention with its local key-value block;
2. **Ring transfer**: ranks pass key-value blocks around a ring;
3. **Blockwise computation**: on receiving each new block, a rank accumulates blockwise attention against its local query block.

<figure>
  <img src="context-parallel.png" alt="In context parallelism, query blocks stay local while key-value blocks circulate between devices">
  <figcaption>Context parallelism / Ring Attention: every GPU retains its local query block, while key-value blocks move between devices to complete long-sequence attention block by block.</figcaption>
</figure>

This distributes long-sequence activations and attention computation across GPUs, avoiding full attention intermediates for the entire sequence on one GPU. The trade-off is that every attention round must transfer key-value blocks, so CP benefits most from high-bandwidth interconnects; its memory gain generally grows with sequence length.

### 6.5. Pipeline Parallelism (PP)

Pipeline parallelism assigns every rank a consecutive set of layers and passes full data or activations between adjacent ranks. It splits along model depth.

<figure>
  <img src="pipeline-parallelism.png" alt="Illustration of pipeline parallelism" width="50%">
  <figcaption>Pipeline parallelism: model layers are split by depth, and adjacent ranks pass activations between them.</figcaption>
</figure>

The CS336 Lecture 7 example splits a 4-layer model across 2 ranks and splits a batch into 4 micro-batches. Micro-batches reduce pipeline bubbles because different ranks can process micro-batches at different stages simultaneously.

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

#### 6.5.1. Layer-wise Splitting and Pipeline Bubbles

The simplest layer-wise split assigns the first layers to GPU 0, the following layers to GPU 1, and so on. Activations move forward in the forward pass, while activation gradients move backward in the backward pass.

<figure>
  <img src="layer-wise-parallelism.png" alt="Four GPUs each handle consecutive model layers in a layer-wise split">
  <figcaption>Layer-wise splitting: different GPUs handle consecutive model layers, passing activations forward and activation gradients backward.</figcaption>
</figure>

Layer-wise splitting with one batch has low utilization. With `n` GPUs, GPU 1 can start only after GPU 0 finishes the first layer's forward pass; the earlier GPUs mostly wait until gradients return layer by layer. Ideally, each GPU is active for only about `1 / n` of the time. This idle time is called a pipeline bubble.

<figure>
  <img src="layer-wise-utilization.png" alt="Stepped forward and backward execution with low utilization in a four-stage layer-wise split">
  <figcaption>Layer-wise splitting of one batch: forward and backward passes can progress only stage by stage, leaving the four GPUs waiting most of the time.</figcaption>
</figure>

Pipeline parallelism splits a batch into micro-batches. After GPU 0 sends the first micro-batch to the next stage, it immediately starts the second; ranks can therefore process micro-batches at different stages at the same time.

<figure>
  <img src="pipeline-microbatches.png" alt="Forward and backward scheduling of four micro-batches in a four-stage pipeline">
  <figcaption>Micro-batch pipeline scheduling: F<sub>i,j</sub> is the forward pass of micro-batch j at stage i, and B<sub>i,j</sub> is its backward pass; bubbles remain at the start and end.</figcaption>
</figure>

With \(n_{\text{stages}}\) pipeline stages and \(n_{\text{micro}}\) micro-batches, the ratio of bubble time to useful compute is approximately:

\[
\frac{T_{\text{bubble}}}{T_{\text{useful}}}
\approx
\frac{n_{\text{stages}} - 1}{n_{\text{micro}}}
\]

More stages increase fill-and-drain bubbles; more micro-batches amortize them, which is why pipelines usually need a sufficiently large batch. The number of micro-batches cannot grow indefinitely: overly small micro-batches reduce compute efficiency and increase scheduling overhead.

#### 6.5.2. Trade-offs Among Memory, Communication, and Batch Size

- Pipeline parallelism's primary benefit is memory: every GPU stores only consecutive layers, rather than the whole model;
- It also suits slower cross-node links: adjacent stages communicate only point-to-point, transferring activations of size mainly determined by \(b \times s \times h\), rather than exchanging full parameter shards on demand as FSDP does.

<figure>
  <img src="pipeline-batch-size-performance.png" alt="Pipeline-parallel scale and per-GPU throughput at different batch sizes">
  <figcaption>With small batches, more pipeline stages sharply reduce per-GPU throughput because of bubbles; with larger batches, micro-batches hide bubbles more effectively.</figcaption>
</figure>

Pipeline parallelism is therefore often used across slower inter-node links to improve model-memory scaling, but it is sensitive to batch size and the number of micro-batches.

#### 6.5.3. More Complex Scheduling and Zero-bubble Pipelines

One device can be assigned multiple pipeline stages, or an interleaved schedule can switch between stages and micro-batches. This reduces idle time at the cost of more inter-stage activation transfers and scheduling complexity: it trades communication bandwidth for utilization.

Zero-bubble pipelines further observe that backpropagation can be split into:

1. Activation backpropagation, which computes and passes input gradients and must return them to the preceding stage quickly;
2. Weight-gradient computation, which can be delayed once activations and upstream gradients are available.

<figure>
  <img src="zero-bubble-schedules.png" alt="Comparison of standard 1F1B scheduling with two zero-bubble pipeline schedules">
  <figcaption>Zero-bubble schedules: the upper schedule is standard 1F1B; the lower schedules fill idle regions with deferrable weight-gradient computation to improve device utilization.</figcaption>
</figure>

A scheduler can therefore place weight-gradient computation in what would otherwise be bubbles, reducing or even eliminating idle time. The trade-off is more complex scheduling and dependencies; weight updates must still occur only after all micro-batch gradients for the training step finish.

### 6.6. Expert Parallelism (EP)

Mixture of Experts (MoE) replaces some MLPs with multiple expert FFNs, and a gate selects a small number of experts for every token. See [Mixture of Experts (MoE)](../mixture-of-experts/) for routing details; this section focuses on how experts are placed across GPUs.

Tensor parallelism splits the width of one matrix, so every rank participates in the same matrix multiplication. Expert parallelism splits the **expert axis**: every rank holds one or more complete experts rather than a slice of every expert's matrix. EP therefore routes activations to the device that owns the target expert instead of splitting the matrix multiplication itself.

#### 6.6.1. Two All-to-all Operations: Dispatch Tokens, Then Return Results

Let an MoE layer have \(E\) experts. One EP forward pass is:

1. The gate on each rank selects experts for local tokens and encodes and groups tokens by the device holding each target expert;
2. The first All-to-all, or dispatch, sends token activations to the target expert ranks;
3. Every rank computes the received tokens with its locally stored complete expert FFNs;
4. The second All-to-all, or combine, returns outputs to the tokens' original ranks, where gate weights merge expert outputs.

<figure>
  <img src="expert-parallelism-routing.png" alt="Expert parallelism places complete experts on different devices and routes tokens through two All-to-all operations" width="100%">
  <figcaption>Expert parallelism: ordinary layers still operate on model replicas; in an MoE layer, the gate first dispatches tokens to experts through All-to-all, then returns and combines the results through another All-to-all.</figcaption>
</figure>

> **How does All-to-all act like a transpose here?**
>
> Let \(S_{i,j}\) denote the block of tokens currently on source Rank \(i\) whose target is expert Rank \(j\). Before dispatch, Rank \(i\) holds one whole row \([S_{i,0}, S_{i,1}, \ldots, S_{i,E-1}]\), grouping local tokens by target expert.
>
> | Source rank | To Expert Rank 0 | To Rank 1 | To Rank 2 | To Rank 3 |
> | --- | --- | --- | --- |
> | Rank 0 | \(S_{0,0}\) | \(S_{0,1}\) | \(S_{0,2}\) | \(S_{0,3}\) |
> | Rank 1 | \(S_{1,0}\) | \(S_{1,1}\) | \(S_{1,2}\) | \(S_{1,3}\) |
> | Rank 2 | \(S_{2,0}\) | \(S_{2,1}\) | \(S_{2,2}\) | \(S_{2,3}\) |
> | Rank 3 | \(S_{3,0}\) | \(S_{3,1}\) | \(S_{3,2}\) | \(S_{3,3}\) |
>
> After the first All-to-all, Rank \(j\) receives the original matrix's \(j\)-th column, \([S_{0,j}, S_{1,j}, \ldots, S_{E-1,j}]\). In other words, \(\operatorname{recv}_{j,i}=S_{i,j}\): the source-rank and target-expert-rank indices exchange places. This is a blockwise transpose along the rank dimension.
>
> For example, tokens on Rank 0 that select Expert 1 move from \(S_{0,1}\) to Rank 1; tokens on Rank 2 selecting Expert 1, \(S_{2,1}\), also arrive at Rank 1 and can be computed by Expert 1 together. After expert computation, the second All-to-all reverses the index exchange and returns outputs to their original ranks.
>
> Token counts are often unbalanced in practice, so this is not a literal regular-shape `tensor.T`; it is an “ownership transpose” of variable-length token blocks.

EP's core communication is many-to-many exchange of token activations:

- Unlike PP, it is not limited to adjacent stages;
- Unlike TP, it does not reduce all ranks in a fixed order at every layer;
- The routing destination is determined by the gate for the current batch.

This requires both high **All-to-all bandwidth** and well-balanced routing. If one expert receives too many tokens, its rank becomes the tail-latency bottleneck.

#### 6.6.2. Why MoE MLPs Usually Prefer EP

For an MoE expert MLP, EP behaves similarly to TP in that both distribute expert parameters and activations per GPU and depend on high-bandwidth interconnects. EP is usually more suitable, however:

- **Larger local GEMMs**: EP retains complete expert matrices instead of further shrinking one expert's matrices as TP does. Larger local matrix multiplications usually achieve higher GPU utilization.
- **Lower communication burden**: TP still needs collectives at the linear-layer boundaries of every expert MLP. EP uses two All-to-all operations to dispatch and return tokens, which usually costs less communication for an MoE layer.
- **Simpler compute graph**: each expert computes independently after receiving tokens, making it easier to overlap communication and expert computation.
- **Less local permutation**: when \(\mathrm{EP}=E\), every EP rank holds exactly one expert, so expert index directly identifies the target rank and avoids extra local token permutation caused by several experts on one device.

#### 6.6.3. Combining EP and DP

EP is not necessarily an additional dimension beyond DP. In common layouts, it splits the original DP group into expert sharding and expert replicas. With TP, PP, and other dimensions fixed, the usual relationship is \(\mathrm{DP}=\mathrm{EP}\times\mathrm{EDP}\), where Expert Data Parallelism (EDP) is the number of replicas of the same expert shard.

Thus, if \(\mathrm{DP}=8\) and \(\mathrm{EP}=2\), this does not require \(8\times2=16\) GPUs. It is still 8 GPUs, with \(\mathrm{EDP}=4\).

For example, with 4 experts and \(\mathrm{TP}=\mathrm{PP}=1\), 8 GPUs can be placed as follows:

| EP group | Rank | Expert shard held |
| --- | --- | --- |
| 0 | Rank 0 | Expert 0, 1 |
| 0 | Rank 1 | Expert 2, 3 |
| 1 | Rank 2 | Expert 0, 1 |
| 1 | Rank 3 | Expert 2, 3 |
| 2 | Rank 4 | Expert 0, 1 |
| 2 | Rank 5 | Expert 2, 3 |
| 3 | Rank 6 | Expert 0, 1 |
| 3 | Rank 7 | Expert 2, 3 |

This layout has two views:

- **Across EP**: `[Rank 0, Rank 1]` together hold a complete set of experts but retain different shards. A token on Rank 0 selecting Expert 2 or 3 is sent to Rank 1 by All-to-all; one EP group completes routing and computation for the MoE layer.
- **Across EDP**: `[Rank 0, Rank 2, Rank 4, Rank 6]` all hold Experts 0 and 1, forming four replicas of the same expert shard. They process different data shards and synchronize that expert shard's gradients after backpropagation.

Dense attention parameters are therefore replicated on all 8 ranks and synchronize gradients with ordinary \(\mathrm{DP}=8\), whereas Experts 0 and 1 synchronize only across their four replicas with \(\mathrm{EDP}=4\). The condition \(\mathrm{EP}\leq\mathrm{DP}\) means that EP is carved from the existing DP dimension and should not be multiplied into the total GPU count again. Frameworks may lay out ranks differently, but the relationship of “shard experts horizontally, replicate them vertically” remains the same.

#### 6.6.4. Combining EP with TP and PP

EP can also combine with TP and PP, but process groups for different layers cannot simply all be shared.

The difficulty is that MoE usually replaces only MLPs, while attention remains dense:

- Attention layers cannot use EP; higher TP helps distribute their matrix multiplications;
- Expert MLPs prefer lower TP and higher EP to retain larger local GEMMs.

If every layer shares the same TP size, these requirements must be compromised. DP and TP grouping can also leave too few tokens for each expert, reducing matrix-multiplication utilization.

<figure>
  <img src="moe-parallel-folding.png" alt="MoE Parallel Folding uses different parallel groups for attention and MoE layers" width="100%">
  <figcaption>MoE Parallel Folding: attention and MoE layers use different parallel decompositions to match the needs of the two computations.</figcaption>
</figure>

Megatron calls this decoupling MoE Parallel Folding:

- Attention layers can use \(\mathrm{TP}\times\mathrm{CP}\times\mathrm{DP}\times\mathrm{PP}\), where Context Parallelism (CP) splits along the sequence dimension;
- MoE layers can use \(\mathrm{ETP}\times\mathrm{EP}\times\mathrm{EDP}\times\mathrm{PP}\), where Expert Tensor Parallelism (ETP) is used only when an individual expert must be split further.

The two share PP stage boundaries but reorganize TP, DP, and EP process groups over the same devices. This does not split the model across two independent sets of hardware; the same GPUs use different communication groups when entering attention or MoE layers. It improves per-layer compute efficiency at the cost of more complex device mapping, token routing, and communication scheduling.

### 6.7. Parallelization Strategy Comparison

<figure>
  <img src="llm-parallelism-comparison-table.png" alt="Comparison of LLM parallelization strategies in communication, memory, bandwidth, batch scaling, and ease of use" width="100%">
  <figcaption>Lecture 8 overview of the main trade-offs among DDP, FSDP, PP, TP, SP/CP, and EP.</figcaption>
</figure>

### 6.8. Parallelization Configuration Guidance

#### 6.8.1. Matching Batch Size to the Parallelization Strategy

**The relationship between per-device batch size and the parallelization strategy** matters; this is not simply a matter of making the batch as large as possible. We use \(B / N\), global batch size divided by the number of chips, to measure the work assigned to each device.

- If \(B / N\) is too small, there is too little compute to amortize communication. GPUs wait for communication even when the model fits.
- With intermediate \(B / N\), mixing FSDP with Model Parallelism (MP) is more communication-efficient.
- With sufficiently large \(B / N\), pure FSDP can also amortize communication overhead.

Batch size must therefore satisfy two constraints:

- Training stability and memory limits;
- Within that range, an effective per-device batch large enough to amortize communication;
- When the micro-batch cannot grow, Gradient Accumulation can increase the effective batch.

#### 6.8.2. Configuration Order and Practical Experience

The configuration order is:

1. When the model does not fit in memory:
   - First use the smallest sufficient model parallelism, including PP, TP, EP, SP, and CP, to make it fit;
   - Increase TP/EP within a node and PP across nodes, or use ZeRO-3 / FSDP depending on bandwidth.
2. Once the model fits: use remaining GPUs for DP to scale global batch size and throughput.

| Scenario | Configuration guidance |
| --- | --- |
| Frequent TP or EP communication | Keep it within one NVLink domain, typically no more than 8 GPUs in one node; across nodes, prefer increasing PP. |
| Significant PP bubbles | When PP ≥ 2, use Virtual Pipeline Parallelism (VPP) to reduce bubbles. |
| MoE expert layers | Prefer EP; local GEMMs are larger and communication is usually lower than TP. |
| Sequence length of about ≥8K tokens | Enable CP and overlap KV communication with computation. |
| Insufficient activation memory | Use Activation Recomputation, trading extra compute for a larger batch. |

---

## References

[1] Stanford CS336, "Lecture 7: Parallelism." [Online]. Available: https://cs336.stanford.edu/lectures/?trace=lecture_07.

[2] NVIDIA, "How to reason about collective operations." [Online]. Available: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce.

[3] Stas Bekman, "Sample benchmarking code." [Online]. Available: https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py.

[4] Stanford CS336, "Lecture 08." [Online]. Available: https://cs336.stanford.edu/lectures/?trace=lecture_08.

[5] V. Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models." Proceedings of Machine Learning and Systems, 2023. [Online]. Available: https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf.
