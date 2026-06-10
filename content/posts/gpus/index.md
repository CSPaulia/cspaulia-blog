---
title: "GPUs, TPUs"
date: 2026-06-10T11:17:05+08:00
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
description: "CS336 Lecture 5 学习笔记。"
disableHLJS: true
hideSummary: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
UseHugoToc: true
---

## 1. GPU 和 CPU 的区别

![CPU 和 GPU 的结构对比](gpu-vs-cpu.png)

- **CPU** 把更多芯片面积留给控制逻辑和缓存，适合处理复杂分支、系统调度、少量但灵活的任务；

- **GPU** 把更多面积留给计算单元，适合同时执行大量相似的小计算，比如矩阵乘法和深度学习训练。

基础单元解释：

- **算术逻辑单元**（**Arithmetic Logic Unit，ALU**）：负责加法、乘法、比较等计算；
- **Control**：控制单元，负责解释指令、安排计算顺序、处理分支；
- **Cache**：高速缓存，容量比 DRAM 小，但离计算单元更近、访问更快。它用来临时保存马上要用或刚用过的数据，减少反复访问慢速内存的次数；
- **DRAM**：容量较大的外部内存，用来存放更多数据，但访问速度比 Cache 慢。

GPU 有许多小的 ALU，优化了计算延迟（每个线程（Thread）完成计算更快了）。

GPU 降低了控制和存储的比重，保证了计算的吞吐量（总的处理的数据量增大了）。

**总结**：CPU 针对少量但复杂的线程进行了优化，而 GPU 针对大量但简单的线程进行了优化。

## 2. GPU 的详细构造

### 2.1 GPU 与 SM

<figure>
  <img src="ga100-full-gpu.png" alt="GA100 GPU 结构图">
  <figcaption>GA100 GPU 的整体结构。一个完整 GA100 包含 128 个 SM；实际 A100 启用其中 108 个 SM。</figcaption>
</figure>

<figure>
  <img src="ga100-sm.png" alt="GA100 SM 结构图">
  <figcaption>GA100 中单个 SM（Streaming Multiprocessor）的内部结构；图中 SM 被分成 4 个处理分区，每个分区都有自己的调度、寄存器和计算单元。</figcaption>
</figure>

GPU 中包含许多 **SM**（Streaming Multiprocessor），每个 SM 都可以独立执行一个 **block**（可以理解成一个任务）。

每个 SM 中又包含许多 **SP**（Streaming Processor），这些 SP 可以**并行**执行多个 **thread**。

### 2.2 内存层级

<figure>
  <img src="gpu-memory-hierarchy.png" alt="GPU 内存层级图">
  <figcaption>GPU 的内存层级。L1 / shared memory 在 SM 内部，L2 cache 在 GPU 芯片上，global memory / DRAM 在 GPU 旁边的显存芯片中。</figcaption>
</figure>

GPU 的内存越靠近 **SM**，访问速度越快：

- **Register**：每个 thread 私有，用来存放正在计算的临时变量。它在 SM 内部，离 ALU 最近，速度最快，但容量最小；
- **Shared memory**：在 SM 内部，同一个 block 里的 thread 可以共享。它通常由片上 SRAM 实现，需要程序员显式读写；
- **L1 cache**：在 SM 内部，用来缓存最近访问的数据。它也在片上，很多 GPU 上 L1 和 shared memory 会共享同一块物理 SRAM，只是使用方式不同；
- **L2 cache**：在 GPU 芯片上，被多个 SM 共享。它仍然是片上 cache，但离具体的 ALU 更远；
- **Global memory / DRAM**：在 GPU 旁边的显存芯片中，容量最大，但访问最慢。

所以它们不是完全一样的东西。**Register、L1/shared memory、L2 cache** 都在 GPU 芯片内部，通常属于高速片上存储；**global memory** 是片外 DRAM/HBM。速度差异既来自距离，也来自物理介质、容量、共享范围和访问方式的不同。

为什么需要这么多层存储？因为**又快又大的内存做不出来**。越快的存储越贵、越占芯片面积、容量越小；越大的存储越便宜、容量越大，但离计算单元更远、访问更慢。

这些存储层的作用不同：

- **Register**：服务单个 thread，保存当前计算马上要用的变量；
- **Shared memory**：服务一个 block，让同一个 block 里的 thread 交换和复用数据；
- **L1 cache**：服务一个 SM，自动缓存最近访问的数据，减少去更远层级取数据；
- **L2 cache**：服务整颗 GPU，让不同 SM 访问 global memory 前先经过一个共享缓存；
- **Global memory**：服务整个 GPU，存放模型参数、输入输出、激活值等大数据。

### 2.3 板卡组成

<figure>
  <img src="gpu-board-vram.png" alt="GPU 板卡结构图">
  <figcaption>GPU 板卡结构。GPU 芯片负责计算，旁边的 VRAM / HBM 负责存放数据，板卡上的接口和供电模块负责连接与供电。</figcaption>
</figure>

从物理板卡上看，GPU 不只是一个计算芯片，而是一整块加速卡：

- **Graphics Processing Unit（GPU）**：核心计算芯片，里面包含许多 SM，负责执行 CUDA kernel；
- **Video Memory（VRAM）**：GPU 旁边的显存芯片，也就是前面说的 global memory / DRAM。模型参数、激活值、输入输出等大数据主要放在这里；
- **Motherboard interface**：连接主板的接口，一般是 PCIe。CPU 和 GPU 之间的数据传输会经过这里；
- **Interconnection interface**：GPU 之间互联的接口，用来让多张 GPU 交换数据；
- **Voltage Regulator Module（VRM）**：电压调节模块，把外部供电转换成 GPU 芯片和显存需要的稳定电压；
- **Network interface / DPU**：图中这张板卡还包含网络和 DPU 相关模块，用来处理网络通信或数据搬运。

### 2.4 执行模型

<figure>
  <img src="cuda-execution-model.svg" alt="CUDA 执行模型图">
  <figcaption>CUDA 执行模型。程序被拆成 blocks，block 被分配到 SM 上执行；每个 block 又被拆成多个 warp，每个 warp 通常包含 32 个连续 thread。</figcaption>
</figure>

GPU 的执行模型中有三个重要概念：

- **Thread**：真正执行计算的单位。许多 thread 会并行工作，它们通常执行相同的指令，但处理不同的数据，这种模式叫 **SIMT**（Single Instruction, Multiple Threads）；
- **Block**：一组 thread。每个 block 会被分配到某个 SM 上执行，并拥有自己的 shared memory；
- **Warp**：GPU 实际调度 thread 的基本单位。thread 总是按 warp 执行，一个 warp 通常包含 32 个连续编号的 thread。

它们和硬件的关系可以这样理解：

- **Block**：分配给一个 SM；
- **Warp**：SM 内部实际调度的单位；
- **Thread**：warp 里的单个逻辑执行流，最终由 SM 内部计算单元执行。

### 2.5 内存访问范围

<figure>
  <img src="cuda-memory-access-scope.png" alt="CUDA 内存访问范围图">
  <figcaption>CUDA 内存访问范围。每个 thread 有自己的 registers；同一个 block 内的 threads 可以共享 shared memory；跨 block 的数据交换需要经过 global memory。</figcaption>
</figure>

CUDA 程序中的内存访问范围可以这样理解：

- **每个 thread** 可以读写自己的 **registers** 和 local memory；
- **每个 block** 有自己的 **shared memory**，block 内的 threads 可以共同访问；
- **整个 grid** 可以访问 **global memory**，不同 block 之间交换数据通常需要通过 global memory；
- **constant memory** 是整个 grid 可读的只读内存；
- **host code** 可以在 CPU 侧和 GPU 的 global / constant memory 之间传输数据。

## 3. TPU 简单对比

<figure>
  <img src="tpu-tensorcore-layout.png" alt="TPU TensorCore 抽象结构图">
  <figcaption>TPU TensorCore 的抽象结构。TPU 和 GPU 在高层上很像：轻量控制、大矩阵乘法单元、快速内存。</figcaption>
</figure>

GPU、TPU 和很多 AI 加速器在高层上是相似的：

- 都有较轻量的控制单元；
- 都有很快的矩阵乘法单元；
- 都有靠近计算单元的高速内存；
- 都依赖 HBM 存放权重、激活值、优化器状态和 batch 数据。

TPU 中几个重要部件可以这样理解：

- **Scalar Unit**：类似控制单元，负责调度指令；
- **VPU**（Vector Unit）：做 elementwise 操作，比如激活函数，也会把数据送入矩阵乘法单元；
- **MXU**（Matrix Multiply Unit）：负责大规模矩阵乘法，是 TPU FLOPs 的主要来源；
- **VMEM / SMEM**：靠近计算单元的片上高速内存；
- **HBM**：高带宽显存，用来存放大规模数据。HBM bandwidth 决定数据进入计算单元的速度。

GPU 和 TPU 的大致对应关系：

| GPU | TPU | 含义 |
| --- | --- | --- |
| Streaming Multiprocessor（SM） | TensorCore | 包含其他单元的核心计算模块 |
| Warp Scheduler | VPU | SIMD 向量算术单元 |
| CUDA Core | VPU ALU | SIMD ALU |
| SMEM（L1 Cache） | VMEM | 快速片上缓存 |
| Tensor Core | MXU | 矩阵乘法单元 |
| HBM（也就是 GMEM） | HBM | 高带宽大容量显存 |

从数量上看，GPU 往往有更多小计算模块，TPU 则有更少但更大的矩阵计算模块：

| GPU | TPU | H100 数量 | TPU v5p 数量 |
| --- | --- | ---: | ---: |
| SM | Tensor Core | 132 | 2 |
| Warp Scheduler | VPU slots | 528 | 8 |
| SMEM（L1 cache） | VMEM | 32MB | 128MB |
| Registers | Vector Registers（VRegs） | 32MB | 256kB |
| Tensor Core | MXU | 528 | 8 |

主要差异：

- **GPU** 有更多 SM，调度粒度更细，通用性更强；
- **TPU** 的 TensorCore 数量更少，但矩阵乘法单元更大，更偏向规则的大矩阵计算；
- **GPU** 有 warp，硬件以 warp 为单位调度 thread；
- **TPU** 没有 warp，更多是 block 级执行，这会带来矩阵乘法和非矩阵计算之间的取舍；
- 多 GPU / 多 TPU 的差异，还和加速器之间如何互联有关。

**总结**：GPU 和 TPU 的共同目标都是让矩阵乘法跑得很快。GPU 更通用，TPU 更专门面向大规模矩阵计算。

## 4. GPU 模型的优势与限制

早期 NVIDIA GPU 主要面向图形渲染，提供的是 programmable shaders。研究者发现，矩阵乘法也可以“伪装”成图形计算来运行，这就是早期用 GPU 做通用计算的思路之一。

现在的 GPU 已经有专门的矩阵乘法硬件，例如 **Tensor Core**。Tensor Core 是专门为矩阵乘加设计的电路，因此 matmul 的速度远高于普通浮点运算。

<figure>
  <img src="matmul-vs-nonmatmul-flops.png" alt="Matmul 和非 Matmul FLOPs 对比">
  <figcaption>随着 GPU 架构演进，matmul FLOPs 增长明显快于普通浮点运算。</figcaption>
</figure>

这对大模型非常重要，因为 Transformer 里的主要计算就是大规模矩阵乘法。也就是说，现代 GPU 不是“所有计算都一样快”，而是**特别擅长矩阵乘法**。

但是，计算能力增长得比内存和互联带宽更快：

<figure>
  <img src="compute-vs-memory-scaling.png" alt="计算能力和内存带宽扩展速度对比">
  <figcaption>硬件 FLOPs 增长速度快于 DRAM 带宽和互联带宽。</figcaption>
</figure>

这会带来一个问题：计算单元越来越快，但数据不一定能及时送到计算单元。也就是说，GPU 程序慢不一定是因为“算得不够快”，也可能是因为“数据搬得太慢”。这就是为什么现在的很多工程优化都是面向内存和传输的。

**总结（GPU 的优势）**：

- **容易扩展**：如果任务足够大，可以通过增加更多 SM 来提升吞吐量；
- **相对容易编程**：SIMT 模型让程序员可以像写很多 thread 一样写程序，硬件负责把它们组织成 warp 执行；
- **线程很轻量**：GPU 可以在大量 thread 之间切换。当一些 thread 等待内存时，SM 可以调度其他 ready thread 继续执行；
