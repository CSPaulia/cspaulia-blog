---
title: "GPU Architecture and Machine Learning Optimization"
date: 2026-06-10T11:17:05+08:00
series:
  main: "Large Language Model"
  subseries: "System and Hardware"
draft: false
categories: ["大语言模型", "系统"]
tags: ["GPU", "CUDA", "并行计算", "训练"]
author: "CSPaulia"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "CS336 Lecture 5 Study Notes."
disableHLJS: true
hideSummary: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
UseHugoToc: true
disableShare: false
searchHidden: false
ShowRssButtonInSectionTermList: true
cover:
    image: "gpu-vs-cpu.png"
    alt: "GPU vs CPU"
    caption: "CPU 和 GPU 的结构对比"
    relative: true
    hidden: false
    hiddenInList: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

## 1. GPU Architecture

### 1.1 Differences between GPU and CPU

![Structural comparison of CPU and GPU](gpu-vs-cpu.png)

- **CPU** allocates more chip area to control logic and cache, suitable for handling complex branches, system scheduling, and a small number of flexible tasks;
- **GPU** dedicates more area to compute units, suitable for executing a large number of similar small computations simultaneously, such as matrix multiplication and deep learning training.

Basic unit explanation:

- **Arithmetic Logic Unit (ALU)**: Responsible for computations such as addition, multiplication, and comparison;
- **Control**: Control unit, responsible for interpreting instructions, scheduling computation order, and handling branches;
- **Cache**: High-speed cache, smaller capacity than DRAM but closer to the compute unit and faster to access. It temporarily stores data that is about to be used or has just been used, reducing repeated accesses to slow memory;
- **DRAM**: External memory with larger capacity, used to store more data, but access speed is slower than Cache.

GPUs have many small ALUs, optimized for computation latency (each thread completes computation faster).

GPUs reduce the proportion of control and storage, ensuring computation throughput (the total amount of data processed increases).

**Summary**: CPU is optimized for a small number of complex threads, while GPU is optimized for a large number of simple threads.

### 1.2 Detailed Structure of GPU

#### 1.2.1 GPU and SM

<figure>
  <img src="ga100-full-gpu.png" alt="GA100 GPU architecture diagram">
  <figcaption>Overall structure of GA100 GPU. A complete GA100 contains 128 SMs; actual A100 enables 108 of them. </figcaption>
</figure>

<figure>
  <img src="ga100-sm.png" alt="GA100 SM architecture diagram">
  <figcaption>Internal structure of a single SM (Streaming Multiprocessor) in GA100; the SM in the figure is divided into 4 processing partitions, each with its own scheduler, registers, and compute units. </figcaption>
</figure>

A GPU contains many **SMs** (Streaming Multiprocessors), each SM can independently execute a **block** (which can be understood as a task).

Each SM contains many **SPs** (Streaming Processors), which can execute multiple **threads** in **parallel**.

#### 1.2.2 Memory Hierarchy

<figure>
  <img src="gpu-memory-hierarchy.png" alt="GPU memory hierarchy diagram">
  <figcaption>Memory hierarchy of GPU. L1 / shared memory is inside the SM, L2 cache is on the GPU chip, global memory / DRAM is in the VRAM chips next to the GPU. </figcaption>
</figure>

The closer the memory is to the **SM**, the faster the access speed:

- **Register**: Private to each thread, used to store temporary variables for current computation. It is inside the SM, closest to the ALU, fastest, but smallest capacity;
- **Shared memory**: Inside the SM, shared by threads within the same block. It is typically implemented with on-chip SRAM and requires explicit read/write by the programmer;
- **L1 cache**: Inside the SM, used to cache recently accessed data. It is also on-chip. On many GPUs, L1 and shared memory share the same physical SRAM, just used differently;
- **L2 cache**: On the GPU chip, shared by multiple SMs. It is still an on-chip cache, but farther from the specific ALU;
- **Global memory / DRAM**: In the VRAM chips next to the GPU, largest capacity, but slowest access.

So they are not exactly the same thing. **Registers, L1/shared memory, L2 cache** are all inside the GPU chip, generally high-speed on-chip storage; **global memory** is off-chip DRAM/HBM. Speed differences come not only from distance but also from physical medium, capacity, sharing scope, and access method.

Why so many layers of storage? Because **fast and large memory cannot be made**. The faster the storage, the more expensive, the more chip area it takes, and the smaller the capacity; the larger the storage, the cheaper and larger, but farther from compute units and slower to access.

The roles of these storage layers are different:

- **Register**: Serves a single thread, holds variables needed immediately for current computation;
- **Shared memory**: Serves a block, allows threads within the same block to exchange and reuse data;
- **L1 cache**: Serves an SM, automatically caches recently accessed data, reducing accesses to farther levels;
- **L2 cache**: Serves the entire GPU, provides a shared cache before SMs access global memory;
- **Global memory**: Serves the entire GPU, stores model parameters, inputs, outputs, activations, and other large data.

The following lists key hardware specifications for three generations of GPUs: A100, H100, and B200, providing a more intuitive view of evolution across generations:

| Accelerator                        | A100      | H100      | B200      |
|------------------------------------|-----------|-----------|-----------|
| # SMs                              |       108 |       132 |       148 |
| Register size (per SM)             |    256 KB |    256 KB |    256 KB |
| L1 cache + shared memory (per SM)  |    192 KB |    256 KB |    256 KB |
| L2 cache size                      |     40 MB |     50 MB | 96-126 MB |
| HBM size                           |     80 GB |     80 GB |    192 GB |
| Register bandwidth                 | ~116 TB/s | ~401 TB/s | ~447 TB/s |
| L1 cache + shared memory bandwidth |  ~19 TB/s |  ~33 TB/s |  ~19 TB/s |
| L2 cache bandwidth                 | ~5-8 TB/s |  ~12 TB/s |   ~9 TB/s |
| HBM bandwidth                      |    2 TB/s | 3.35 TB/s |    8 TB/s |

(B200 also introduces **tensor memory (TMEM)** between Tensor Cores and registers/shared memory, invisible to the programmer but can further improve Tensor Core efficiency.)

### 1.2.3 Board Composition

<figure>
  <img src="gpu-board-vram.png" alt="GPU board architecture diagram">
  <figcaption>GPU board structure. The GPU chip performs computation, the adjacent VRAM / HBM stores data, and the board's interfaces and power modules handle connection and power supply. </figcaption>
</figure>

From a physical board perspective, a GPU is not just a compute chip but an entire accelerator card:

- **Graphics Processing Unit (GPU)**: The core compute chip, containing many SMs, responsible for executing CUDA kernels;
- **Video Memory (VRAM)**: The VRAM chips next to the GPU, i.e., the earlier mentioned global memory / DRAM. Large data like model parameters, activations, inputs, and outputs are mainly stored here;
- **Motherboard interface**: The interface connecting to the motherboard, typically PCIe. Data transfer between CPU and GPU goes through here;
- **Interconnection interface**: Interface for connecting multiple GPUs to exchange data;
- **Voltage Regulator Module (VRM)**: Converts external power supply into stable voltage required by the GPU chip and VRAM;
- **Network interface / DPU**: The board in this figure also includes network and DPU-related modules for handling network communication or data movement.

#### 1.2.4 Execution Model

<figure>
  <img src="cuda-execution-model.svg" alt="CUDA execution model diagram">
  <figcaption>CUDA execution model. A program is split into blocks, blocks are assigned to SMs for execution; each block is further split into warps, each warp typically contains 32 consecutive threads. </figcaption>
</figure>

There are three important concepts in the GPU execution model:

- **Thread**: The actual unit of computation. Many threads work in parallel, typically executing the same instructions but on different data. This pattern is called **SIMT** (Single Instruction, Multiple Threads);
- **Block**: A group of threads. Each block is assigned to a specific SM for execution and has its own shared memory;
- **Warp**: The basic unit for GPU thread scheduling. Threads are always executed in warps, with one warp typically containing 32 consecutively numbered threads.

The relationship with hardware can be understood as:

- **Block**: Assigned to an SM;
- **Warp**: The unit scheduled within an SM;
- **Thread**: A single logical execution flow within a warp, ultimately executed by the compute units inside the SM.

Specifically, when you launch a CUDA kernel (e.g., `my_kernel__HTMLTAG_27__>>(args)` or `my_kernel[(grid,)](args)` in Triton), the GPU execution flow is as follows:

1. **Grid Division**: The entire computation task is wrapped into a **grid**. The grid dimensions are determined by launch parameters—for example, `gridDim` with `(M, N)` means there are M × N blocks. You **explicitly specify** the grid division; you decide "how many blocks to cut the task into";

2. **Block Assignment to SMs**: The hardware **assigns** blocks from the grid one by one to SMs with available resources. An SM can hold multiple blocks simultaneously (depending on shared memory and register capacity). Assigned blocks stay on the SM until execution completes;

3. **Block Internal Grouping into Warps**: After a block enters an SM, the hardware groups every 32 consecutive threads into one **warp** based on thread ID order. This process is **automatic by hardware**—you only need to specify the block size, no manual warp management;

4. **Warp Scheduler Scheduling**: The **warp scheduler** inside the SM selects a "ready" warp each clock cycle and issues **the same instruction** to all 32 threads simultaneously. These 32 threads execute the same instruction on different data—this is **SIMT (Single Instruction, Multiple Threads)**;

5. **Zero-overhead Switching**: When a warp is blocked (e.g., waiting for HBM read/write), the warp scheduler switches directly to another ready warp. The switch itself consumes no extra clock cycles. An SM typically hosts multiple warps simultaneously, using frequent switching to hide memory latency.

Thus, the mapping chain from code to hardware is:

```
kernel 启动参数 → Grid（block 数量由你指定）
                   → Block（分配到各个 SM）
                       → Warp（硬件自动按 32 个 thread 分组）
                           → Thread（执行计算的逻辑个体）
```

Simple note: **You determine the grid and block sizes; the hardware groups threads in a block into warps and schedules execution.**

#### 1.2.5 Memory Access Scope

<figure>
  <img src="cuda-memory-access-scope.png" alt="CUDA memory access scope diagram">
  <figcaption>CUDA memory access scope. Each thread has its own registers; threads within the same block can share shared memory; data exchange across blocks must go through global memory. </figcaption>
</figure>

The memory access scope in a CUDA program can be understood as:

- **Each thread** can read/write its own **registers** and local memory;
- **Each block** has its own **shared memory**, accessible by threads within the block;
- **The entire grid** can access **global memory**; data exchange between different blocks typically requires global memory;
- **Constant memory** is read-only memory accessible by the entire grid;
- **Host code** can transfer data between the CPU side and GPU global / constant memory.

#### 1.2.6 Warp Occupancy (Register Pressure and Occupancy)

Each thread can use 0 to 255 registers. The more registers a thread uses, the fewer threads an SM can simultaneously accommodate, resulting in lower **occupancy**.

Use a concrete example: Assume each thread block has 128 threads, each thread uses 160 registers, the SM has at most 65536 registers, and supports a maximum of 64 concurrent warps:

```
num_threads_per_block = 128
num_registers_per_thread = 160

max_registers = 65536  # SM 上的寄存器总数
max_warps = 64         # SM 最多支持的并发 warp 数

num_registers_per_block = 128 × 160 = 20480
num_blocks = 65536 // 20480 = 3      # 受寄存器数量限制，最多放 3 个 block
num_warps = 3 × 128 / 32 = 12       # 实际运行的 warp 数
occupancy = 12 / 64 = 18.75%        # 占用率不到 20%
```

Low occupancy is not necessarily bad—if each thread is doing more work (e.g., **thread coarsening**, where one thread processes multiple elements), low occupancy can be acceptable. The key is not occupancy itself, but whether the SM's compute resources are efficiently utilized.

#### 1.2.7 Bank Conflicts (Shared Memory Bank Conflicts)

Shared memory is divided into **32 banks**, each bank is 4 bytes wide. The arrangement is as follows:

```
B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 ...
... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
```

Within each clock cycle, each bank can only be accessed by one thread (unless accessing the exact same address, which triggers a broadcast). If multiple threads simultaneously access different addresses in the same bank, the accesses are **serialized**—this is a **bank conflict**.

Worst-case example: A matrix row spans all banks. If 32 threads all access the first column, they all hit different rows but the same column offset—i.e., all hit bank 0—resulting in a 32-way bank conflict.

In matrix multiplication `A @ B`, bank conflicts are somewhat unavoidable: we need simultaneous access to rows of A and columns of B.

A common mitigation technique is **swizzling**: applying a transformation (e.g., XOR of row and column) to the shared memory address to scramble the bank assignment, thereby reducing conflicts.

#### 1.2.8 Block Occupancy (Block-Level Occupancy)

Thread blocks are scheduled to SMs in waves. If the number of SMs does not evenly divide the number of thread blocks, the last wave will have fewer blocks than SMs, leaving some SMs idle (wave quantization problem).

Take B200 as an example: It has 148 SMs. If 160 thread blocks are launched, the first wave of 148 blocks fill all SMs, the second wave has only 12 blocks, leaving 136 SMs idle.

The solution is straightforward: **make the number of thread blocks divisible by the number of SMs** to avoid a "starved" last wave.

### 1.3 Simple Comparison with TPU

<figure>
  <img src="tpu-tensorcore-layout.png" alt="TPU TensorCore abstract architecture diagram">
  <figcaption>Abstract structure of TPU TensorCore. TPUs and GPUs are similar at a high level: lightweight control, large matrix multiplication units, fast memory. </figcaption>
</figure>

GPUs, TPUs, and many AI accelerators are similar at a high level:

- All have relatively lightweight control units;
- All have fast matrix multiplication units;
- All have high-speed memory close to the compute units;
- All rely on HBM to store weights, activations, optimizer states, and batch data.

Key components in TPU can be understood as:

- **Scalar Unit**: Similar to control unit, responsible for instruction scheduling;
- **VPU (Vector Unit)**: Performs elementwise operations like activation functions, and also feeds data into the matrix multiplication unit;
- **MXU (Matrix Multiply Unit)**: Responsible for large-scale matrix multiplication, the main source of TPU FLOPs;
- **VMEM / SMEM**: On-chip high-speed memory close to compute units;
- **HBM**: High-bandwidth memory for storing large-scale data. HBM bandwidth determines how fast data enters the compute units.

Approximate correspondence between GPU and TPU:

| GPU | TPU | Meaning |
| --- | --- | --- |
| Streaming Multiprocessor (SM) | TensorCore | Core compute module containing other units |
| Warp Scheduler | VPU | SIMD vector arithmetic unit |
| CUDA Core | VPU ALU | SIMD ALU |
| SMEM (L1 Cache) | VMEM | Fast on-chip cache |
| Tensor Core | MXU | Matrix multiplication unit |
| HBM (i.e., GMEM) | HBM | High-bandwidth large-capacity memory |

In terms of numbers, GPUs tend to have more small compute modules, while TPUs have fewer but larger matrix compute modules:

| GPU | TPU | H100 Count | TPU v5p Count |
| --- | --- | ---: | ---: |
| SM | Tensor Core | 132 | 2 |
| Warp Scheduler | VPU slots | 528 | 8 |
| SMEM (L1 cache) | VMEM | 32MB | 128MB |
| Registers | Vector Registers (VRegs) | 32MB | 256kB |
| Tensor Core | MXU | 528 | 8 |

Key differences:

- **GPU** has more SMs, finer scheduling granularity, greater general-purpose capability;
- **TPU** has fewer TensorCore units, but larger matrix multiplication units, more oriented toward regular large matrix computations;
- **GPU** has warps, hardware schedules threads in warp units;
- **TPU** has no warps; execution is more block-level, which brings trade-offs between matrix multiplication and non-matrix computation;
- Differences between multi-GPU / multi-TPU also relate to how accelerators are interconnected.

**Summary**: Both GPU and TPU aim to make matrix multiplication very fast. GPU is more general-purpose, while TPU is more specialized for large-scale matrix computation.

### 1.4 Strengths and Limitations of the GPU Model

Early NVIDIA GPUs were primarily designed for graphics rendering, offering programmable shaders. Researchers discovered that matrix multiplication could be "disguised" as graphics computation—this was one of the early approaches to general-purpose computing on GPUs.

Modern GPUs now have dedicated matrix multiplication hardware, such as **Tensor Cores**. Tensor Cores are circuits specifically designed for matrix multiply-accumulate, so matmul speeds are much higher than ordinary floating-point operations.

<figure>
  <img src="matmul-vs-nonmatmul-flops.png" alt="Comparison of Matmul and non-Matmul FLOPs">
  <figcaption>As GPU architecture evolves, matmul FLOPs grow significantly faster than ordinary floating-point operations. </figcaption>
</figure>

This is very important for large models because the main computation in Transformers is large-scale matrix multiplication. In other words, modern GPUs are not "all computations are equally fast"—they are **especially good at matrix multiplication**.

However, compute capability grows faster than memory and interconnect bandwidth:

<figure>
  <img src="compute-vs-memory-scaling.png" alt="Comparison of compute capability and memory bandwidth scaling speed">
  <figcaption>Hardware FLOPs grow faster than DRAM bandwidth and interconnect bandwidth. </figcaption>
</figure>

This leads to a problem: compute units are getting faster, but data may not be delivered to them in time. That is, a slow GPU program may not be because "computation is not fast enough," but because "data movement is too slow." This is why many engineering optimizations today are focused on memory and data transfer.

**Summary (GPU Strengths)**:

- **Easily scalable**: If the task is large enough, throughput can be increased by adding more SMs;
- **Relatively easy to program**: The SIMT model allows programmers to write programs as if writing many threads, and the hardware handles organizing them into warps for execution;
- **Threads are lightweight**: GPUs can switch between a large number of threads. When some threads are waiting for memory, the SM can schedule other ready threads to continue execution;

## 2. How to Use GPU Better in Machine Learning

The next section discusses: when GPUs already have strong computational power, why machine learning programs can still run slowly, and how to avoid programs being limited by memory access and data transfer.

### 2.1 Roofline Model: What is Bottlenecking the Program?

Whether a machine learning workload is fast or not cannot be determined solely by the GPU's peak FLOPs. The GPU's compute units are powerful, but only if data can be delivered to them in time.

The **Roofline Model** aims to answer: Is a program currently limited by **compute capability** or **memory bandwidth**?

<figure>
  <img src="roofline-model.svg" alt="Roofline Model concept diagram">
  <figcaption>Core idea of the Roofline Model: When data reuse rate is low, the program is more likely to be limited by memory bandwidth; when data reuse rate is high enough, the program can approach the GPU's compute ceiling. </figcaption>
</figure>

The horizontal axis in the figure can be understood as **arithmetic intensity**, i.e., "how many computations can be done per data load." If an operator repeatedly reads data from **global memory**, but each piece of data is used only once or twice, it is likely **memory-bound**.

So the key question of this section is: **How to avoid a program being memory-bound?**

Intuitively, the optimization direction is to reuse data as many times as possible once it is loaded. For example, put data into **shared memory** or **cache** for reuse, reduce repeated accesses to global memory, and prevent the GPU's compute units from constantly waiting for data.

### 2.2 Control Divergence: Branching Also Slows Down GPU

GPUs can be slowed not only by memory but also by **control divergence**.

GPUs use the **SIMT (Single Instruction, Multiple Threads)** model: threads within the same **warp** usually execute the same instruction. If these threads take different paths in an `if / else` statement, the GPU often needs to execute the different branches separately.

<figure>
  <img src="control-divergence.png" alt="Control divergence schematic diagram">
  <figcaption>Control divergence diagram: When threads in the same warp diverge into different branches, the GPU executes these branches in batches rather than all threads advancing simultaneously. </figcaption>
</figure>

If some threads in a warp satisfy the condition and others do not, the GPU typically executes as follows:

- First execute the `if` branch's `A; B;`, only enabling threads that satisfy the condition; other threads are **masked**;
- Then execute the `else` branch's `X; Y;`, only enabling threads that do not satisfy the condition; the previous threads are masked;
- Finally, all threads rejoin and execute `Z;` together.

Thus, the condition check itself is not the problem; the real issue is **threads within the same warp diverging into different paths**. The more divergence, the lower the parallel efficiency.

### 2.3 Technique 1: Low Precision Computation—Move Less Data

#### 2.3.1. Low Precision Improves Arithmetic Intensity

The core idea of low precision computation is simple: **The fewer bits a number occupies, the less data needs to be moved from memory.**

For example, **Float 32 (FP32)** typically occupies 4 bytes, while **half precision (Float 16, FP16)** typically occupies 2 bytes.

#### 2.3.2. Low Precision Accelerates Matrix Multiplication

> 0.40625 in FP8 is represented as 0 0101 101;
>
> 0.40625 x 0.40625 = 0.1650390625
>
> 0.1650390625 cannot be exactly represented in FP8. The nearby FP8 numbers are 0.15625 (0 0100 010) and 0.171875 (0 0100 011), where 0.171875 is closer.
> 
> But 0.1650390625 can be represented in FP16; the FP16 representation is 0 01100 0101001000.

<figure>
  <img src="tensor-core-mixed-precision.png" alt="Tensor Core mixed-precision computation schematic diagram">
  <figcaption>Tensor Cores often use mixed precision. They can use FP16 or BF16 as input to increase throughput, but use FP32 for accumulation to minimize numerical errors. </figcaption>
</figure>

- **Suitable for 16-bit storage (FP16/BF16)**: Matrix multiplication, and most elementwise operations like ReLU, tanh, add, sub, mul;
- **Require higher precision (FP32/FP16)**: Summation where small numbers are added to large numbers, and reduction operations like sum, softmax, normalization;
- **Require larger numeric range (FP32/BF16)**: Operations whose output can be much larger than input, such as exp, log, pow, and loss functions.

#### 2.3.3. More Advanced Low Precision

**Microscaling FP8 (MXFP8)** works by not sharing a single scale factor across a whole block of data, but by having a smaller group of data share one **scaling factor**. This allows each group to have its own numeric range.

<figure>
  <img src="mxfp8-scaling-factors.png" alt="MXFP8 scaling factors schematic diagram">
  <figcaption>Normal FP8 can be understood as one large block of data (in the figure, 4 x 8 = 32 FP8 data) sharing one scale factor; MXFP8 assigns separate scale factors to smaller data blocks. </figcaption>
</figure>

Key points about MXFP8:

- Typically uses **E4M3**, i.e., 4 exponent bits, 3 mantissa bits;
- The scale factor itself can be FP8, e.g., **E8M0**;
- The scale factor is computed statistically. For example, FP8 E4M3 has a range of [-448, 448]. During actual training, some data may be too large for normal FP8 to cover; we can compute the absolute maximum of these values $x_{\textbf{absmax}} = \max(|x|)$, then derive a scale factor $s = \frac{x_{\textbf{absmax}}}{448}$ to adapt FP8's range to the actual data. When data are too small, normal FP8's precision may be insufficient; we can also compute a suitable scale factor to expand the FP8 range.
- Because scale factors are assigned to small blocks, matrix transposition becomes more complex: after transposition, the original grouping direction changes, requiring re-handling of scale factors.

The benefit is that compared to normal FP8, MXFP8 can more flexibly adapt to the numeric ranges of different data blocks, making training more stable.

<figure>
  <img src="mxfp8-training-practice.png" alt="MXFP8 training workflow schematic diagram">
  <figcaption>MXFP8 training flow diagram: Before matrix multiplication, weights, activations, or gradients are quantized to MXFP8; the computation result is typically output as BF16 or FP32. </figcaption>
</figure>

In actual training, MXFP8 does not directly convert everything to FP8. As shown in the figure:

- **Forward propagation (FPROP)**: Weights and activations are first quantized to MXFP8; after matrix multiplication, the output is **BF16**;
- **Data gradient in backpropagation (DGRAD)**: Inputs are quantized to MXFP8; output returns to **BF16**;
- **Weight gradient in backpropagation (WGRAD)**: Inputs are quantized to MXFP8; the output is typically retained as **FP32** for more stable high-precision weight updates.

**MXFP4** is more aggressive. 4 bits can represent very few numbers, so it relies more heavily on scaling factors to expand the representable range.

<figure>
  <img src="mxfp4-values.png" alt="Diagram of representable values for MXFP4">
  <figcaption>A single MXFP4 number can only represent a few discrete values, so it must rely on scaling factors to cover a larger numeric range. Currently, no model has been successfully trained using MXFP4. </figcaption>
</figure>

### 2.4 Technique 2: Operator Fusion—Move Data Back and Forth Less

<figure>
  <img src="operator-fusion-sin-cos-before.png" alt="Example computation graph of sin and cos">
  <figcaption>Without fusing, `sin(x)^2 + cos(x)^2` is split into multiple pointwise operations, each step may correspond to an independent CUDA kernel call. Each kernel call requires reading and writing data, incurring transfer overhead. </figcaption>
</figure>

The idea of operator fusion is to combine these consecutive small operations into a single kernel.

<figure>
  <img src="operator-fusion-before-after.png" alt="Comparison diagram before and after operator fusion">
  <figcaption>Before operator fusion, multiple pointwise operations are executed separately; after fusing, the compiler can combine them into one CUDA kernel. </figcaption>
</figure>

The benefits are straightforward:

- **Reduce kernel launch overhead**;
- **Reduce writing intermediate results back to global memory**;
- **Allow data to be reused multiple times in registers or cache**.

Thus, operator fusion also fundamentally addresses the memory-bound problem: move less data, do more computation. Simple fusions like this can often be done automatically by compilers, e.g., `torch.compile`.

### 2.5 Technique 3: Recomputation

During backpropagation, we need to store activations from the forward pass and compute the Jacobian matrices needed for backprop. Take a three-layer sigmoid as an example; without recomputation:

<figure>
  <img src="activation-storage-old.png" alt="Old forward pass and old backward pass">
  <figcaption>Without recomputation, 8 read or write operations are needed (low arithmetic intensity). </figcaption>
</figure>

<figure>
  <img src="activation-recompute-new.png" alt="New forward pass and new backward pass">
  <figcaption>With recomputation, only 6 read or write operations are needed (high arithmetic intensity). </figcaption>
</figure>

### 2.6 Technique 4: Memory Coalescing

<strong>Memory coalescing </strong> addresses the core question: can threads in a warp access a contiguous segment of memory together?

This relates to how <strong>Dynamic Random Access Memory (DRAM) </strong> is read. DRAM, i.e., GPU's <strong>global memory </strong>, typically does not read a single tiny piece; it reads a contiguous chunk of data at once in <strong>burst mode </strong>.

<figure>
  <img src="dram-burst-mode.png" alt="DRAM burst mode schematic diagram">
  <figcaption>DRAM reads data in burst sections. When accessing one location, neighboring locations in the same segment are also sent to the processor together. </figcaption>
</figure>

Thus, the GPU's most preferred access pattern is: threads within the same <strong>warp </strong> accessing adjacent addresses simultaneously. This way, multiple thread reads can be merged into fewer DRAM requests.

<figure>
  <img src="memory-coalescing-burst.png" alt="Memory coalescing schematic diagram">
  <figcaption>If threads in a warp access the same burst section, the hardware can merge these accesses into one DRAM request. </figcaption>
</figure>

Conversely, if threads in the same warp access scattered addresses, the GPU needs to issue more memory requests. The compute units may not even start computation before waiting for data.

Matrix multiplication often encounters this problem. For a <strong>row-major matrix </strong>, elements in a row are contiguous in memory. However, note that memory coalescing considers not "where a single thread goes sequentially," but "whether the addresses accessed by different threads in the same warp at the same time are contiguous."

<figure>
  <img src="matrix-memory-coalescing.png" alt="Memory coalescing in matrix multiplication schematic diagram">
  <figcaption>In a row-major matrix, if different threads access adjacent addresses at the same time, it is easier to achieve coalescing; if they access addresses with a large stride, coalescing is harder. </figcaption>
</figure>

Therefore, when writing GPU kernels, try to make threads in a warp access contiguous memory. This reduces DRAM requests and better utilizes global memory bandwidth.

### 2.7 Technique 5: Tiling

<strong>Tiling </strong> reorganizes threads and computation order to minimize accesses to <strong>global memory </strong>.

In ordinary <strong>matrix multiplication </strong>, the same input element may be repeatedly read from global memory, and these reads may not satisfy memory coalescing.

<figure>
  <img src="tiling-matmul-nontiled.png" alt="Repeated memory access in ordinary matrix multiplication">
  <figcaption>In ordinary matrix multiplication, the same element may be read repeatedly by multiple threads; the access pattern may also hinder memory coalescing. </figcaption>
</figure>

Tiling cuts the matrix into smaller <strong>tiles </strong> and first loads the needed tiles into <strong>shared memory (SHM) </strong>. Matrix multiplication proceeds in stages:

1. Load the two required tiles into SHM;
2. Use these two tiles to compute a portion of the output matrix;
3. Load the next set of tiles;
4. Repeat.

<figure>
  <img src="tiling-shared-memory-phases.png" alt="Tiling placing matrix blocks into shared memory">
  <figcaption>Tiling first loads reusable matrix blocks into shared memory. Subsequent reads access shared memory instead of repeatedly accessing global memory. </figcaption>
</figure>

The benefit is straightforward: repeated reads now happen in shared memory rather than global memory; also, global memory reads can be made more coalesced.

<figure>
  <img src="tiling-math.png" alt="Tiling reducing the number of global memory accesses">
  <figcaption>The core benefit of tiling is data reuse: after loading a small block of data, it is used multiple times within the tile. </figcaption>
</figure>

If the matrix size is `N` and the tile size is `T`:

- Without tiling: each input element is read from global memory `N` times;
- With tiling: each input element is read from global memory `N / T` times, then reused `T` times within each tile;
- So, global memory accesses are reduced by a factor of approximately `T`.

Tiling also has costs. The tile size may not evenly divide the matrix dimensions, causing some thread blocks to do a lot of ineffective work.

<figure>
  <img src="tiling-tile-size-utilization.png" alt="Low utilization due to tile size not divisible by matrix size">
  <figcaption>When matrix dimensions are not divisible by tile size, many empty slots appear in edge tiles, reducing GPU utilization. </figcaption>
</figure>

When choosing tile size, at least three factors must be considered:

1. Whether it facilitates <strong>memory coalescing </strong>;
2. Whether the shared memory capacity is sufficient;
3. Whether the tile size divides the matrix dimensions well.

Another detail is <strong>memory alignment </strong>. DRAM reads data in bursts; if tiles are aligned with burst sections, reads are faster; if not, more segments may need to be read.

<figure>
  <img src="tiling-memory-alignment.png" alt="Memory alignment in tiling">
  <figcaption>The left tile is aligned with burst sections, making reads cleaner; the right tile is not aligned, potentially spanning multiple burst sections. </figcaption>
</figure>

Therefore, some matrix dimensions inherently make coalesced access difficult, requiring <strong>padding </strong> to make the memory layout more GPU-friendly.

> My thoughts:
>
> The significance of tiling lies in the fact that the amount of matrix data exceeds the capacity of the GPU's internal high-speed storage (registers, shared memory, L1 cache), making it impossible to load all data at once. Tiling works by cutting the matrix into smaller blocks, loading only the currently needed block each time, thereby reusing data within the block and reducing accesses to global memory.

---

## References

[1] Stanford CS336, "Lecture 5 - GPUs," YouTube. [Online video]. Available: https://www.youtube.com/watch?v=izZba4UA7iY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&index=7.