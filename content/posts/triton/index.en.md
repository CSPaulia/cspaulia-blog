---
title: "Introduction to Triton"
date: 2026-07-14T10:00:00+08:00
series:
  main: "Large Language Model"
  subseries: "System and Hardware"
draft: false
categories: ["大语言模型", "系统"]
tags: ["Triton", "GPU", "CUDA", "并行计算", "Kernel"]
author: "CSPaulia"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "CS336 Lecture 6 Study Notes."
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
    image: "<image path/url>"
    alt: "Triton cover"
    caption: "Triton"
    relative: true
    hidden: false
    hiddenInList: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

## 1. Why Do We Need Triton?

In the [previous article]({{< ref "/posts/gpus" >}}), we discussed GPU hardware architecture and optimization techniques. However, in practice, writing a CUDA kernel directly has a high barrier:

- You need to manage low-level details such as threads, warps, and shared memory;
- You need to manually handle issues like memory coalescing, bank conflicts, and occupancy;
- Writing an efficient matrix multiplication may require hundreds of lines of CUDA code.

**Triton** (developed by OpenAI) offers a compromise: it uses **thread blocks** as the unit of programming, rather than individual threads. You only need to describe what each block does, and Triton’s compiler handles translating it into efficient PTX code.

In simple terms:

| | CUDA | Triton |
|---|---|---|
| Programming Granularity | What each thread does | What each thread block does |
| Control | Extremely fine-grained | Powerful enough (especially for beginners) |
| Manual Management Required | shared memory, warps, coalescing… | Mostly handled by the compiler |
| Core Idea | Thread-level logic | **load → compute → store** (load data into shared memory, compute, then write back to global memory) |

## 2. Golden Rule of Optimization: Benchmark First, Then Profile

Before writing any Triton kernel, it’s worth remembering a golden rule:

> 1. Benchmark and profile your code
> 2. Make a change
> 3. Benchmark and profile again
> 4. Repeat…

**Benchmarking** measures the end-to-end wall-clock time. It only tells you "how long this code took," but not where the time was spent. Typical uses of benchmarking are:

- Comparing which implementation is faster (e.g., naive vs. builtin vs. compiled);
- Observing how performance scales with size (e.g., whether time grows linearly or cubically as matrix dimensions increase).

**Profiling**, on the other hand, tells you which kernels the time was spent on, allowing you to see specific bottlenecks.

### 2.1. Benchmarking: Comparing Which Implementation is Faster

Here’s an example benchmark for matrix multiplication:

```python
import torch

def run_operation2(dim: int, operation):
    """创建两个 dim×dim 随机矩阵，返回一个可调用的操作函数"""
    x = torch.randn(dim, dim, device="cuda")
    y = torch.randn(dim, dim, device="cuda")
    return lambda: operation(x, y)

def benchmark(run, num_warmups: int = 1, num_trials: int = 3) -> float:
    """运行 `run` 多次并返回平均耗时（毫秒）"""

    # 1. Warmup：第一次运行往往偏慢（JIT 编译、缓存未命中），
    #    我们关心的是稳态性能，所以先预热几次
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # 等待所有 CUDA 线程完成（重要！）

    # 2. 正式计时
    times: list[float] = []
    for trial in range(num_trials):
        # 使用 CUDA Event 获得精确的 GPU 端计时
        # （避免把 CPU 端的 launch overhead 也算进去）
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()   # 记录开始时间
        run()                  # 实际执行计算
        end_event.record()     # 记录结束时间

        torch.cuda.synchronize()  # 等待 GPU 完成
        times.append(start_event.elapsed_time(end_event))

    return sum(times) / len(times)

# 对 1024×1024 矩阵乘法做 benchmark
matmul = run_operation2(dim=1024, operation=lambda a, b: a @ b)
avg_time = benchmark(matmul)

# 观察时间随维度的 scaling
for dim in [256, 512, 1024, 2048, 4096, 8192]:
    op = run_operation2(dim=dim, operation=lambda a, b: a @ b)
    t = benchmark(op)
    print(f"dim={dim}: {t:.3f} ms")
```

Key points:

- **Warmup**: The first launch of a GPU kernel may involve JIT compilation and cache initialization; not warming up can make the first measurement significantly larger;
    ```python
    for _ in range(num_warmups):
        run()
    ```
- **`torch.cuda.synchronize()`**: GPU execution is asynchronous — the CPU issues the kernel instruction and does not wait for the GPU to finish. Without synchronizing, you only measure the time it took the CPU to issue the instruction, not the actual GPU computation time;
    ```python
    torch.cuda.synchronize()
    ```
- **CUDA Events**: `torch.cuda.Event` provides precise GPU-side timestamps, avoiding counting CPU-side kernel launch overhead;
    ```python
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    run()
    end_event.record()
    ```
- **Multiple measurements**: Taking the average over several measurements allows you to observe variance (e.g., if some kernel launch times are unstable).

For matrix multiplication, you would observe: for small dimensions, the time is roughly constant (dominated by kernel launch overhead); as dimensions increase, time grows **cubically** (O(N³): M×K×N, when M=K=N).

### 2.2. Profiling: Seeing Where the Time is Spent

PyTorch has a built-in profiler that can directly show the underlying CUDA kernel invocations:

```python
from torch.profiler import ProfilerActivity

def profile(run, num_warmups: int = 1):
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        run()
        torch.cuda.synchronize()

    # 按 CUDA 时间排序，显示前 10 个 kernel
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        max_name_column_width=100,
        row_limit=10
    )
    return table

# 对一个简单的 add 操作做 profile
add_op = run_operation2(dim=2048, operation=lambda a, b: a + b)
print(profile(add_op))

# 对矩阵乘法做 profile
matmul_op = run_operation2(dim=2048, operation=lambda a, b: a @ b)
print(profile(matmul_op))
```

Taking the `dim=2048` matrix multiplication as an example, the actual profiler output looks approximately like:

```
                                                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls


cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bias_f32_relu         0.00%       0.000us         0.00%       0.000us       0.000us     329.345us       100.00%     329.345us     329.345us             1

                                                                        cuLaunchKernelEx         0.95%      27.515us        99.46%       2.867ms       2.867ms       0.000us         0.00%       0.000us       0.000us             1

                                                                 Activity Buffer Request        98.50%       2.839ms        98.50%       2.839ms       2.839ms       0.000us         0.00%       0.000us       0.000us             1

                                                                   cudaDeviceSynchronize         0.54%      15.626us         0.54%      15.626us       7.813us       0.000us         0.00%       0.000us       0.000us             2

Self CPU time total: 2.883ms
Self CUDA time total: 329.345us
```

From the output, we can observe:

- **The entire matmul consists of only one CUDA kernel**, with all computation done in that kernel and no extra HBM reads/writes — this indicates that cuBLAS’s implementation already does good kernel fusion and tiling;
- **Different dimensions may trigger different kernel implementations**: matmuls with `dim=128` and `dim=2048` may call kernels with different names. cuBLAS automatically selects the optimal strategy (e.g., different tile sizes) based on matrix size, GPU architecture, etc.;
- **CUDA kernel names reveal a lot of implementation details**. Take this kernel name as an example:
  - `cutlass3x`: Based on the CUTLASS 3.x library (NVIDIA’s CUDA linear algebra template library);
  - `sm100`: Blackwell architecture (B200);
  - `simt_sgemm`: SIMT single-precision general matrix multiply (sgemm);
  - `f32_f32_f32_f32_f32`: 5 float32 precision parameters (A, B, C and accumulator precision, etc.);
  - `64x64x16`: Output tile size M=64, N=64, K=16;
  - `1x1x1`: Warp-level tile partitioning;
  - `3`: Possibly indicates a 3-stage software pipeline;
  - `nnn`: Transpose modes for matrices A, B, C, all non-transposed;
  - `align1`: Memory alignment;
  - `bias_f32_relu`: Supports float32 bias and ReLU fusion — meaning one kernel completes matmul + bias + ReLU.

### 2.3 A Complete Example: Three GeLU Versions Compared

Using benchmarking + profiling to compare different implementations of the same operation is the most illustrative approach. Take the GeLU activation function as an example:

```python
# 1. Naive PyTorch 实现（非融合，多个独立 kernel）
def naive_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

# 2. PyTorch 内置实现（融合 kernel）
def builtin_gelu(x: torch.Tensor):
    return torch.nn.functional.gelu(x, approximate="tanh")

# 3. 用 torch.compile 编译 naive 版本（让编译器尝试融合）
compiled_gelu = torch.compile(naive_gelu)

def run_operation1(dim: int, operation):
    """创建单个 dim×dim 随机矩阵，返回一个可调用的操作函数"""
    x = torch.randn(dim, dim, device="cuda")
    return lambda: operation(x)

# Benchmark
naive_time   = benchmark(run_operation1(dim=16384, operation=naive_gelu))
builtin_time = benchmark(run_operation1(dim=16384, operation=builtin_gelu))
compiled_time = benchmark(run_operation1(dim=16384, operation=compiled_gelu))

print(f"naive_gelu:   {naive_time:.3f} ms")
print(f"builtin_gelu: {builtin_time:.3f} ms")
print(f"compiled_gelu:{compiled_time:.3f} ms")
```

Benchmark results (B200, dim=16384):

```
naive_gelu:   3.758 ms
builtin_gelu: 0.667 ms
compiled_gelu:0.939 ms
```

The naive version is slowest because it constantly reads and writes intermediate results from/to HBM; the builtin is fastest because it fuses all operations into a single kernel; the compiled version is slower than builtin because the compiler is not yet intelligent enough to fully fuse everything.

## 3. Triton Programming Model

### 3.1 Triton Introduction

The core difference between CUDA (developed by NVIDIA) and Triton (developed by OpenAI) lies in **programming granularity**:

| | CUDA | Triton |
|---|---|---|
| Developer | NVIDIA | OpenAI |
| Programming Granularity | Specify what each **thread** does | Specify what each **thread block** does |
| Advantages | Extremely fine-grained control | Powerful enough (especially for beginners), compiler auto-optimizes |
| Disadvantages | Need to manually manage shared memory, warps, coalescing, etc. | Less flexible than CUDA; some extreme optimizations not possible |

Triton’s programming framework can be summarized in three steps:

> **load → compute → store**
>
> Load data from HBM into shared memory → compute in shared memory / registers → write results back to HBM.

### 3.2 Triton: GeLU Implementation Example

GeLU is an element-wise operation — each output element depends on exactly one input element, and threads do not need to communicate. This is the simplest Triton kernel, perfectly embodying the "load → compute → store" three-stage structure.

#### Launch Function

```python
import triton
import triton.language as tl

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda and x.is_contiguous()

    y = torch.empty_like(x)

    num_elements = x.numel()
    BLOCK_SIZE = 1024                         # 每个 block 处理 1024 个元素
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)  # 向上取整，确保覆盖所有元素

    # 启动 kernel：(num_blocks,) 是 1D grid，每个 block 有 BLOCK_SIZE 个 thread
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    return y
```

`(num_blocks,)` defines a 1D grid. For tasks requiring 2D partitioning (e.g., matrix multiplication), you can use `(M, N)` to define a 2D grid.

#### Kernel Body

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # ===== 1. 计算索引：当前 block 负责哪些元素 =====
    pid = tl.program_id(axis=0)                  # 我是第几个 block？
    start = pid * BLOCK_SIZE                      # 本 block 的起始位置
    offsets = start + tl.arange(0, BLOCK_SIZE)    # 本 block 内每个 thread 的偏移

    mask = offsets __HTMLTAG_1__ torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y
```

#### Key Points

- **`for start in range(0, N, BLOCK_SIZE)`**: Loop over tiles; each iteration loads a block from HBM and accumulates into registers. This is the core mechanism of cross-tile reduction — data is loaded in batches, and partial sums stay in registers;
- **`acc = tl.zeros([BLOCK_SIZE])`**: Each thread allocates one accumulator. It is not shared — each thread independently accumulates the data it is responsible for;
- **Two reductions**: The first is implicit — each thread’s own `acc += x` is done in registers (once per tile); the second is explicit — `tl.sum(acc, axis=0)` merges the partial sums from BLOCK_SIZE threads into a single scalar.

This is essentially a "baby version" of matrix multiplication tiling — splitting data into chunks, loading them in a loop, accumulating in registers, and finally merging. Next, extending the 1D `for` loop to 2D gives us tiled matrix multiplication.

### 3.5 Triton: Matrix Multiplication + ReLU (2D Tiling + Kernel Fusion)

Matrix multiplication is a core operation in deep learning. `C = A @ B`, A is M×K, B is K×N.

**Naive approach** (without tiling):

- Fix output position (m, n), iterate over k: read A[m, k], read B[k, n], multiply-accumulate, write C[m, n]
- Total accesses: M×K×N HBM reads, M×N writes
- Arithmetic intensity: O(1) — almost certainly **memory-bound**

**Ideal approach** (put A and B entirely into shared memory):

- Total accesses: M×K + K×N reads, M×N writes
- Arithmetic intensity: O(N)
- Problem: A and B are too large to fit in shared memory

**Tiling (compromise)**:

- Partition C into output tiles (each tile corresponds to one thread block)
- For each output tile, loop along the K dimension:
  1. Load row tile of A and column tile of B into shared memory
  2. Do matrix multiplication within the tile (`tl.dot`, using Tensor Cores)
  3. Accumulate into partial sum
- Write back the output tile
- Arithmetic intensity: O(tile_size)

![GEMM Tiled](gemm_tiled.png)

If ReLU is additionally needed, it can be **directly fused into the same kernel**, saving an extra HBM read/write.

#### Launch Function

First, review a detail: matrices are stored linearly in memory. `stride_row` and `stride_col` determine how to compute the linear index from (row, col):

```python
x = torch.tensor([[0., 1, 2, 3],
                  [4, 5, 6, 7]])   # 2×4, row-major: stride_row=4, stride_col=1
# x[1][2] = x_ptr + 1*4 + 2*1 = 6
```

```python
def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device)

    # Tile 尺寸
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    # 2D grid：每个 block 负责 C 的一个输出 tile
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
```

Here we use a **2D grid**: `program_id(0)` selects the row tile of C, `program_id(1)` selects the column tile of C.

#### Kernel Body

```python
@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,     # A 的行 stride 和列 stride
    stride_bk, stride_bn,     # B 的行 stride 和列 stride
    stride_cm, stride_cn,     # C 的行 stride 和列 stride
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 当前 block 负责 C 的第 (pid_m, pid_n) 个 tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 索引范围
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    indices_k = tl.arange(0, BLOCK_K)                      # [BLOCK_K]

    # 初始指针：A 的行 tile，B 的列 tile
    a_ptrs = (a_ptr + indices_m[:, None] * stride_am
                    + indices_k[None, :] * stride_ak)      # [BLOCK_M, BLOCK_K]
    b_ptrs = (b_ptr + indices_k[:, None] * stride_bk
                    + indices_n[None, :] * stride_bn)      # [BLOCK_K, BLOCK_N]

    # 累加器（在寄存器中）
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 1. Load + Compute：沿 K 方向遍历 tile
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs,
                    mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N),
                    other=0.0)

        acc += tl.dot(a, b)                       # Tensor Core 加速

        a_ptrs += BLOCK_K * stride_ak             # 前进到下一个行 tile
        b_ptrs += BLOCK_K * stride_bk             # 前进到下一个列 tile

    # 2. Kernel Fusion：ReLU
    acc = tl.maximum(acc, 0.0)

    # 3. Store：写回输出 tile
    c_ptrs = (c_ptr + indices_m[:, None] * stride_cm
                    + indices_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc,
             mask=(indices_m[:, None] < M) & (indices_n[None, :] < N))
```

#### Key Points

- **2D grid**: `program_id(0)` and `program_id(1)` correspond to the row and column of the output tile, differing from the 1D grid used for GeLU/Softmax;
- **`tl.dot(a, b)`**: Triton’s matrix multiplication primitive; the compiler automatically maps it to Tensor Cores, no need to manually call MMA instructions;
- **`for` loop along K**: Same idea as the `for start in range(0, N, BLOCK_SIZE)` loop in Row Sum — Row Sum is 1D slicing, matrix multiplication is 2D slicing (row tile of A × column tile of B);
- **`[:, None]` and `[None, :]`**: Broadcasting semantics used to construct 2D pointer matrices. `[BLOCK_M] + [BLOCK_K]` → `[BLOCK_M, BLOCK_K]`;
- **Kernel Fusion**: After `tl.dot`, directly apply `tl.maximum(acc, 0.0)` to fuse ReLU into the same kernel. If written separately, `a @ b` writes back to HBM, then reads back for ReLU, then writes again — incurring two extra HBM accesses.

## 4. Summary

- Understand the programming models (PyTorch, Triton, PTX) to ensure correctness;
- Understand hardware (SM, warp, occupancy, bank conflicts, etc.) to optimize performance;
- Benchmark helps you understand scaling;
- Profiling lets you see what is executed and how long it takes;
- Core of Triton: think in terms of thread blocks (load into shared memory → compute (fusion) → write back to HBM);
- Four examples: GeLU (element-wise), Softmax (row-wise reduction), Row Sum (first taste of tiling), Matrix Multiplication (full tiling).

---

## References

[1] Stanford CS336, "Lecture 6 - Benchmarking, Profiling, and Writing Kernels," Spring 2026. [Online]. Available: https://cs336.stanford.edu/lectures/?trace=lecture_06.

[2] Triton Documentation, "Fused Softmax Tutorial." [Online]. Available: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html.