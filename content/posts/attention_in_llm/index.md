---
title: "大语言模型中的注意力机制及其变体"
date: 2025-04-23T11:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "大语言模型"
    subseries: "架构与训练"
categories: ["大语言模型"]
tags: ["架构", "训练"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "注意力到底有几种呢？"
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: false # to disable highlightjs
disableShare: false
hideSummary: true
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "cover.png" # image path/url
    alt: "cover" # alt text
    caption: "cover" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## 稀疏注意力

{{< figure src="sparse_attentions.png" alt="稀疏注意力掩码示例" caption="稀疏注意力掩码示例" >}}

### 稀疏注意力的动机

标准自注意力的计算可以表述为：给定输入序列 \(X \in \mathbb{R}^{N \times d}\)，其中 \(N\) 为序列长度，\(d\) 为隐藏维度，经线性投影得到 Query、Key、Value：

\[Q = XW_Q, \quad K = XW_K, \quad V = XW_V\]

则自注意力输出为：

\[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V\]

其中 \(QK^T \in \mathbb{R}^{N \times N}\) 的每个元素 \((i, j)\) 表示第 \(i\) 个 token 对第 \(j\) 个 token 的注意力分数，\(M\) 为可选的注意力掩码矩阵。计算 \(QK^T\) 的复杂度为 \(O(N^2 d)\)，当 \(N\) 很大时（例如长文档、高分辨率图像），计算和显存开销都难以承受。

**稀疏注意力的基本思路**是限制每个 token 只与部分 token 交互，将复杂度从 \(O(N^2)\) 降至 \(O(Nk)\)，其中 \(k \ll N\)。记第 \(i\) 个位置的注意力范围（attendable set）为 \(\mathcal{A}(i)\)，则稀疏注意力掩码为：

\[M_{ij} = \begin{cases} 0 & \text{if } j \in \mathcal{A}(i) \\ -\infty & \text{otherwise} \end{cases}\]

不同的稀疏模式，本质上是对 \(\mathcal{A}(i)\) 的不同构造方式。

---

### 滑动窗口注意力 (Sliding Window)

**定义**：每个位置只与前后固定窗口内的邻居交互。给定窗口大小 \(w\)，位置 \(i\) 的注意力范围为：

\[\mathcal{A}_{\text{sw}}(i) = \{j \mid |i - j| \leq w\}\]

对应的掩码矩阵呈**带状**结构：

\[M^{\text{sw}}_{ij} = \begin{cases} 0 & |i-j| \leq w \\ -\infty & \text{otherwise} \end{cases}\]

复杂度：\(O(Nw)\)，当 \(w \ll N\) 时接近线性。

**特点**：由于自然语言具有强烈的局部性（语法搭配、邻近修饰关系），滑动窗口注意力用较低的计算代价捕获了主要的局部语义。Mistral 7B 即采用滑动窗口注意力（\(w=4096\)），在长序列推理时显著降低了 KV 缓存开销。但其局限性在于窗口外的远距离依赖（如长距离指代、跨段落的语义关联）被直接截断，无法建模。

---

### 分块注意力 (Block，即 Sparse Transformer 的 "Fixed")

与滑动窗口不同，分块注意力将序列按固定大小切分为若干个**互不重叠的块（block）**，每个位置只与同一块内的所有 token 交互，不同块之间完全隔离。设块大小为 \(b\)，位置 \(i\) 所属的块索引为 \(\lfloor i/b \rfloor\)，则注意力范围为：

\[\mathcal{A}_{\text{block}}(i) = \{j \mid \lfloor j/b \rfloor = \lfloor i/b \rfloor\}\]

掩码矩阵呈**分块对角**结构：

\[M^{\text{block}}_{ij} = \begin{cases} 0 & \lfloor i/b \rfloor = \lfloor j/b \rfloor \\ -\infty & \text{otherwise} \end{cases}\]

复杂度：\(O(Nb)\)。由于块之间互不重叠，可以用分块矩阵乘法高效实现。

**对比滑动窗口**：滑动窗口是"窗口随 token 移动"，分块是"窗口固定不动"。分块的缺点是块边界处的 token 相邻却无法互相关注。

---

### 跨步注意力 (Strided)

**定义**：Block Attention 局限于同一块内，Strided Attention 则以均匀间隔覆盖全局，使得同一个块内的不同 token 各自连接不同块的 token。给定步长 \(l\)，位置 \(i\) 的注意力范围为：

\[\mathcal{A}_{\text{strided}}(i) = \{j \mid (i-j) \bmod l = 0\}\]

即第 \(i\) 个位置关注所有与其下标差能被步长 \(l\) 整除的位置。对应的掩码矩阵呈**交叉条纹**结构：

\[M^{\text{strided}}_{ij} = \begin{cases} 0 & (i-j) \bmod l = 0 \\ -\infty & \text{otherwise} \end{cases}\]

复杂度：\(O(N^2/l)\)，当 \(l\) 与 \(\sqrt{N}\) 同阶时约为 \(O(N\sqrt{N})\)。

**特点**：Strided Attention 以稀疏采样的方式实现了跨块的信息交互。其代价是局部细节可能因为步长间隔而丢失——两个相邻 token 如果下标差不满足取模条件，在 Strided 层中完全无法关注彼此。

---

### 小结

| 模式 | 注意力范围 \(\mathcal{A}(i)\) | 复杂度 | 代表工作 |
|------|---------------------------|--------|---------|
| 滑动窗口 (Sliding Window) | \(\{j : \|i-j\| \leq w\}\) | \(O(Nw)\) | Mistral 7B, Longformer, Gemma 2 |
| 分块 (Block) | \(\{j : \lfloor j/b \rfloor = \lfloor i/b \rfloor\}\) | \(O(Nb)\) | Sparse Transformer |
| 跨步 (Strided) | \(\{j : (i-j) \bmod l = 0\}\) | \(O(N^2/l)\) | Sparse Transformer |

目前滑动窗口注意力比较常见，通常和全局注意力结合使用（例如 GEMMA 系列采用 5 滑动窗口（local）+ 1 Group Query Attention（global））。

---

## 线性注意力（Linear Attention）

[线性注意力和 Mamba（Linear Attention and Mamba)](../mamba/index.md)

## Flash Attention

### 问题：注意力不是计算瓶颈，是显存带宽瓶颈

标准自注意力的计算量为 \(O(N^2 d)\)，但现代 GPU 的 FLOPS 远大于显存带宽（HBM 带宽增速远落后于计算增速）。实际运行时，GPU 需要反复将 \(N \times N\) 的注意力矩阵写入 HBM 再读出：

\[S = QK^T \in \mathbb{R}^{N \times N}\]

\[P = \text{softmax}(S) \in \mathbb{R}^{N \times N}\]

\[O = PV \in \mathbb{R}^{N \times d}\]

其中 \(S\) 和 \(P\) 都是 \(O(N^2)\) 的中间结果，每次 softmax 和矩阵乘都需要在 HBM 上完成读写。**注意力操作的瓶颈不是 FLOP，而是 IO。**

### 关键观察：GPU 存储层次

GPU 有两级显存：

| 级别 | 容量 | 带宽 | 延迟 |
|------|------|------|------|
| HBM（高带宽显存） | 大（80 GB, H100） | 低（~3 TB/s） | 高 |
| SRAM（片上缓存） | 小（~20 MB, H100） | 高（~19 TB/s） | 低 |

标准注意力在 HBM 上反复读写 \(O(N^2)\) 数据——HBM 带宽成为瓶颈，而 SRAM 被浪费。

**Flash Attention 的核心思想**：将 \(Q, K, V\) 切分为块，每次只加载一个块到 SRAM 上完成局部计算，只将最终输出 \(O\) 写回 HBM，中间结果 \(S, P\) 不落盘。

### 在线 Softmax：平铺的数学基础

标准 softmax 需要完整行才能计算（先求全局最大值、再归一化）。Flash Attention 的关键数学工具是 **online softmax**：通过维护两个运行统计量，在逐块处理时增量更新。

对第 \(i\) 行，将 \(K^T\) 沿列方向切分为 \(K_1, \dots, K_{T_c}\) 共 \(T_c\) 块。在处理第 \(j\) 个块时：

\[m^{(j)} = \max(m^{(j-1)}, \max(Q_i K_j))\]

\[\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot \ell^{(j-1)} + \sum e^{Q_i K_j - m^{(j)}}\]

\[O_i^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot O_i^{(j-1)} + e^{Q_i K_j - m^{(j)}} V_j\]

其中 \(m\) 为当前运行最大值，\(\ell\) 为当前运行 softmax 分母。处理完所有块后，\(O_i = O_i^{(T_c)} / \ell^{(T_c)}\)。

**为什么等价于标准 softmax**：从数值稳定的 softmax 出发：

\[
\text{softmax}(s_1, \dots, s_N) = \frac{e^{s_j - m}}{\sum_k e^{s_k - m}},\quad m = \max(s_1, \dots, s_N)
\]

其中 \(s_k = Q_i \cdot K_k / \sqrt{d_k}\)，是查询位置 \(i\) 与键位置 \(k\) 的标量注意力分数。整行 \(\{s_1, \dots, s_N\}\) 之间彼此比较，每个查询位置独立做一次。

该式等价于先对每个 \(s_j\) 计算 \(a_j = e^{s_j - m}\)，再令 \(p_j = a_j / \sum a_k\)，最终输出 \(O = \sum p_j V_j\)。

现在假设只处理了前 \(B\) 个 token，已得到：
\[
m_{\text{old}} = \max_{k \leq B} s_k,\quad L_{\text{old}} = \sum_{k \leq B} e^{s_k - m_{\text{old}}},\quad O_{\text{old}} = \sum_{k \leq B} e^{s_k - m_{\text{old}}} V_k
\]

当新一批 token（第 \(B+1\) 到 \(B'\) 个）到达，设这批内部的临时统计量为：
\[
m_{\text{new}} = \max_{B < k \leq B'} s_k,\quad L_{\text{new}}' = \sum_{B < k \leq B'} e^{s_k - m_{\text{new}}},\quad O_{\text{new}}' = \sum_{B < k \leq B'} e^{s_k - m_{\text{new}}} V_k
\]

此时全局最大值变为 \(m' = \max(m_{\text{old}}, m_{\text{new}})\)。分两种情况：

**情况一**：\(m_{\text{new}} \leq m_{\text{old}}\)，最大值不变（\(m' = m_{\text{old}}\)）。旧结果无需修正，新批次以 \(m_{\text{old}}\) 为基准重新计算指数即可：
\[
L' = L_{\text{old}} + \sum_{B < k \leq B'} e^{s_k - m_{\text{old}}},\quad
O' = O_{\text{old}} + \sum_{B < k \leq B'} e^{s_k - m_{\text{old}}} V_k
\]

**情况二**：\(m_{\text{new}} > m_{\text{old}}\)，最大值被刷新（\(m' = m_{\text{new}}\)）。关键步骤——旧结果需要以新的最大值为基准重新定标。利用恒等式：
\[
e^{s_k - m'} = e^{s_k - m_{\text{old}}} \cdot e^{m_{\text{old}} - m'}
\]
因此每一项旧指数、旧分母、旧输出都需要乘以折扣因子 \(e^{m_{\text{old}} - m'}\)（其中 \(m_{\text{old}} - m' < 0\)，因子小于 1）：
\[
L' = e^{m_{\text{old}} - m'} \cdot L_{\text{old}} + \sum_{B < k \leq B'} e^{s_k - m'},\quad
O' = e^{m_{\text{old}} - m'} \cdot O_{\text{old}} + \sum_{B < k \leq B'} e^{s_k - m'} V_k
\]

处理完所有块后，\(O_i = O_i^{(T_c)} / \ell^{(T_c)}\) 完成最终归一化。


### 复杂度分析

| | 标准注意力 | Flash Attention |
|------|---------|---------|
| 计算量 | \(O(N^2 d)\) | \(O(N^2 d)\) （相同） |
| HBM 读取 | \(O(N^2)\) （每次读写完整 N×N 矩阵） | \(O(N^2 d^2 / M)\) （不落 N×N 矩阵） |
| 显存占用 | \(O(N^2)\) | \(O(Nd)\) |
| 数值结果 | — | 与标准注意力完全一致 |

Flash Attention 没有改变注意力的算术结果，只改变了 IO 模式。它不是近似方法——输出与标准 softmax attention 在数值上等价。

### 反向传播：重计算换显存

标准注意力的反向传播需要前向保存的 softmax 矩阵 \(P \in \mathbb{R}^{N \times N}\) 来计算梯度。梯度公式为：

\[
\frac{\partial L}{\partial Q} \propto dP \cdot K,\quad
\frac{\partial L}{\partial K} \propto dP^T \cdot Q,\quad
\frac{\partial L}{\partial V} \propto P^T \cdot dO
\]

其中 \(dO\) 和 \(dP\) 为上游传入的对 \(O\) 和 \(P\) 的梯度。存 \(P\) 意味着反向又回到了 \(O(N^2)\) 显存——前向省掉的，反向全还回去。

**Flash Attention 的解决方案**：前向不存 \(P\)，只保留输出 \(O \in \mathbb{R}^{N \times d}\) 以及每行的统计量 \(m_i, \ell_i \in \mathbb{R}^{N}\)（共 \(O(Nd)\)）。反向时，利用这些元数据分块精确重建 \(P\)：

\[
P_{ij} = e^{Q_i K_j - m_i} / \ell_i
\]

由于 \(m_i, \ell_i\) 恰好是做 softmax 所需的全部信息，配合原始的 \(Q_i, K_j\)，可以在任意时刻精确恢复 \(P\) 的一块。随后按标准链式法则分块计算梯度，整个过程与前向一样做分块平铺，确保中间结果不落地 \(N \times N\) 矩阵。

**代价**：每个 training step 中需要额外执行一次前向计算（重计算 \(P\)），用约 30% 的额外 FLOP 换取 \(O(N^2) \to O(Nd)\) 的显存节省。对于长序列训练，这一取舍是决定性的——没有它，数十万 token 的上下文根本无法在单张 GPU 上训练。

### 后续迭代

- **Flash Attention 2 (Dao, 2023)**：优化了 tile 调度顺序，减少线程块间的同步开销；针对 A100/H100 的非对称 SM 结构改进。在 H100 上达到理论峰值吞吐的 75%。
- **Flash Attention 3 (Shah et al., 2024)**：利用 Hopper 架构的 TMA（Tensor Memory Accelerator）和 WGMMA 指令异步传输数据，对 FP8 实现端到端加速，针对长上下文推理进一步优化。

---

## 参考文献

1. Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). *Generating Long Sequences with Sparse Transformers.* arXiv:1904.10509.
2. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.
3. Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691.
4. Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* arXiv:2407.08608.
5. Gemma Team, Google DeepMind. (2025). *Gemma 3 Technical Report.* arXiv:2503.19786.
6. Gemma Team, Google DeepMind. (2026). *Gemma 4: Frontier Open Models Under Apache 2.0.* Google AI Blog, April 2026.
7. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.

---

