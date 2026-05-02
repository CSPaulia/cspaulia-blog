---
title: "大语言模型中的注意力机制及其变体"
date: 2026-04-23T11:30:03+08:00
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

## 分组查询注意力（Grouped Query Attention，GQA）和多查询注意力（Multi-Query Attention，MQA）

见[博客](../transformer_in_LLM/index.md#mqa-gqa)。

## 稀疏注意力（Sparse Attention） {#sparse-attention}

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

### 滑动窗口注意力 (Sliding Window) {#sliding-window}

**定义**：每个位置只与前后固定窗口内的邻居交互。给定窗口大小 \(w\)，位置 \(i\) 的注意力范围为：

\[\mathcal{A}_{\text{sw}}(i) = \{j \mid |i - j| \leq w\}\]

对应的掩码矩阵呈**带状**结构：

\[M^{\text{sw}}_{ij} = \begin{cases} 0 & |i-j| \leq w \\ -\infty & \text{otherwise} \end{cases}\]

复杂度：\(O(Nw)\)，当 \(w \ll N\) 时接近线性。

**特点**：由于自然语言具有强烈的局部性（语法搭配、邻近修饰关系），滑动窗口注意力用较低的计算代价捕获了主要的局部语义。Mistral 7B 即采用滑动窗口注意力（\(w=4096\)），在长序列推理时显著降低了 KV 缓存开销。但其局限性在于窗口外的远距离依赖（如长距离指代、跨段落的语义关联）被直接截断，无法建模。

---

### 分块注意力 (Block，即 Sparse Transformer 的 "Fixed") {#block-attention}

与滑动窗口不同，分块注意力将序列按固定大小切分为若干个**互不重叠的块（block）**，每个位置只与同一块内的所有 token 交互，不同块之间完全隔离。设块大小为 \(b\)，位置 \(i\) 所属的块索引为 \(\lfloor i/b \rfloor\)，则注意力范围为：

\[\mathcal{A}_{\text{block}}(i) = \{j \mid \lfloor j/b \rfloor = \lfloor i/b \rfloor\}\]

掩码矩阵呈**分块对角**结构：

\[M^{\text{block}}_{ij} = \begin{cases} 0 & \lfloor i/b \rfloor = \lfloor j/b \rfloor \\ -\infty & \text{otherwise} \end{cases}\]

复杂度：\(O(Nb)\)。由于块之间互不重叠，可以用分块矩阵乘法高效实现。

**对比滑动窗口**：滑动窗口是"窗口随 token 移动"，分块是"窗口固定不动"。分块的缺点是块边界处的 token 相邻却无法互相关注。

---

### 跨步注意力 (Strided) {#strided-attention}

**定义**：Block Attention 局限于同一块内，Strided Attention 则以均匀间隔覆盖全局，使得同一个块内的不同 token 各自连接不同块的 token。给定步长 \(l\)，位置 \(i\) 的注意力范围为：

\[\mathcal{A}_{\text{strided}}(i) = \{j \mid (i-j) \bmod l = 0\}\]

即第 \(i\) 个位置关注所有与其下标差能被步长 \(l\) 整除的位置。对应的掩码矩阵呈**交叉条纹**结构：

\[M^{\text{strided}}_{ij} = \begin{cases} 0 & (i-j) \bmod l = 0 \\ -\infty & \text{otherwise} \end{cases}\]

复杂度：\(O(N^2/l)\)，当 \(l\) 与 \(\sqrt{N}\) 同阶时约为 \(O(N\sqrt{N})\)。

**特点**：Strided Attention 以稀疏采样的方式实现了跨块的信息交互。其代价是局部细节可能因为步长间隔而丢失——两个相邻 token 如果下标差不满足取模条件，在 Strided 层中完全无法关注彼此。

---

### DeepSeek 稀疏注意力（DeepSeek Sparse Attention，DSA） {#dsa}

假设模型正在处理一个 128K token 的长上下文，当前最后一句问题是：

```text
“请根据前文第 3 份合同里的违约条款回答……”
```

普通 attention：

```text
当前 token 直接和前面 128K 个 token 全部算 attention
```

DSA：

```text
1. 轻量化索引器（lighting Indexer）快速阅读历史 token
2. 给每一个历史 token 计算一个相关性分数
3. 选取 top-k 相关 token 作为当前 token 的注意力范围
4. 主注意力模块对这 top-k token 进行全注意力计算
```

DSA 的主要思想与上文中所有稀疏注意力相同，都是缩小当前查询 token 的交互 token 数量。对于长度为 $N$ 的查询序列，其交互 token 数量从 $N$ 降至 $k$，复杂度从 $O(N^2)$ 降至 $O(Nk)$。

#### 轻量化索引器（Lightweight Indexer）

轻量化索引器的核心是计算当前查询 token $\mathbf{h}_t \in \mathbb{R}^d$ 与历史 token $\mathbf{h}_s \in \mathbb{R}^d$ 的相关性分数。在 [DeepSeek V3.2](https://arxiv.org/pdf/2512.02556) 中，索引器被定义为：

$$
I_{t,s}=\sum_{j=1}^{H^I} w^I_{t,j} \cdot \mathrm{ReLU}\left(\mathbf{q}^I_{t,j} \cdot \mathbf{k}^I_{s}\right).
$$

其中 $H^I$ 是索引器的头（head）数。$w^I_{t,j}$ 是第 $j$ 个头的权重，来自于当前查询 token $\mathbf{h}\_t$。$\mathbf{q}^I_{t,j}$ 和 $\mathbf{k}^I_{s}$ 分别是查询 token 和历史 token 在索引器中的查询向量和键向量。

根据 \(I_{t,s}\) 的值，模型选取 top-k 相关的历史 token 进行后续的全注意力计算。当前查询 token \(\mathbf{h}_t\) 的交互目标从原来的 KV 序列中的所有 token $C$ 变为选取的 top-k token 集合 \(\{\mathbf{c}_s\}\)，即：

\[
\mathbf{u}_t = \mathrm{Attention}\left(\mathbf{h_t},C\right) \Longrightarrow \mathbf{u}_t = \mathrm{Attention}\left(\mathbf{h}_t, \{\mathbf{c}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\right).
\]

---

### 小结

| 模式 | 注意力范围 \(\mathcal{A}(i)\) | 复杂度 | 代表工作 |
|------|---------------------------|--------|---------|
| 滑动窗口 (Sliding Window) | \(\{j : \|i-j\| \leq w\}\) | \(O(Nw)\) | Mistral 7B, Longformer, Gemma 2 |
| 分块 (Block) | \(\{j : \lfloor j/b \rfloor = \lfloor i/b \rfloor\}\) | \(O(Nb)\) | Sparse Transformer |
| 跨步 (Strided) | \(\{j : (i-j) \bmod l = 0\}\) | \(O(N^2/l)\) | Sparse Transformer |

目前滑动窗口注意力比较常见，通常和全局注意力结合使用（例如 GEMMA 系列采用 5 滑动窗口（local）+ 1 Group Query Attention（global））。

---

## 多头潜在注意力（Multi-Head Latent Attention，MLA） {#mla}

MLA 是 [DeepSeek-V2](https://arxiv.org/abs/2405.04434) 提出的注意力机制，核心动机与 GQA 相同——降低推理时的 KV 缓存开销。但 GQA 只是减少 KV 头的数量，MLA 则直奔本质：**把 KV 压缩到低秩潜在空间，只存压缩后的向量，用的时候再解压。**

### 从 MHA 到 MLA

标准 MHA 中，输入 \( \mathbf{h}_t \in \mathbb{R}^d \) 经三组投影得到 Q、K、V：

\[
\mathbf{q}_t = W^Q \mathbf{h}_t,\quad \mathbf{k}_t = W^K \mathbf{h}_t,\quad \mathbf{v}_t = W^V \mathbf{h}_t
\]

推理时，每个 token 的 \( \mathbf{k}_t, \mathbf{v}_t \) 都要存进 KV 缓存。\( H \) 个头、每头维度 \( d_h \)，单个 token 的 KV 占 \( 2H d_h \) 个元素。

MLA 的做法：**不直接投影到 K 和 V，先投影到一个低维压缩向量 \( \mathbf{c}^{KV}_t \)，再从这里展开。**

\[
\mathbf{c}^{KV}_t = W^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}
\]

\[
\mathbf{k}_t^C = W^{UK} \mathbf{c}^{KV}_t \in \mathbb{R}^{H d_h},\quad
\mathbf{v}_t^C = W^{UV} \mathbf{c}^{KV}_t \in \mathbb{R}^{H d_h}
\]

其中 \( d_c \ll H d_h \)。在 DeepSeek-V2/V3 中，\( d_c = 512 \)，而 \( H d_h = 128 \times 128 = 16384 \)，仅 KV 缓存即从 16384 降至 512。

### RoPE 的解耦处理

低秩压缩与 RoPE 天然冲突：压缩向量 \( \mathbf{c}^{KV}_t \) 内信息混合了所有头，无法按头独立旋转。MLA 的方案是**将位置编码从压缩分支中解耦出来**：额外引入一个不经过压缩的轻量 RoPE 通道。

K 侧的 RoPE 分量 \( \mathbf{k}^R_t \) 由输入直接线性投影得到，**所有头共享同一份**（仅需缓存一份，节省缓存开销）：

\[
\mathbf{k}^R_t = \text{RoPE}(W^{KR} \mathbf{h}_t) \in \mathbb{R}^{d_r}
\]

Q 侧的 RoPE 分量 **每个头独立**（无需缓存，保证不同头之间的查询特征多样性）：

\[
\mathbf{q}^R_{t,i} = \text{RoPE}(W^{QR}_i \mathbf{h}_t) \in \mathbb{R}^{d_r}
\]

\(i\) 表示属于第 \(i\) 个头。最终每头的 K 和 Q 由压缩分支和 RoPE 分支拼接：

\[
\mathbf{k}_{t,i} = [\mathbf{k}^C_{t,i}; \mathbf{k}^R_t],\quad
\mathbf{q}_{t,i} = [\mathbf{q}^C_{t,i}; \mathbf{q}^R_{t,i}]
\]

其中 \( \mathbf{k}^C_{t,i}, \mathbf{q}^C_{t,i} \) 来自解压后的低秩分支，不施加 RoPE；\( \mathbf{k}^R_t, \mathbf{q}^R_{t,i} \) 分别施加 RoPE 旋转后参与点积。**\( \mathbf{k}^R_t \) 跨头共享是 MLA 控制缓存大小的关键**——否则 K 的 RoPE 部分也要存 \( H \times d_r \) 个元素，缓存优势打折。

缓存内容：\( \mathbf{c}^{KV}_t \)（512 维）+ \( \mathbf{k}^R_t \)（\( d_r = 64 \) 维）= 576 维。

### Q 的压缩：减少训练时中间激活存储 {#q-compression-mla}

MLA 对 Q 同样做低秩压缩：

\[
\mathbf{c}^{Q}_t = W^{DQ} \mathbf{h}_t \in \mathbb{R}^{d'_c},\quad
\mathbf{q}^C_t = W^{UQ} \mathbf{c}^{Q}_t
\]

训练时，前向传播的中间激活需要保留给反向传播。不压缩 Q 时需存储 \( \mathbf{q}_t \in \mathbb{R}^{H d_h} \)；压缩后仅存 \( \mathbf{c}^{Q}_t \in \mathbb{R}^{d'_c} \)（\( d'_c \) 通常为 1536），反向时解压重算 \( \mathbf{q}_t \)。这与 Flash Attention 的思路一致——用少量重计算换显存。Q 压缩不影响推理，因为推理无需存激活。

### 注意力计算

\[
    \mathbf{o}_{t,i} = \sum_{j=1}^t \text{Softmax}_j \left(\frac{\mathbf{q}_{t,i}^\top \cdot \mathbf{k}_{j,i}}{\sqrt{d_h + d_r}}\right) \mathbf{v}^C_{j,i}, \\
    \mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \dots; \mathbf{o}_{t,n_h}]
\]

### 复杂度对比

| 机制 | KV 缓存（每 token） | 压缩策略 |
|------|-------------------|---------|
| MHA | \(2 H d_h\) | 无 |
| GQA | \(2 G d_h\)（\(G < H\)） | 减少 KV 头数 |
| MQA | \(2 d_h\) | KV 全共享 |
| **MLA** | \(d_c + d_r\) | 低秩压缩 |

MLA 将 KV 缓存的压缩从"减少头数"推进到"减少秩"，是当前推理效率最优的注意力设计之一。

---

## 压缩稀疏注意力（Compressed Sparse Attention, CSA）{#csa}

**压缩稀疏注意力** = 压缩键值条目（KV entries）+ DeepSeek 稀疏注意力 + 键值共享多查询注意力。来自 [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)。

{{< figure src="csa.png" alt="CSA 结构示意图" caption="CSA 结构示意图" >}}

### 压缩键值条目 {#compressed-kv-entries-csa}

对于一个长度为 \(n\) 的输入隐层序列 \(H \in \mathbb{R^{n \times d}}\)，CSA 会根据它计算出两段 KV entries：

\[
    C^a = H \cdot W^{aKV},\quad C^b = H \cdot W^{bKV}
\]

同时分别计算它们的压缩权重：

\[
    Z^a = H \cdot W^{aZ},\quad Z^b = H \cdot W^{bZ}
\]

其中，\(W^{aKV}, W^{bKV}, W^{aZ}, W^{bZ}\) 是可学习的线性变换矩阵。

CSA 会将 m 个 KV entries 压缩成 1 个 entries。假设将 KV entries 以 m 个为单位划分为一组，对于第 \(i\) 组，CSA 选择从 \(C^a\) 中选取第 \(i\) 组的 KV entry, 从 \(C^b\) 中选取第 \(i - 1\) 组的 KV entry, 并根据它们的压缩权重 \(Z^a\) 和 \(Z^b\) 计算出压缩后的 KV entry。权重的计算方式为：

\[
    [S^{a}_{m \cdot i:m \cdot (i+1) -1}; S^{b}_{m \cdot (i-1):m \cdot i -1}] = \text{Softmax}_{\text{row}}\left([Z^a_{m \cdot i:m \cdot (i+1) -1} + B^a; Z^b_{m \cdot (i-1):m \cdot i -1} + B^b]\right)
\]

其中 \(B^a\) 和 \(B^b\) 是可学习的偏置项。最终第 \(i\) 组的压缩 KV entry 计算方式为：

\[
    C^{\text{Comp}}_i = \sum_{j=m \cdot i}^{m \cdot (i+1) -1} S^a_j \odot C^a_j + \sum_{j=m \cdot (i-1)}^{m \cdot i -1} S^b_j \odot C^b_{j}
\]

因此，一段原本长度为 \(n\) 的 KV entries 会被压缩成长度为 \(n/m\) 的 KV entries。

### DSA 在 CSA 中的应用

\[
I_{t,s}=\sum_{h=1}^{{n_h}^I} w^I_{t,h} \cdot \mathrm{ReLU}\left(\mathbf{q}^I_{t,h} \cdot \mathbf{k}^{I\text{Comp}}_{s}\right).
\]

\[
\mathbf{u}_t = \mathrm{Attention}\left(\mathbf{h_t},C\right) \Longrightarrow \mathbf{u}_t = \mathrm{Attention}\left(\mathbf{h}_t, \{\mathbf{c}_s^{\text{Comp}} \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\right).
\]

### CSA 中 DSA 的注意力机制为 Shared Key-Value MQA

已知，CSA 中采用了 DSA 来选取压缩后的 top-k KV entries 进行全注意力计算。

\[
    \mathrm{Attention}\left(\mathbf{h}_t, \{\mathbf{c}_s^{\text{Comp}} \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\right).
\]

其中 \(\mathrm{Attention}\) 的计算方式为 Shared Key-Value MQA：

\[
    \mathrm{CoreAttn}\left(\text{query}=\mathbf{q}_{t,i}, \text{key}=C^{\text{SprsComp}}_{t},\text{value}=C^{\text{SprsComp}}_{t} \right).
\]

其中，\(C^{\text{SprsComp}}_{t} = \{\mathbf{c}_s^{\text{Comp}} \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\)，\(\mathbf{q}_{t,i}\) 表示第 \(i\) 个注意力头的第 \(t\) 个 token 的查询向量。

> 这里 \(\mathbf{q}_{t,i}\) 计算采用了 MLA 中的思想，如[段落](#q-compression-mla)中所述，对查询向量也进行了低秩压缩，后进行特征“上投影”（up-projection）。

---

## 重度压缩注意力（Heavily Compressed Attention，HCA）{#hca}

**重度压缩注意力** = 重度压缩键值条目 + 键值共享多查询注意力。来自 [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)。

{{< figure src="hca.png" alt="HCA 结构示意图" caption="HCA 结构示意图" >}}

### 重度压缩键值条目

请对照[段落](#compressed-kv-entries-csa)中的压缩键值条目，理解重度压缩键值条目的区别：

\[
    C = H \cdot W^{KV},\quad Z = H \cdot W^{Z}
\]

\[
    S_{m' \cdot i:m' \cdot (i+1) -1} = \text{Softmax}_{\text{row}}\left(Z_{m' \cdot i:m' \cdot (i+1) -1} + B\right),\\
    C^{\text{Comp}}_i = \sum_{j=m' \cdot i}^{m' \cdot (i+1) -1} S_j \odot C_j
\]

其中 \(m' \ll m\)。

### HCA 中的注意力机制 Shared Key-Value MQA

\[
    \mathbf{o}_{t,i} = \mathrm{CoreAttn}\left(\text{query}=\mathbf{q}_{t,i}, \text{key}=C^{\text{Comp}}_{t},\text{value}=C^{\text{Comp}}_{t} \right).
\]

---

## 线性注意力（Linear Attention）

[线性注意力和 Mamba（Linear Attention and Mamba)](../mamba/index.md)

---

## Flash Attention {#flash-attention}

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

