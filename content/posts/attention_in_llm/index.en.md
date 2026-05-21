---
title: "Attention Mechanisms and Their Variants in Large Language Models"
date: 2026-04-23T11:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "Large language model"
    subseries: "Architecture and Training"
categories: ["大语言模型"]
tags: ["架构", "训练"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "How many types of attention are there?"
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

## Grouped Query Attention (GQA) and Multi-Query Attention (MQA)

See [blog](../transformer_in_LLM/index.md#mqa-gqa).

## Sparse Attention {#sparse-attention}

{{< figure src="sparse_attentions.png" alt="Sparse attention mask example" caption="Sparse attention mask example" >}}

### Motivation for Sparse Attention

The computation of standard self-attention can be expressed as: given an input sequence \(X \in \mathbb{R}^{N \times d}\), where \(N\) is the sequence length and \(d\) is the hidden dimension, Query, Key, and Value are obtained via linear projections:

\[Q = XW_Q, \quad K = XW_K, \quad V = XW_V\]

Then the self-attention output is:

\[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V\]

Where each element \((i, j)\) of \(QK^T \in \mathbb{R}^{N \times N}\) represents the attention score of the \(i\)-th token to the \(j\)-th token, and \(M\) is an optional attention mask matrix. The complexity of computing \(QK^T\) is \(O(N^2 d)\); when \(N\) is large (e.g., long documents, high-resolution images), both computational and memory costs become prohibitive.

**The basic idea of sparse attention** is to restrict each token to interact only with a subset of tokens, reducing the complexity from \(O(N^2)\) to \(O(Nk)\), where \(k \ll N\). Let the attendable set for position \(i\) be \(\mathcal{A}(i)\); then the sparse attention mask is:

\[M_{ij} = \begin{cases} 0 & \text{if } j \in \mathcal{A}(i) \\ -\infty & \text{otherwise} \end{cases}\]

Different sparse patterns essentially correspond to different constructions of \(\mathcal{A}(i)\).

---

### Sliding Window Attention {#sliding-window}

**Definition**: Each position interacts only with neighbors within a fixed window before and after it. Given a window size \(w\), the attention range for position \(i\) is:

\[\mathcal{A}_{\text{sw}}(i) = \{j \mid |i - j| \leq w\}\]

The corresponding mask matrix has a **banded** structure:

\[M^{\text{sw}}_{ij} = \begin{cases} 0 & |i-j| \leq w \\ -\infty & \text{otherwise} \end{cases}\]

Complexity: \(O(Nw)\), which approaches linear when \(w \ll N\).

**Characteristics**: Since natural language exhibits strong locality (syntactic collocations, adjacent modifier relationships), sliding window attention captures the main local semantics with a lower computational cost. Mistral 7B uses sliding window attention (\(w=4096\)), significantly reducing KV cache overhead during long sequence inference. However, its limitation is that long-range dependencies beyond the window (e.g., long-distance anaphora, cross-paragraph semantic connections) are directly truncated and cannot be modeled.

---

### Block Attention (i.e., Sparse Transformer's "Fixed") {#block-attention}

Unlike sliding window, block attention divides the sequence into fixed-size **non-overlapping blocks**. Each position interacts only with all tokens within the same block; different blocks are completely isolated. Let the block size be \(b\); the block index for position \(i\) is \(\lfloor i/b \rfloor\); then the attention range is:

\[\mathcal{A}_{\text{block}}(i) = \{j \mid \lfloor j/b \rfloor = \lfloor i/b \rfloor\}\]

The mask matrix has a **block diagonal** structure:

\[M^{\text{block}}_{ij} = \begin{cases} 0 & \lfloor i/b \rfloor = \lfloor j/b \rfloor \\ -\infty & \text{otherwise} \end{cases}\]

Complexity: \(O(Nb)\). Since blocks do not overlap, efficient implementation using block matrix multiplication is possible.

**Comparison with sliding window**: Sliding window means "the window moves with the token," while block means "the window is fixed." A drawback of blocks is that tokens at block boundaries are adjacent but cannot attend to each other.

---

### Strided Attention {#strided-attention}

**Definition**: While block attention is confined to the same block, strided attention covers the entire sequence at uniform intervals, so that different tokens within the same block connect to tokens from different blocks. Given a stride \(l\), the attention range for position \(i\) is:

\[\mathcal{A}_{\text{strided}}(i) = \{j \mid (i-j) \bmod l = 0\}\]

That is, the \(i\)-th position attends to all positions where the difference in indices is divisible by the stride \(l\). The corresponding mask matrix has a **cross-stripe** structure:

\[M^{\text{strided}}_{ij} = \begin{cases} 0 & (i-j) \bmod l = 0 \\ -\infty & \text{otherwise} \end{cases}\]

Complexity: \(O(N^2/l)\), approximately \(O(N\sqrt{N})\) when \(l\) is of the same order as \(\sqrt{N}\).

**Characteristics**: Strided attention enables cross-block information interaction through sparse sampling. Its cost is that local details may be lost due to the stride interval—two adjacent tokens whose index difference does not satisfy the modulo condition cannot attend to each other at all in a strided layer.

---

### DeepSeek Sparse Attention (DSA) {#dsa}

Assume the model is processing a 128K token long context, and the last sentence of the current query is:

```text
“请根据前文第 3 份合同里的违约条款回答……”
```

Standard attention:

```text
当前 token 直接和前面 128K 个 token 全部算 attention
```

DSA:

```text
1. 轻量化索引器（lighting Indexer）快速阅读历史 token
2. 给每一个历史 token 计算一个相关性分数
3. 选取 top-k 相关 token 作为当前 token 的注意力范围
4. 主注意力模块对这 top-k token 进行全注意力计算
```

The main idea of DSA, like all sparse attention mentioned above, is to reduce the number of tokens each query token interacts with. For a query sequence of length $N$, the number of interacting tokens is reduced from $N$ to $k$, and the complexity is reduced from $O(N^2)$ to $O(Nk)$.

#### Lightweight Indexer

The core of the lightweight indexer is to compute the relevance score between the current query token $\mathbf{h}_t \in \mathbb{R}^d$ and historical tokens $\mathbf{h}_s \in \mathbb{R}^d$. In [DeepSeek V3.2](https://arxiv.org/pdf/2512.02556), the indexer is defined as:

$$
I_{t,s}=\sum_{j=1}^{H^I} w^I_{t,j} \cdot \mathrm{ReLU}\left(\mathbf{q}^I_{t,j} \cdot \mathbf{k}^I_{s}\right).
$$

Where $H^I$ is the number of heads in the indexer. $w^I_{t,j}$ is the weight for the $j$-th head, derived from the current query token $\mathbf{h}\_t$. $\mathbf{q}^I_{t,j}$ and $\mathbf{k}^I_{s}$ are the query vector and key vector of the query token and historical token in the indexer, respectively.

Based on the value of \(I_{t,s}\), the model selects the top-k relevant historical tokens for subsequent full attention computation. The interaction target of the current query token \(\mathbf{h}_t\) changes from all tokens in the KV sequence $C$ to the selected top-k token set \(\{\mathbf{c}_s\}\), i.e.:

\[
\mathbf{u}_t = \mathrm{Attention}\left(\mathbf{h_t},C\right) \Longrightarrow \mathbf{u}_t = \mathrm{Attention}\left(\mathbf{h}_t, \{\mathbf{c}_s \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\right).
\]

---

### Summary

| Pattern | Attention Range \(\mathcal{A}(i)\) | Complexity | Representative Work |
|---------|---------------------------|--------|-------------------|
| Sliding Window | \(\{j : \|i-j\| \leq w\}\) | \(O(Nw)\) | Mistral 7B, Longformer, Gemma 2 |
| Block | \(\{j : \lfloor j/b \rfloor = \lfloor i/b \rfloor\}\) | \(O(Nb)\) | Sparse Transformer |
| Strided | \(\{j : (i-j) \bmod l = 0\}\) | \(O(N^2/l)\) | Sparse Transformer |

Currently, sliding window attention is common, often used in combination with global attention (e.g., the GEMMA series uses 5 sliding window (local) + 1 Group Query Attention (global)).

---

## Multi-Head Latent Attention (MLA) {#mla}

MLA is an attention mechanism proposed by [DeepSeek-V2](https://arxiv.org/abs/2405.04434). Its core motivation is the same as GQA—to reduce the KV cache overhead during inference. However, GQA only reduces the number of KV heads, while MLA goes straight to the essence: **compress KV into a low-rank latent space, store only the compressed vectors, and decompress when needed.**

### From MHA to MLA

In standard MHA, the input \( \mathbf{h}_t \in \mathbb{R}^d \) is projected through three sets to obtain Q, K, V:

\[
\mathbf{q}_t = W^Q \mathbf{h}_t,\quad \mathbf{k}_t = W^K \mathbf{h}_t,\quad \mathbf{v}_t = W^V \mathbf{h}_t
\]

During inference, the \( \mathbf{k}_t, \mathbf{v}_t \) for each token must be stored in the KV cache. With \( H \) heads and dimension \( d_h \) per head, the KV of a single token occupies \( 2H d_h \) elements.

MLA's approach: **Instead of projecting directly to K and V, first project to a low-dimensional compressed vector \( \mathbf{c}^{KV}_t \), then expand from there.**

\[
\mathbf{c}^{KV}_t = W^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}
\]

\[
\mathbf{k}_t^C = W^{UK} \mathbf{c}^{KV}_t \in \mathbb{R}^{H d_h},\quad
\mathbf{v}_t^C = W^{UV} \mathbf{c}^{KV}_t \in \mathbb{R}^{H d_h}
\]

Where \( d_c \ll H d_h \). In DeepSeek-V2/V3, \( d_c = 512 \), while \( H d_h = 128 \times 128 = 16384 \), so the KV cache alone is reduced from 16384 to 512.

### Decoupled RoPE Processing

Low-rank compression inherently conflicts with RoPE: the information in the compressed vector \( \mathbf{c}^{KV}_t \) is mixed across heads, making it impossible to rotate independently per head. MLA's solution is to **decouple the positional encoding from the compression branch**: additionally introduce a lightweight RoPE channel that is not compressed.

The RoPE component on the K side, \( \mathbf{k}^R_t \), is obtained by a direct linear projection from the input, **shared across all heads** (only one copy needs to be cached, saving cache overhead):

\[
\mathbf{k}^R_t = \text{RoPE}(W^{KR} \mathbf{h}_t) \in \mathbb{R}^{d_r}
\]

The RoPE component on the Q side is **independent per head** (no caching needed, ensuring diversity of query features across heads):

\[
\mathbf{q}^R_{t,i} = \text{RoPE}(W^{QR}_i \mathbf{h}_t) \in \mathbb{R}^{d_r}
\]

\(i\) indicates belonging to the \(i\)-th head. The final K and Q for each head are concatenated from the compression branch and the RoPE branch:

\[
\mathbf{k}_{t,i} = [\mathbf{k}^C_{t,i}; \mathbf{k}^R_t],\quad
\mathbf{q}_{t,i} = [\mathbf{q}^C_{t,i}; \mathbf{q}^R_{t,i}]
\]

Where \( \mathbf{k}^C_{t,i}, \mathbf{q}^C_{t,i} \) comes from the decompressed low-rank branch, without RoPE applied; \( \mathbf{k}^R_t, \mathbf{q}^R_{t,i} \) is rotated by RoPE before participating in the dot product. **The cross-head sharing of \( \mathbf{k}^R_t \) is key for MLA to control cache size**—otherwise, the RoPE part of K would also need to store \( H \times d_r \) elements, reducing the cache advantage.

Cached content: \( \mathbf{c}^{KV}_t \) (512 dimensions) + \( \mathbf{k}^R_t \) (\( d_r = 64 \) dimensions) = 576 dimensions.

### Q Compression: Reducing Intermediate Activation Storage During Training {#q-compression-mla}

MLA also applies low-rank compression to Q:

\[
\mathbf{c}^{Q}_t = W^{DQ} \mathbf{h}_t \in \mathbb{R}^{d'_c},\quad
\mathbf{q}^C_t = W^{UQ} \mathbf{c}^{Q}_t
\]

During training, the intermediate activations from the forward pass need to be retained for backpropagation. Without compressing Q, one would need to store \( \mathbf{q}_t \in \mathbb{R}^{H d_h} \); after compression, only \( \mathbf{c}^{Q}_t \in \mathbb{R}^{d'_c} \) is stored (\( d'_c \) is typically 1536), and \( \mathbf{q}_t \) is recomputed during backpropagation. This is consistent with the idea of Flash Attention—trading a small amount of recomputation for memory savings. Q compression does not affect inference, as activations do not need to be stored then.

### Attention Computation

\[
    \mathbf{o}_{t,i} = \sum_{j=1}^t \text{Softmax}_j \left(\frac{\mathbf{q}_{t,i}^\top \cdot \mathbf{k}_{j,i}}{\sqrt{d_h + d_r}}\right) \mathbf{v}^C_{j,i}, \\
    \mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \dots; \mathbf{o}_{t,n_h}]
\]

### Complexity Comparison

| Mechanism | KV Cache (per token) | Compression Strategy |
|-----------|---------------------|--------------------|
| MHA | \(2 H d_h\) | None |
| GQA | \(2 G d_h\) (\(G < H\)) | Reduce number of KV heads |
| MQA | \(2 d_h\) | All KV shared |
| **MLA** | \(d_c + d_r\) | Low-rank compression |

MLA advances KV cache compression from "reducing the number of heads" to "reducing the rank," making it one of the most inference-efficient attention designs currently.

---

## Compressed Sparse Attention (CSA) {#csa}

**Compressed Sparse Attention** = Compressed KV entries + DeepSeek Sparse Attention + Shared Key-Value Multi-Query Attention. From [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf).

{{< figure src="csa.png" alt="CSA structure diagram" caption="CSA structure diagram" >}}

### Compressed KV Entries {#compressed-kv-entries-csa}

For an input hidden sequence \(H \in \mathbb{R^{n \times d}}\) of length \(n\), CSA computes two segments of KV entries based on it:

\[
    C^a = H \cdot W^{aKV},\quad C^b = H \cdot W^{bKV}
\]

It simultaneously computes their compression weights:

\[
    Z^a = H \cdot W^{aZ},\quad Z^b = H \cdot W^{bZ}
\]

Where \(W^{aKV}, W^{bKV}, W^{aZ}, W^{bZ}\) are learnable linear transformation matrices.

CSA compresses m KV entries into 1 entry. Assuming the KV entries are grouped into groups of m, for the \(i\)-th group, CSA selects the \(i\)-th group's KV entry from \(C^a\) and selects the \(i - 1\)-th group's KV entry from \(C^b\), and computes the compressed KV entry based on their compression weights \(Z^a\) and \(Z^b\). The weights are computed as:

\[
    [S^{a}_{m \cdot i:m \cdot (i+1) -1}; S^{b}_{m \cdot (i-1):m \cdot i -1}] = \text{Softmax}_{\text{row}}\left([Z^a_{m \cdot i:m \cdot (i+1) -1} + B^a; Z^b_{m \cdot (i-1):m \cdot i -1} + B^b]\right)
\]

Where \(B^a\) and \(B^b\) are learnable bias terms. Finally, the compressed KV entry for the \(i\)-th group is computed as:

\[
    C^{\text{Comp}}_i = \sum_{j=m \cdot i}^{m \cdot (i+1) -1} S^a_j \odot C^a_j + \sum_{j=m \cdot (i-1)}^{m \cdot i -1} S^b_j \odot C^b_{j}
\]

Therefore, a segment of KV entries originally of length \(n\) is compressed into KV entries of length \(n/m\).

### Application of DSA in CSA

\[
I_{t,s}=\sum_{h=1}^{{n_h}^I} w^I_{t,h} \cdot \mathrm{ReLU}\left(\mathbf{q}^I_{t,h} \cdot \mathbf{k}^{I\text{Comp}}_{s}\right).
\]

\[
\mathbf{u}_t = \mathrm{Attention}\left(\mathbf{h_t},C\right) \Longrightarrow \mathbf{u}_t = \mathrm{Attention}\left(\mathbf{h}_t, \{\mathbf{c}_s^{\text{Comp}} \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\right).
\]

### Attention Mechanism of DSA in CSA: Shared Key-Value MQA

It is known that CSA uses DSA to select the top-k compressed KV entries for full attention computation.

\[
    \mathrm{Attention}\left(\mathbf{h}_t, \{\mathbf{c}_s^{\text{Comp}} \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\right).
\]

Where \(\mathrm{Attention}\) is computed via Shared Key-Value MQA:

\[
    \mathrm{CoreAttn}\left(\text{query}=\mathbf{q}_{t,i}, \text{key}=C^{\text{SprsComp}}_{t},\text{value}=C^{\text{SprsComp}}_{t} \right).
\]

Where \(C^{\text{SprsComp}}_{t} = \{\mathbf{c}_s^{\text{Comp}} \mid I_{t,s} \in \text{Top-k}(I_{t,:})\}\), \(\mathbf{q}_{t,i}\) represents the query vector of the \(t\)-th token in the \(i\)-th attention head.

> The computation of \(\mathbf{q}_{t,i}\) here adopts the idea from MLA, as described in the [section](#q-compression-mla), also applying low-rank compression to the query vector, followed by feature "up-projection".

---

## Heavily Compressed Attention (HCA) {#hca}

**Heavily Compressed Attention** = Heavily compressed KV entries + Shared Key-Value Multi-Query Attention. From [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf).

{{< figure src="hca.png" alt="HCA structure diagram" caption="HCA structure diagram" >}}

### Heavily Compressed KV Entries

Please refer to the [section](#compressed-kv-entries-csa) on compressed KV entries to understand the difference for heavily compressed KV entries:

\[
    C = H \cdot W^{KV},\quad Z = H \cdot W^{Z}
\]

\[
    S_{m' \cdot i:m' \cdot (i+1) -1} = \text{Softmax}_{\text{row}}\left(Z_{m' \cdot i:m' \cdot (i+1) -1} + B\right),\\
    C^{\text{Comp}}_i = \sum_{j=m' \cdot i}^{m' \cdot (i+1) -1} S_j \odot C_j
\]

Where \(m' \ll m\).

### Attention Mechanism in HCA: Shared Key-Value MQA

\[
    \mathbf{o}_{t,i} = \mathrm{CoreAttn}\left(\text{query}=\mathbf{q}_{t,i}, \text{key}=C^{\text{Comp}}_{t},\text{value}=C^{\text{Comp}}_{t} \right).
\]

---

## Linear Attention

[Linear Attention and Mamba](../mamba/index.md)

---

## Flash Attention {#flash-attention}

### Problem: Attention Bottleneck is Not Computation, but Memory Bandwidth

The computational cost of standard self-attention is \(O(N^2 d)\), but modern GPU FLOPS far exceed memory bandwidth (HBM bandwidth growth lags far behind compute growth). In practice, the GPU repeatedly writes and reads the \(N \times N\) attention matrix to/from HBM:

\[S = QK^T \in \mathbb{R}^{N \times N}\]

\[P = \text{softmax}(S) \in \mathbb{R}^{N \times N}\]

\[O = PV \in \mathbb{R}^{N \times d}\]

Where \(S\) and \(P\) are both \(O(N^2)\) intermediate results; each softmax and matrix multiplication requires reads/writes on HBM. **The bottleneck of attention operations is not FLOPs, but IO.**

### Key Observation: GPU Memory Hierarchy

GPUs have two levels of memory:

| Level | Capacity | Bandwidth | Latency |
|-------|----------|-----------|---------|
| HBM (High Bandwidth Memory) | Large (80 GB, H100) | Low (~3 TB/s) | High |
| SRAM (On-chip Cache) | Small (~20 MB, H100) | High (~19 TB/s) | Low |

Standard attention repeatedly reads and writes \(O(N^2)\) data on HBM—HBM bandwidth becomes the bottleneck, while SRAM is wasted.

**Core idea of Flash Attention**: Divide \(Q, K, V\) into blocks, load only one block at a time onto SRAM to perform local computation, and write only the final output \(O\) back to HBM, while intermediate results \(S, P\) are never materialized.

### Online Softmax: Mathematical Basis for Tiling

Standard softmax requires the entire row to compute (first find the global maximum, then normalize). The key mathematical tool for Flash Attention is **online softmax**: by maintaining two running statistics, it incrementally updates when processing blocks.

For the \(i\)-th row, split \(K^T\) along the column dimension into \(T_c\) blocks, \(K_1, \dots, K_{T_c}\). When processing the \(j\)-th block:

\[m^{(j)} = \max(m^{(j-1)}, \max(Q_i K_j))\]

\[\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot \ell^{(j-1)} + \sum e^{Q_i K_j - m^{(j)}}\]

\[O_i^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot O_i^{(j-1)} + e^{Q_i K_j - m^{(j)}} V_j\]

Where \(m\) is the current running maximum, and \(\ell\) is the current running softmax denominator. After processing all blocks, \(O_i = O_i^{(T_c)} / \ell^{(T_c)}\).

**Why it is equivalent to standard softmax**: Starting from numerically stable softmax:

\[
\text{softmax}(s_1, \dots, s_N) = \frac{e^{s_j - m}}{\sum_k e^{s_k - m}},\quad m = \max(s_1, \dots, s_N)
\]

Where \(s_k = Q_i \cdot K_k / \sqrt{d_k}\) is the scalar attention score between query position \(i\) and key position \(k\). The entire row \(\{s_1, \dots, s_N\}\) is compared with each other, with each query position performing the operation independently.

This is equivalent to first computing \(a_j = e^{s_j - m}\) for each \(s_j\), then letting \(p_j = a_j / \sum a_k\), and finally outputting \(O = \sum p_j V_j\).

Now assume only the first \(B\) tokens have been processed, giving:
\[
m_{\text{old}} = \max_{k \leq B} s_k,\quad L_{\text{old}} = \sum_{k \leq B} e^{s_k - m_{\text{old}}},\quad O_{\text{old}} = \sum_{k \leq B} e^{s_k - m_{\text{old}}} V_k
\]

When a new batch of tokens (the \(B+1\)-th to \(B'\)-th tokens) arrives, let the temporary statistics within this batch be:
\[
m_{\text{new}} = \max_{B < k \leq B'} s_k,\quad L_{\text{new}}' = \sum_{B < k \leq B'} e^{s_k - m_{\text{new}}},\quad O_{\text{new}}' = \sum_{B < k \leq B'} e^{s_k - m_{\text{new}}} V_k
\]

At this point, the global maximum becomes \(m' = \max(m_{\text{old}}, m_{\text{new}})\). Two cases exist:

**Case 1**: \(m_{\text{new}} \leq m_{\text{old}}\), the maximum is unchanged (\(m' = m_{\text{old}}\)). Old results need no correction; the new batch computes exponentials based on \(m_{\text{old}}\):
\[
L' = L_{\text{old}} + \sum_{B < k \leq B'} e^{s_k - m_{\text{old}}},\quad
O' = O_{\text{old}} + \sum_{B < k \leq B'} e^{s_k - m_{\text{old}}} V_k
\]

**Case 2**: \(m_{\text{new}} > m_{\text{old}}\), the maximum is refreshed (\(m' = m_{\text{new}}\)). Key step—the old results need to be rescaled based on the new maximum. Using the identity:
\[
e^{s_k - m'} = e^{s_k - m_{\text{old}}} \cdot e^{m_{\text{old}} - m'}
\]
Therefore, each old exponential, old denominator, and old output must be multiplied by a discount factor \(e^{m_{\text{old}} - m'}\) (where \(m_{\text{old}} - m' < 0\), factor less than 1):
\[
L' = e^{m_{\text{old}} - m'} \cdot L_{\text{old}} + \sum_{B < k \leq B'} e^{s_k - m'},\quad
O' = e^{m_{\text{old}} - m'} \cdot O_{\text{old}} + \sum_{B < k \leq B'} e^{s_k - m'} V_k
\]

After processing all blocks, \(O_i = O_i^{(T_c)} / \ell^{(T_c)}\) completes the final normalization.

### Complexity Analysis

| | Standard Attention | Flash Attention |
|---|--------------------|-----------------|
| Computation | \(O(N^2 d)\) | \(O(N^2 d)\) (same) |
| HBM Reads | \(O(N^2)\) (read/write full N×N matrix each time) | \(O(N^2 d^2 / M)\) (N×N matrix never materialized) |
| Memory Usage | \(O(N^2)\) | \(O(Nd)\) |
| Numerical Result | — | Exactly the same as standard attention |

Flash Attention does not change the arithmetic result of attention; it only changes the IO pattern. It is not an approximation—the output is numerically equivalent to standard softmax attention.

### Backpropagation: Recomputation for Memory Savings

The backward pass of standard attention requires the softmax matrix \(P \in \mathbb{R}^{N \times N}\) saved from the forward pass to compute gradients. The gradient formula is:

\[
\frac{\partial L}{\partial Q} \propto dP \cdot K,\quad
\frac{\partial L}{\partial K} \propto dP^T \cdot Q,\quad
\frac{\partial L}{\partial V} \propto P^T \cdot dO
\]

Where \(dO\) and \(dP\) are the upstream gradients passed to \(O\) and \(P\). Storing \(P\) means the backward pass again requires \(O(N^2)\) memory—the memory saved in the forward pass is all returned in the backward.

**Flash Attention's solution**: Do not store \(P\) in the forward pass; only keep the output \(O \in \mathbb{R}^{N \times d}\) and the statistics \(m_i, \ell_i \in \mathbb{R}^{N}\) for each row (a total of \(O(Nd)\)). During the backward pass, use these metadata to accurately reconstruct \(P\) in blocks:

\[
P_{ij} = e^{Q_i K_j - m_i} / \ell_i
\]

Since \(m_i, \ell_i\) is exactly all the information needed for softmax, together with the original \(Q_i, K_j\), a block of \(P\) can be precisely recovered at any time. Then the gradients are computed block by block following the standard chain rule, ensuring intermediate results never materialize the full \(N \times N\) matrix.

**Cost**: Each training step requires an additional forward pass (recomputing \(P\)), trading about 30% extra FLOPs for a memory saving of \(O(N^2) \to O(Nd)\). This trade-off is decisive for long sequence training—without it, contexts of hundreds of thousands of tokens simply cannot be trained on a single GPU.

### Subsequent Iterations

- **Flash Attention 2 (Dao, 2023)**: Optimized tile scheduling order to reduce synchronization overhead between thread blocks; improved for the asymmetric SM structure of A100/H100. Reaches 75% of theoretical peak throughput on H100.
- **Flash Attention 3 (Shah et al., 2024)**: Leverages Hopper architecture's TMA (Tensor Memory Accelerator) and WGMMA instructions for asynchronous data transfer, achieves end-to-end acceleration for FP8, and further optimizes for long context inference.

---

## References

1. Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). *Generating Long Sequences with Sparse Transformers.* arXiv:1904.10509.
2. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.
3. Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691.
4. Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* arXiv:2407.08608.
5. Gemma Team, Google DeepMind. (2025). *Gemma 3 Technical Report.* arXiv:2503.19786.
6. Gemma Team, Google DeepMind. (2026). *Gemma 4: Frontier Open Models Under Apache 2.0.* Google AI Blog, April 2026.
7. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.

---