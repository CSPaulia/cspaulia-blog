---
title: "Residual Connections and Their Variants"
date: 2026-04-29T11:30:03+08:00
series:
    main: "Large Language Model"
    subseries: "Architecture and Training"
categories: ["Large Language Models"]
tags: ["Architecture", "Training"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Residual Connections and Their Variants"
disableHLJS: false
disableShare: false
hideSummary: true
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
tocEndLevel: 2 # show only h2 headings in the table of contents for this post
cover:
    image: "hc.png"
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

## 1. Residual Connection

Straight to the formula:

\[
    h_{l+1} = h_l + \mathcal{T}(h_l)
\]

Where \(h_l\) is the input at layer \(l\), and \(\mathcal{T}\) is a neural network layer (e.g. convolution, attention, or fully-connected).

## 2. Hyper-Connection (HC)

> The original paper is not the clearest — in fact, it's rather messy...

The core idea: [Hyper-Connection](http://arxiv.org/abs/2409.19606) recasts the residual connection as a "matrix read/write" form, then replaces the fixed matrices with learnable ones.

\[
    \mathbf{H}^{l+1} = \mathbf{A}_r \mathbf{H}^l + \mathbf{B} \mathcal{T}(\mathbf{A}_m^{\top} \mathbf{H}^l)
\]

{{< figure src="../../../posts/residual/hc.png" alt="HC" caption="Hyper-Connection Diagram" >}}

### 2.1. From One Residual Stream to Many

A classic residual network has a single residual stream, maintaining hidden state \(\mathbf{h}^l\).

Hyper-Connection introduces multiple residual streams, each maintaining different hidden information. Specifically, for the initial input \(\mathbf{h}^0\), it is replicated \(n\) times according to the hyperparameter expansion rate \(n\), yielding \(n\) residual streams:

\[
    \begin{aligned}
        \mathbf{h}^0_1 &= \mathbf{h}^0 \\
        \mathbf{h}^0_2 &= \mathbf{h}^0 \\
        &\vdots \\
        \mathbf{h}^0_n &= \mathbf{h}^0
    \end{aligned}
\]

Each stream performs a residual connection independently:

\[
    \mathbf{h}^{l+1}_i = \mathbf{h}^l_i + \mathcal{T}(\mathbf{h}^l_i), \quad i = 1, 2, \ldots, n
\]

We can write the residual streams in matrix form:

\[
    \mathbf{H}^l = \begin{bmatrix}
        \mathbf{h}^l_1 \\
        \mathbf{h}^l_2 \\
        \vdots \\
        \mathbf{h}^l_n
    \end{bmatrix}
\]

### 2.2. Merging Residual Streams

The neural network layer \(\mathcal{T}\) typically accepts a single hidden input \(\mathbf{h}^l\), not a matrix \(\mathbf{H}^l\). Thus, the multiple residual streams in \(\mathbf{H}^l\) must be merged into one:

\[
    \mathbf{h}^l_0 = \alpha_1 \mathbf{h}^l_1 + \alpha_2 \mathbf{h}^l_2 + \ldots + \alpha_n \mathbf{h}^l_n = \mathbf{A}_m^{\top} \mathbf{H}^l
\]

Where \(\alpha_i\) are learnable weights and \(\mathbf{A}_m = [\alpha_1, \alpha_2, \ldots, \alpha_n]^\top\) is a learnable weight vector.

### 2.3. Fusing New and Old Information

The new information is:

\[
    \mathcal{T}(\mathbf{h}^l_0) = \mathcal{T}(\mathbf{A}_m^{\top} \mathbf{H}^l)
\]

The old residual stream information is:

\[
    \mathbf{H}^l
\]

Simple addition is a bit dull — we can introduce a learnable weight matrix \(\mathbf{A}_r \in \mathbb{R}^{n \times n}\) to adjust the weight of old information, and a learnable weight vector \(\mathbf{B} \in \mathbb{R}^{n}\) to adjust the weight of new information:

\[
    \mathbf{H}^{l+1} = \mathbf{A}_r \mathbf{H}^l + \mathbf{B} \mathcal{T}(\mathbf{A}_m^{\top} \mathbf{H}^l)
\]

This is the core formula of **Static Hyper-Connection (SHC)**.

### 2.4. Dynamic Hyper-Connection (DHC)

Here are the formulas — the paper does not explain the design motivation...:

\[
    \overline{\mathbf{H}} = \text{norm}(\mathbf{H}) \\
    \mathcal{B} = s_\beta \cdot \text{tanh}(\mathbf{W}_\beta \overline{\mathbf{H}}) + \mathbf{B} \\
    \mathcal{A}_m = s_\alpha \cdot \text{tanh}(\mathbf{W}_m \overline{\mathbf{H}}) + \mathbf{A}_m \\
    \mathcal{A}_r = s_\alpha \cdot \text{tanh}(\mathbf{W}_r \overline{\mathbf{H}}) + \mathbf{A}_r \\
\]

Where the dynamic parameters \(\mathbf{W}_\beta\), \(\mathbf{W}_m\), and \(\mathbf{W}_r\) are learnable weight matrices. \(s_\alpha\) and \(s_\beta\) are learnable scaling factors.

\[
    \mathbf{H}^{l+1} = \mathcal{A}_r \mathbf{H}^l + \mathcal{B} \mathcal{T}(\mathcal{A}_m^{\top} \mathbf{H}^l)
\]

### 2.5. Advantages

Hyper-Connection significantly boosts model performance without a noticeable increase in parameter count.

## 3. Manifold-Constrained Hyper-Connection (mHC)

### 3.1. Problems with Hyper-Connection

Single-layer hyper-connection:

\[
    \mathbf{H}^{l+1} = \mathbf{A}_r \mathbf{H}^l + \mathbf{B} \mathcal{T}(\mathbf{A}_m^{\top} \mathbf{H}^l)
\]

In the [mHC](https://arxiv.org/abs/2512.24880) paper, \(\mathbf{H}^{l+1}\) is written as \(\mathbf{x}_{i+1}\), \(\mathbf{H}^l\) as \(\mathbf{x}_i\), \(\mathbf{A}_r\) as \(\mathcal{H}^{res}_i\), \(\mathbf{B}\) as \({\mathcal{H}^{post}_i}^\top\), \(\mathbf{A}_m\) as \(\mathcal{H}^{pre}_i\), and \(\mathcal{T}\) as \(\mathcal{F}_i\). Then a single hyper-connection layer can be written as:

\[
    \mathbf{x}_{i+1} = \mathcal{H}^{res}_i \mathbf{x}_i + {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i)
\]

This notation is, in my opinion, much clearer.

Two-layer hyper-connection expands to:

\[
    \mathbf{x}_{i+2} = \mathcal{H}^{res}_{i+1} \mathbf{x}_{i+1} + {\mathcal{H}^{post}_{i+1}}^\top \mathcal{F}_{i+1}(\mathcal{H}^{pre}_{i+1} \mathbf{x}_{i+1}) \\
    = \mathcal{H}^{res}_{i+1} (\mathcal{H}^{res}_i \mathbf{x}_i + {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i)) + {\mathcal{H}^{post}_{i+1}}^\top \mathcal{F}_{i+1}(\mathcal{H}^{pre}_{i+1} \mathbf{x}_{i+1}) \\
    = \mathcal{H}^{res}_{i+1} \mathcal{H}^{res}_i \mathbf{x}_i + \mathcal{H}^{res}_{i+1} {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i) + {\mathcal{H}^{post}_{i+1}}^\top \mathcal{F}_{i+1}(\mathcal{H}^{pre}_{i+1} \mathbf{x}_{i+1})
\]

It is straightforward to derive the multi-layer form:

\[
    \mathbf{x}_{L} = \mathcal{H}^{res}_{L-1} \cdots \mathcal{H}^{res}_{L-l} \mathbf{x}_l + \sum_{i=l}^{L-1} \left( \mathcal{H}^{res}_{L-1} \cdots \mathcal{H}^{res}_{i+1} {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i) \right) \\
    = \left( \prod_{i=l}^{L-1} \mathcal{H}^{res}_i \right) \mathbf{x}_l + \sum_{i=l}^{L-1} \left( \prod_{j=1}^{L-1-i} \mathcal{H}^{res}_{L-j} \right) {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i)
\]

Where the old-information flow is \(\left( \prod_{i=l}^{L-1} \mathcal{H}^{res}_i \right) \mathbf{x}_l\), and the new-information flow is \(\sum_{i=l}^{L-1} \left( \prod_{j=1}^{L-1-i} \mathcal{H}^{res}_{L-j} \right) {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i)\).

Crucially, the coefficient of the old-information flow is \(\prod_{i=l}^{L-1} \mathcal{H}^{res}_i\). In a classic residual connection, the coefficient is 1, so the norm of the old-information flow remains unchanged. In hyper-connection, the coefficient is \(\prod_{i=l}^{L-1} \mathcal{H}^{res}_i\), which can amplify or attenuate the flow, affecting both information propagation and gradient flow.

<details>
<summary>Impact of Norms on Gradients and Information</summary>
Consider the norm upper bound of \(\mathcal{H}^{res}_{L-1} \cdots \mathcal{H}^{res}_{L-l} \mathbf{x}_l\):

\[
    \| \mathcal{H}^{res}_{L-1} \cdots \mathcal{H}^{res}_{L-l} \mathbf{x}_l \| \leq \left( \prod_{i=l}^{L-1} \| \mathcal{H}^{res}_i \| \right) \| \mathbf{x}_l \|
\]

If \(\| \mathcal{H}^{res}_i \| = 0.9\), then after 100 layers the norm upper bound is \(0.9^{100} \| \mathbf{x}_l \| \approx 2.6561 \times 10^{-5} \| \mathbf{x}_l \|\) — information propagation nearly vanishes. If \(\| \mathcal{H}^{res}_i \| = 1.1\), after 100 layers the bound is \(1.1^{100} \| \mathbf{x}_l \| \approx 13780.6 \| \mathbf{x}_l \|\) — information is amplified over ten-thousand-fold.

During backpropagation, gradients are multiplied by the transposed Jacobians:

\[
    \frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \left( \frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l} \right)^\top \left( \frac{\partial \mathbf{x}_{l+2}}{\partial \mathbf{x}_{l+1}} \right)^\top \cdots \left( \frac{\partial \mathbf{x}_{L}}{\partial \mathbf{x}_{L-1}} \right)^\top \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \\
    = J_l^\top J_{l+1}^\top \cdots J_{L-1}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}
\]

If most singular values of these Jacobians are less than 1, gradients shrink as they propagate backward — **vanishing gradients**. If most are greater than 1, they grow — **exploding gradients**.
</details>

The mHC paper proves that HC training is unstable, and that the extra learnable parameters cause significant memory-access overhead, severely impacting training speed.

{{< figure src="../../../posts/residual/hc_io.png" alt="hc" >}}

### 3.2. Design of mHC

To address the training instability of HC, mHC constrains \(\mathcal{H}^{\mathrm{res}}_l\) to lie in a special set:

\[
\mathcal{P}_{\mathcal{M}^{\mathrm{res}}}
\left(\mathcal{H}^{\mathrm{res}}_l\right)
:=
\left\{
\mathcal{H}^{\mathrm{res}}_l \in \mathbb{R}^{n \times n}
\ \middle|\
\mathcal{H}^{\mathrm{res}}_l \mathbf{1}_n = \mathbf{1}_n,\ 
\mathbf{1}_n^{\top}\mathcal{H}^{\mathrm{res}}_l = \mathbf{1}_n^{\top},\ 
\mathcal{H}^{\mathrm{res}}_l \geq 0
\right\}.
\]

Such a matrix is called a **Doubly Stochastic Matrix**. Its advantages are:
1. Norm bound: \(\| \mathcal{H}^{\mathrm{res}}_l \| \leq 1\), preventing gradient explosion;
2. Closure under composition: the product of two doubly stochastic matrices remains doubly stochastic.
3. Birkhoff Polytope property: any doubly stochastic matrix can be written as a convex combination of permutation matrices. For example, \(\mathcal{P}_{\mathcal{M}^{\mathrm{res}}} \left(\mathcal{H}^{\mathrm{res}}_l\right) \) can be expressed as:

\[
\mathcal{P}_{\mathcal{M}^{\mathrm{res}}} \left(\mathcal{H}^{\mathrm{res}}_l\right) = 
\begin{bmatrix}
0.5 & 0.5 \\
0.5 & 0.5
\end{bmatrix} = 0.5 \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix} + 0.5 \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\]

<details>
<summary>Permutation Matrices</summary>
A permutation matrix has exactly one 1 in each row and each column, and zeros elsewhere. For example, a 4×4 permutation matrix:

\[
    P = \begin{bmatrix}
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1 \\
        1 & 0 & 0 & 0
    \end{bmatrix}
\]
</details>

Thus \(\mathcal{H}^{\mathrm{res}}_l\) is not an arbitrary matrix — it represents a "probabilistic soft permutation / soft mixing" of multiple residual information streams. This constraint makes multi-layer residual mappings more stable, more interpretable, and less prone to gradient explosion.

The complete mHC design is:

\[
\left\{
\begin{aligned}
\vec{\mathbf{x}}'_l &= \mathrm{RMSNorm}(\vec{\mathbf{x}}_l) \\[2pt]
\tilde{\mathcal{H}}^{\mathrm{pre}}_l
&= \alpha^{\mathrm{pre}}_l \cdot
\left(\vec{\mathbf{x}}'_l \varphi^{\mathrm{pre}}_l\right) + \mathbf{b}^{\mathrm{pre}}_l \\[2pt]
\tilde{\mathcal{H}}^{\mathrm{post}}_l
&= \alpha^{\mathrm{post}}_l \cdot
\left(\vec{\mathbf{x}}'_l \varphi^{\mathrm{post}}_l\right) + \mathbf{b}^{\mathrm{post}}_l \\[2pt]
\tilde{\mathcal{H}}^{\mathrm{res}}_l
&= \alpha^{\mathrm{res}}_l \cdot
\mathrm{mat}\!\left(\vec{\mathbf{x}}'_l \varphi^{\mathrm{res}}_l\right) + \mathbf{b}^{\mathrm{res}}_l
\end{aligned}
\right.
\]

\[
\left\{
\begin{aligned}
\mathcal{H}^{\mathrm{pre}}_l
&= \sigma\!\left(\tilde{\mathcal{H}}^{\mathrm{pre}}_l\right) \\[2pt]
\mathcal{H}^{\mathrm{post}}_l
&= 2\sigma\!\left(\tilde{\mathcal{H}}^{\mathrm{post}}_l\right) \\[2pt]
\mathcal{H}^{\mathrm{res}}_l
&= \mathrm{Sinkhorn\text{-}Knopp}
\!\left(\tilde{\mathcal{H}}^{\mathrm{res}}_l\right)
\end{aligned}
\right.
\]

\[
\mathbf{M}^{(t)} = \mathcal{T}_r
\left(
\mathcal{T}_c
\left(
\mathbf{M}^{(t-1)}
\right)
\right)
\]

Where \(\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}\) is the flattened form of \(\mathbf{x}_l \in \mathbb{R}^{n \times C}\). \(\varphi^{\cdot}_l\) denotes linear transformation weight matrices, \(\mathbf{b}^{\cdot}_l\) bias terms, and \(\alpha^{\cdot}_l\) learnable scaling factors. \(\sigma\) is the Sigmoid function, \(\mathrm{mat}\) reshapes a vector (\(\mathbb{R}^{1 \times n^2}\)) into a matrix (\(\mathbb{R}^{n \times n}\)), and \(\mathrm{Sinkhorn\text{-}Knopp}\) is an algorithm that iteratively transforms a matrix into a doubly stochastic matrix. The Sinkhorn-Knopp procedure is:

\[
    \mathbf{M}^{(0)} = \exp\!\left(\tilde{\mathcal{H}}^{\mathrm{res}}_l\right), \\
    \mathbf{M}^{(t)} = \mathcal{T}_r\!\left(\mathcal{T}_c\!\left(\mathbf{M}^{(t-1)}\right)\right)
\]

As \(t \to \infty\), \(\mathbf{M}^{(t)}\) converges to a doubly stochastic matrix (as guaranteed by Sinkhorn's theorem). Here \(\mathcal{T}_c\) is column normalization—each element is divided by its column sum, making every column sum to 1; \(\mathcal{T}_r\) is row normalization—each element is divided by its row sum, making every row sum to 1:

\[
\mathcal{T}_c(\mathbf{M})_{ij} = \frac{M_{ij}}{\sum_i M_{ij}}, \quad
\mathcal{T}_r(\mathbf{M})_{ij} = \frac{M_{ij}}{\sum_j M_{ij}}
\]

That is, each round performs column normalization first and row normalization second. The original paper only describes the two qualitatively as "row and column normalization" and gives no component-wise formulas; the practical implementation takes \(t_{\max} = 20\), yielding an approximate solution.

### 3.3. Infrastructure Design

1. **Kernel Fusion**: Too many small operations cause repeated HBM reads/writes and large kernel-launch overhead. Fusing multiple small operations into a single kernel reduces memory access and launch overhead.
2. **Recomputation**: Intermediate quantities are not stored during the forward pass; they are recomputed during backpropagation.
3. **Communication-Computation Overlap**: \(n\) information streams increase communication cost; overlapping communication with computation mitigates this overhead.

### 3.4. Single-Pass mHC: Shifting One Step to Break the Data Dependency

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) keeps mHC but improves its implementation. The tech report writes mHC in a more compact form:

\[
X_{l+1} = B_l X_l + C_l F_l(A_l X_l), \quad (A_l, B_l, C_l) = H(X_l)
\]

where \(X_l \in \mathbb{R}^{n \times d}\) are the \(n\) residual streams between adjacent blocks (the \(\mathbf{x}_l\) in the notation above), \(B_l \in \mathbb{R}^{n \times n}\) is the residual mixing matrix (i.e., \(\mathcal{H}^{\mathrm{res}}_l\)), \(A_l \in \mathbb{R}^{1 \times n}\) and \(C_l \in \mathbb{R}^{n \times 1}\) are the input-mixing and output-contraction coefficients (playing the roles of \(\mathcal{H}^{\mathrm{pre}}_l\) and \(\mathcal{H}^{\mathrm{post}}_l\)), and all three are predicted from \(X_l\) by a coefficient predictor \(H\).

**Problem of the original implementation**: DeepSeek-V4 executes the equation above with three kernels in sequence—residual update, coefficient prediction, and input mixing:

\[
X_l = B_{l-1} X_{l-1} + C_{l-1} Y_{l-1} \quad \text{(residual update, contraction over } n\text{)}
\]

\[
(A_l, B_l, C_l) = H(X_l) \quad \text{(coefficient prediction, contraction over } nd\text{)}
\]

\[
\hat{X}_l = A_l X_l \quad \text{(input mixing, contraction over } n\text{)}
\]

**The Single-Pass mHC improvement**: shift the input-mixing coefficients by one step, so each block uses the coefficients predicted by the previous block, \(A_{l-1}\):

\[
X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l), \quad (A_l, B_l, C_l) = H(X_l)
\]

Input mixing no longer depends on the current block's coefficient prediction, so each tile of \(X_l\) can be used immediately for both input mixing and coefficient prediction without waiting for the full reduction. Empirically, this shift incurs negligible performance degradation.

**The Mega-mHC kernel**: for deployment, residual update, input mixing, and coefficient prediction are fused into a single kernel. It processes \(X_l\) in tiles along the hidden dimension: each tile computes the mixed input and accumulates the statistics needed to predict the coefficients for the next block. The three implementations compare as follows:

| Implementation | Kernels | Notes | Activation memory traffic (reads + writes) |
|----------------|---------|-------|--------------------------------------------|
| DeepSeek-V4 original | 3 | Residual update, coefficient prediction, and input mixing run sequentially | \((4n+4)d\) |
| Mega-mHC (without the shift) | 1 | Three operations fused; still bound by the \(A_l\) dependency, requiring two passes | \((3n+2)d\) |
| Mega-mHC + Single-Pass mHC | 1 | Fusion plus the one-step shift | \((2n+2)d\) (theoretical lower bound, halved) |

The residual is thereby read once and written once.

The shift takes effect in both training and inference—the model already uses the \(A_{l-1}\) semantics during pre-training, merely with the multi-kernel implementation (the shift only changes which coefficients each block applies); the Mega-mHC kernel fusion is a deployment-side optimization, used only at deployment.

## 4. Identity Hyper-Connection (iHC)

A [Zhihu blog post](https://zhuanlan.zhihu.com/p/2010852389670908320) reports the following experimental conclusion: **replacing mHC's \(\mathcal{H}^{\mathrm{res}}\) directly with the identity matrix \(I\) works even better**—the ranking is Identity HC > mHC > mHC lite > mHC orthogonal. The experiments were conducted on Qwen3 1.7B and 8B dense models trained from scratch on 150B tokens. These are small-scale personal experiments without peer review, so take the following as a reference only.

iHC's approach is to directly replace \(\mathcal{H}^{\mathrm{res}}_l\) in the mHC update with the identity matrix \(I\):

\[\mathbf{x}_{l+1} = \mathbf{x}_l + \mathrm{diag}\left(\mathcal{H}^{\mathrm{post}}_l\right) \mathcal{F}_l\left(\mathcal{H}^{\mathrm{pre}}_l \mathbf{x}_l\right)\]

<figure>
  <img src="../../../posts/residual/ihc_training_curves.png" alt="Identity HC training curve screenshot" loading="lazy" width="100%" />
  <figcaption>A partial screenshot of the training curves of Qwen3 1.7B trained from scratch on 150B tokens. Source: <a href="https://zhuanlan.zhihu.com/p/2010852389670908320">Zhihu post “Your DeepSeek mHC May Not Need the ‘m’”</a>.</figcaption>
</figure>

### 4.1. Observation: The Accumulated Product of \(\mathcal{H}^{\mathrm{res}}\) Collapses

The \(\mathcal{H}^{\mathrm{res}}\) learned by a trained mHC shows the following pattern:

- A single layer (depth = 1): close to the identity matrix—diagonal entries around 0.96, off-diagonal entries around 0.01;
- The accumulated product (depth ≥ 10): collapses to the all-0.25 matrix (uniform mixing).

That is, the single-layer \(\mathcal{H}^{\mathrm{res}}\) learned by Sinkhorn-Knopp is close to the identity matrix, but after multiplying many layers it becomes the all-0.25 matrix, completely homogenizing the information of the \(n = 4\) residual streams.

<figure>
  <img src="../../../posts/residual/ihc_hres_accumulation.png" alt="Accumulated multi-layer H_res matrix" loading="lazy" width="100%" />
  <figcaption>The accumulation of multi-layer \(\mathcal{H}^{\mathrm{res}}\): after 10 layers, the information of the 4 streams collapses entirely into uniform mixing. Source: <a href="https://zhuanlan.zhihu.com/p/2010852389670908320">Zhihu post “Your DeepSeek mHC May Not Need the ‘m’”</a>.</figcaption>
</figure>

The mathematical reason behind this: for a doubly stochastic matrix satisfying the uniform positivity condition (all entries have a positive lower bound \(\delta > 0\)), its Dobrushin ergodic coefficient satisfies \(\tau(P) \leq 1 - d\delta < 1\), so the coefficient of the accumulated product decays geometrically, \(\tau(A_n) \leq (1 - d\delta)^n \to 0\), forcing all rows to converge to each other and the product to converge to the uniform matrix \(\frac{1}{d}\mathbf{1}\mathbf{1}^\top\). Pure permutation matrices or reducible matrices do not satisfy the uniform positivity condition and do not collapse—but Sinkhorn's output is usually strictly positive, satisfying the condition.

The Perron-Frobenius theorem gives the same conclusion: a doubly stochastic matrix has largest eigenvalue 1 and all other eigenvalues with magnitude strictly less than 1 (as long as it is not a reducible permutation matrix), so the smallest singular value of the accumulated product satisfies

\[\sigma_{\min}\left(\prod_{l=1}^L H_l\right) \lesssim \prod_{l=1}^L |\lambda_{\min}(H_l)|\]

Here \(\lambda_{\min}(H_l)\) is the eigenvalue of smallest magnitude of each layer's doubly stochastic matrix—an eigenvalue satisfies \(H_l v = \lambda v\); a doubly stochastic matrix's largest eigenvalue is 1 and all other eigenvalues have magnitude strictly less than 1—and \(\sigma_{\min}\) is the smallest singular value, i.e., the smallest stretch factor when the matrix stretches vectors. For any matrix, the smallest singular value never exceeds the magnitude of any eigenvalue, so the original post gives the heuristic bound, written as \(\lesssim\), that the smallest singular value of the \(L\)-layer product is bounded by the product of the smallest-magnitude eigenvalues of the individual layers. A \(\sigma_{\min}\) approaching 0 means that in some direction, the signal is (almost) completely compressed away after many layers—exactly the signal vanishing discussed in Section 3.1.

The product decays exponentially to zero when \(|\lambda_{\min}| < 1\). On Qwen3-1.7B (28 layers, 56 HC modules), the mean smallest eigenvalue of the Sinkhorn version of \(\mathcal{H}^{\mathrm{res}}\) is 0.49, giving an estimate of \(\sigma_{\min} \sim 0.49^{56} \approx 10^{-17}\), with a measured value of \(9.2 \times 10^{-18}\)—after passing through 56 HC modules, a shallow-layer signal decays to almost nothing except along the mean direction.

In addition, 20 steps of Sinkhorn-Knopp iteration do not guarantee convergence: the measured standard deviation of row sums is 0.12, and this error accumulates across layers; mHC lite also reports that for about 27.9% of inputs the relative range satisfies \(1/\nu \geq 10^{13}\), in which case the column-sum deviation after 20 iterations can reach 100%.

### 4.2. The Advantage of Identity

The identity matrix is itself doubly stochastic (row and column sums equal 1, spectral norm equal to 1, fully norm-preserving)—the simplest possible manifold constraint. \(H^{\mathrm{res}} = I\) means: each residual stream keeps its own information and does not exchange it with other streams.

One might object: doesn't this degenerate to a permutation matrix? The problem is that mHC's \(\mathcal{H}^{\mathrm{res}}\) at different layers are **different approximate permutations**—each layer rearranges the streams, so stream 1 becomes the stream at position 3 after layer 1 and the stream at position 2 after layer 5, and \(\mathcal{H}^{\mathrm{pre}}\) and \(\mathcal{H}^{\mathrm{post}}\) must constantly “track” where each stream has been rearranged to, increasing the learning difficulty. The advantages of identity are:

- Stream 0 is always at position 0—stream semantics remain consistent along depth;
- \(\mathcal{H}^{\mathrm{pre}}\) / \(\mathcal{H}^{\mathrm{post}}\) do not need to adapt to stream rearrangement and directly learn “which stream to read from and which stream to write to”;
- The accumulated product \(I^L = I\) neither collapses nor scrambles.

Cross-stream information mixing does not disappear: the \(\varphi\) projection flattens the \(n\) streams and projects them to \(n^2 + 2n\) dimensions, then splits out \(\mathcal{H}^{\mathrm{pre}}\) (\(n\) dimensions), \(\mathcal{H}^{\mathrm{post}}\) (\(n\) dimensions), and \(\mathcal{H}^{\mathrm{res}}\) (\(n^2\) dimensions). Even with \(H^{\mathrm{res}} = I\), the \(\mathcal{H}^{\mathrm{pre}}\) produced by \(\varphi\) is still input-dependent, and the aggregation and write-back are done dynamically with sigmoid weighting (the update formula is shown at the beginning of this section).

### 4.3. Comparison and Failed Alternatives

| Metric | Sinkhorn | Identity |
|--------|----------|----------|
| Accumulated product | rank-1 collapse (\(\kappa = 10^{17}\)) | \(I\) (\(\kappa = 1\)) |
| Approximation error | row-sum std = 0.12 | exact |
| Extra computation | 20 iterations + backward recomputation | zero |
| Extra parameters | \(nC \times n^2\) projection weights | none |
| Signal propagation | shallow signals decay exponentially | lossless |

The author also tried other alternatives, none of which beat the original mHC:

- **mHC lite (exact doubly stochastic via convex combination, softmax weighting)**: on 1.7B it underperforms the original mHC. It was observed that as \(\alpha_{\mathrm{res}}\) grows (from 0.01 to around 2), the softmax temperature drops and the output tends toward one-hot, so stream mixing actually decreases.
- **Orthogonalization (Cayley transform, Givens rotations)**: the spectral norm is always 1, neither exploding nor vanishing; but \(\alpha_{\mathrm{res}}\) barely moves (stays around 0.01), and allowing negative values can flip some streams, causing capacity collapse.

This approach has also landed in production: the residual pathway of Tencent Hunyuan 4 Preview ([Hy4-preview](https://huggingface.co/tencent/Hy4-preview)) is exactly iHC—the official model card states that “the residual pathway uses iHC (identity Hyper-Connections) to expand inter-layer information flow.” The model has 4 residual streams, drops the Sinkhorn constraint, and fixes \(H^{\mathrm{res}}\) to the identity mapping.
