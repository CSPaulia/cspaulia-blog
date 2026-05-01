---
title: "Residual Connections and Their Variants"
date: 2026-04-29T11:30:03+08:00
series:
    main: "Large Language Models"
    subseries: "Architecture & Training"
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

{{< figure src="hc.png" alt="HC" caption="Hyper-Connection Diagram" >}}

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

{{< figure src="hc_io.png" alt="hc" >}}

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

As \(t \to \infty\), \(\mathbf{M}^{(t)}\) converges to a doubly stochastic matrix.

### 3.3. Infrastructure Design

1. **Kernel Fusion**: Too many small operations cause repeated HBM reads/writes and large kernel-launch overhead. Fusing multiple small operations into a single kernel reduces memory access and launch overhead.
2. **Recomputation**: Intermediate quantities are not stored during the forward pass; they are recomputed during backpropagation.
3. **Communication-Computation Overlap**: \(n\) information streams increase communication cost; overlapping communication with computation mitigates this overhead.
