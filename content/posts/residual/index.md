---
title: "残差连接及其变体"
date: 2026-04-29T11:30:03+08:00
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
description: "残差连接及其变体"
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
    image: "hc.png" # image path/url
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

## 1. 残差连接（Residual Connection）

多说无益，直接上公式：

\[
    h_{l+1} = h_l + \mathcal{T}(h_l)
\]

其中，$h_l$ 是第 $l$ 层的输入，$\mathcal{T}$ 是一个神经网络层（如卷积层、注意力层或全连接层）。

## 2. 超连接（Hyper-Connection，HC）

> 先吐槽一下，原文写的并不够清晰，甚至有些混乱...

思想：[超连接](http://arxiv.org/abs/2409.19606)将残差连接转变为“矩阵读写”形式，然后将固定矩阵转变为可学习矩阵。

\[
    \mathbf{H}^{l+1} = \mathbf{A}_r \mathbf{H}^l + \mathbf{B} \mathcal{T}(\mathbf{A}_m^{\top} \mathbf{H}^l)
\]

{{< figure src="hc.png" alt="HC" caption="超连接示意图" >}}

### 2.1. “单条残差流”到“多条残差流”的转变

传统的残差只有一条残差流，该残差流对隐层信息 \(\mathbf{h}^l\) 维护。

而超连接引入了多条残差流，每条残差流维护不同的隐层信息。具体做法是，对于初始输入 \(\mathbf{h}^0\)，根据超参数拓展率（expansion rate） \(n\) 将其复制 \(n\) 份，得到 \(n\) 条残差流：

\[
    \begin{aligned}
        \mathbf{h}^0_1 &= \mathbf{h}^0 \\
        \mathbf{h}^0_2 &= \mathbf{h}^0 \\
        &\vdots \\
        \mathbf{h}^0_n &= \mathbf{h}^0
    \end{aligned}
\]

每一条残差流均完成残差连接：

\[
    \mathbf{h}^{l+1}_i = \mathbf{h}^l_i + \mathcal{T}(\mathbf{h}^l_i), \quad i = 1, 2, \ldots, n
\]

我们可以将残差流写成一个矩阵的形式：

\[
    \mathbf{H}^l = \begin{bmatrix}
        \mathbf{h}^l_1 \\
        \mathbf{h}^l_2 \\
        \vdots \\
        \mathbf{h}^l_n
    \end{bmatrix}
\]

### 2.2. 残差流的信息整合

对于神经网络层 \(\mathcal{T}\)，往往只接受单个隐层输入 \(\mathbf{h}^l\)，而不是矩阵 \(\mathbf{H}^l\)。因此，需要将矩阵 \(\mathbf{H}^l\) 中的多条残差流整合成一个隐层输入：

\[
    \mathbf{h}^l_0 = \alpha_1 \mathbf{h}^l_1 + \alpha_2 \mathbf{h}^l_2 + \ldots + \alpha_n \mathbf{h}^l_n. = \mathbf{A}_m^{\top} \mathbf{H}^l
\]

其中，\(\alpha_i\) 是可学习的权重，\(\mathbf{A}_m = [\alpha_1, \alpha_2, \ldots, \alpha_n]^\top\) 是一个可学习的权重矩阵。

### 2.3. 新信息和旧信息的融合

新信息为：

\[
    \mathcal{T}(\mathbf{h}^l_0) = \mathcal{T}(\mathbf{A}_m^{\top} \mathbf{H}^l)
\]

旧残差流信息为：

\[
    \mathbf{H}^l
\]

直接相加有点小无聊，我们可以引入一个新的可学习权重矩阵 \(\mathbf{A}_r \in \mathbb{R}^{n \times n}\) 来调整旧信息的权重；引入另一个可学习权重矩阵 \(\mathbf{B} \in \mathbb{R}^{n}\) 来调整新信息的权重：

\[
    \mathbf{H}^{l+1} = \mathbf{A}_r \mathbf{H}^l + \mathbf{B} \mathcal{T}(\mathbf{A}_m^{\top} \mathbf{H}^l)
\]

这便是固定超连接（Static Hyper-Connection，SHC）的核心公式。

### 2.4. 动态超连接（Dynamic Hyper-Connection，DHC）

直接上公式吧，论文中没讲它们的设计初衷是什么...：

\[
    \overline{\mathbf{H}} = \text{norm}(\mathbf{H}) \\
    \mathcal{B} = s_\beta \cdot \text{tanh}(\mathbf{W}_\beta \overline{\mathbf{H}}) + \mathbf{B} \\
    \mathcal{A}_m = s_\alpha \cdot \text{tanh}(\mathbf{W}_m \overline{\mathbf{H}}) + \mathbf{A}_m \\
    \mathcal{A}_r = s_\alpha \cdot \text{tanh}(\mathbf{W}_r \overline{\mathbf{H}}) + \mathbf{A}_r \\
\]

其中动态参数 \(\mathbf{W}_\beta\), \(\mathbf{W}_m\), 和 \(\mathbf{W}_r\) 是可学习的权重矩阵。\(s_\alpha\) 和 \(s_\beta\) 是可学习因子。

\[
    \mathbf{H}^{l+1} = \mathcal{A}_r \mathbf{H}^l + \mathcal{B} \mathcal{T}(\mathcal{A}_m^{\top} \mathbf{H}^l)
\]

### 2.5. 优势

在参数量并没有明显增加的情况下，超连接能够显著提升模型性能。

## 3. 流形约束超连接（Manifold-Constrained Hyper-Connection，mHC）

### 3.1. 超连接的问题

单层超连接：

\[
    \mathbf{H}^{l+1} = \mathbf{A}_r \mathbf{H}^l + \mathbf{B} \mathcal{T}(\mathbf{A}_m^{\top} \mathbf{H}^l)
\]

在 [mHC](https://arxiv.org/abs/2512.24880) 的论文中，将 \(\mathbf{H}^{l+1}\) 写作 \(\mathbf{x}_{i+1}\)，\(\mathbf{H}^l\) 写作 \(\mathbf{x}_i\)，\(\mathbf{A}_r\) 写作 \(\mathcal{H}^{res}_i\)，\(\mathbf{B}\) 写作 \({\mathcal{H}^{post}_i}^\top\)，\(\mathbf{A}_m\) 写作 \(\mathcal{H}^{pre}_i\)，\(\mathcal{T}\) 写作 \(\mathcal{F}_i\)，则单层超连接可以写作：

\[
    \mathbf{x}_{i+1} = \mathcal{H}^{res}_i \mathbf{x}_i + {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i)
\]

这样的写法，我认为更加的易于理解。

两层超连接可以写为：

\[
    \mathbf{x}_{i+2} = \mathcal{H}^{res}_{i+1} \mathbf{x}_{i+1} + {\mathcal{H}^{post}_{i+1}}^\top \mathcal{F}_{i+1}(\mathcal{H}^{pre}_{i+1} \mathbf{x}_{i+1}) \\
    = \mathcal{H}^{res}_{i+1} (\mathcal{H}^{res}_i \mathbf{x}_i + {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i)) + {\mathcal{H}^{post}_{i+1}}^\top \mathcal{F}_{i+1}(\mathcal{H}^{pre}_{i+1} \mathbf{x}_{i+1}) \\
    = \mathcal{H}^{res}_{i+1} \mathcal{H}^{res}_i \mathbf{x}_i + \mathcal{H}^{res}_{i+1} {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i) + {\mathcal{H}^{post}_{i+1}}^\top \mathcal{F}_{i+1}(\mathcal{H}^{pre}_{i+1} \mathbf{x}_{i+1})
\]

不难推导出，多层超连接可以写为：

\[
    \mathbf{x}_{L} = \mathcal{H}^{res}_{L-1} \cdots \mathcal{H}^{res}_{L-l} \mathbf{x}_l + \sum_{i=l}^{L-1} \left( \mathcal{H}^{res}_{L-1} \cdots \mathcal{H}^{res}_{i+1} {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i) \right) \\
    = \left( \prod_{i=l}^{L-1} \mathcal{H}^{res}_i \right) \mathbf{x}_l + \sum_{i=l}^{L-1} \left( \prod_{j=1}^{L-1-i} \mathcal{H}^{res}_{L-j} \right) {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i)
\]

其中，旧信息流为 \(\left( \prod_{i=l}^{L-1} \mathcal{H}^{res}_i \right) \mathbf{x}_l\)，新信息流为 \(\sum_{i=l}^{L-1} \left( \prod_{j=1}^{L-1-i} \mathcal{H}^{res}_{L-j} \right) {\mathcal{H}^{post}_i}^\top \mathcal{F}_i(\mathcal{H}^{pre}_i \mathbf{x}_i)\)。

更其中，旧信息流的系数为 \(\prod_{i=l}^{L-1} \mathcal{H}^{res}_i\)。在经典的残差连接中，由于旧信息的系数为 1，因此旧信息流的范数不会发生变化；而在超连接中，旧信息流的系数为 \(\prod_{i=l}^{L-1} \mathcal{H}^{res}_i\)，因此旧信息流的范数可能会发生变化，会对模型的信息和梯度的传递产生影响。

<details>
<summary>关于范数对梯度和信息的影响</summary>
观察 \(\mathcal{H}^{res}_{L-1} \cdots \mathcal{H}^{res}_{L-l} \mathbf{x}_l\) 的范数上界：

\[
    \| \mathcal{H}^{res}_{L-1} \cdots \mathcal{H}^{res}_{L-l} \mathbf{x}_l \| \leq \left( \prod_{i=l}^{L-1} \| \mathcal{H}^{res}_i \| \right) \| \mathbf{x}_l \|
\]

假如 \(\| \mathcal{H}^{res}_i \| = 0.9\)，那么经过 100 层后，旧信息流的范数上界为 \(0.9^{100} \| \mathbf{x}_l \| \approx 2.6561 \times 10^{-5} \| \mathbf{x}_l \|\)，信息的传递几乎消失；假如 \(\| \mathcal{H}^{res}_i \| = 1.1\)，那么经过 100 层后，旧信息流的范数上界为 \(1.1^{100} \| \mathbf{x}_l \| \approx 13780.6123 \| \mathbf{x}_l \|\)，信息就会被放大一万多倍。

反向传播时，梯度会乘雅可比矩阵的转置：

\[
    \frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \left( \frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l} \right)^\top \left( \frac{\partial \mathcal{x}_{l+2}}{\partial \mathbf{x}_{l+1}} \right)^\top \cdots \left( \frac{\partial \mathbf{x}_{L}}{\partial \mathbf{x}_{L-1}} \right)^\top \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \\
    = J_l^\top J_{l+1}^\top \cdots J_{L-1}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}
\]

如果这些雅可比矩阵的奇异值多数小于 1，梯度往浅层传时就会越来越小，也就是梯度消失。
如果多数大于 1，就会越来越大，也就是梯度爆炸。
</details>

mHC 论文证明了 HC 的训练不够稳定，同时由于额外的学习参数的存在，HC 受内存访问的限制极大影响模型的训练速度。

{{< figure src="hc_io.png" alt="hc" >}}

### 3.2. mHC 的设计

为了解决 HC 的训练不稳定的问题，mHC 限制 \(\mathcal{H}^{\mathrm{res}}_l\) 在一个特殊集合里：

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

我们称这样的矩阵为双随机矩阵（Doubly Stochastic Matrix）。它的好处是：
1. 范数上限为：\(\| \mathcal{H}^{\mathrm{res}}_l \| \leq 1\)，避免梯度爆炸；
2. 复合封闭性：两个双随机矩阵的乘积仍然是双随机矩阵。
3. 符合 Birkhoff 多面体（Birkhoff Polytope）的性质：所有双随机矩阵都可以写成若干置换矩阵（Permutation Matrices）的凸组合。也就是说 \(\mathcal{P}_{\mathcal{M}^{\mathrm{res}}} \left(\mathcal{H}^{\mathrm{res}}_l\right) \) 可能写作：

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
<summary>置换矩阵</summary>
置换矩阵就是每一行、每一列恰好有一个 1，其余都是 0 的矩阵。例如，以下是一个 4x4 的置换矩阵：
\[
    P = \begin{bmatrix}
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1 \\
        1 & 0 & 0 & 0
    \end{bmatrix}
\]
</details>

\(\mathcal{H}^{\mathrm{res}}_l\) 不是任意矩阵，而是多条残差信息流的“概率式软置换 / 软混合”。这种约束让多层残差映射更稳定、更可解释，也更不容易梯度爆炸。

mHC 的完整设计为：

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

其中 \(\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}\) 是 \(\mathbf{x}_l \in \mathbb{R}^{n \times C}\) 的展开形式。\(\varphi^{\cdot}_l\) 是线性变换的权重矩阵，\(\mathbf{b}^{\cdot}_l\) 是偏置项，\(\alpha^{\cdot}_l\) 是可学习的缩放因子。\(\sigma\) 是 Sigmoid 函数，\(\mathrm{mat}\) 是将向量（\(\mathbb{R}^{1 \times n^2}\)）转换为矩阵（\(\mathbb{R}^{n \times n}\)）的操作，\(\mathrm{Sinkhorn\text{-}Knopp}\) 是一个将矩阵转换为双随机矩阵的算法。\(\mathrm{Sinkhorn\text{-}Knopp}\) 的具体实现为：

\[
    \mathbf{M}^{(0)} = \exp\!\left(\tilde{\mathcal{H}}^{\mathrm{res}}_l\right), \\
    \mathbf{M}^{(t)} = \mathcal{T}_r\!\left(\mathcal{T}_c\!\left(\mathbf{M}^{(t-1)}\right)\right)
\]

当 \(t \to \infty\) 时，\(\mathbf{M}^{(t)}\) 会收敛到一个双随机矩阵。

### 3.3. mHC 的工程设计（Infrastructure Design）

1. 计算核融合（Kernel Fusion）：小操作太多，反复读写显存，kernel launch overhead 大；将多个小操作融合成一个大操作，减少显存访问和 kernel launch overhead。
2. 重计算（Recomputation）：前向不存中间量，反向时重新计算。
3. 通信计算重叠（Communication-Computation Overlap）：n 个信息流通信开销增大；通信和计算重叠，减少通信开销。