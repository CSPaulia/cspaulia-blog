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
