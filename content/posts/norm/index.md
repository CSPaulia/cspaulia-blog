---
title: "收集 N 种归一化方法"
date: 2025-05-21T21:15:00+08:00
series:
    main: "深度学习基础"
    subseries: "归一化方法"
# weight: 1
# aliases: ["/first"]
categories: ["深度学习技巧"]
tags: ["归一化"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false # show table of contents
draft: false
hidemeta: false
comments: false
description: "[Epoch 1/100] Updating..."
# canonicalURL: "https://canonical.url/to/page"
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "norm_cover.png" # image path/url
    alt: "归一化方法概览" # alt text
    caption: "归一化方法" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "修改建议" # edit text
    appendFilePath: true # to append file path to Edit link
---

<p align="center">
  {{< img src="LNvsBN.jpg" alt="LNvsBN" >}}
</p>

在上图中，\(N\) 表示样本轴，\(C\) 表示通道轴，\(F\) 表示每个通道的特征数量。BN 取<strong>不同样本的同一个通道</strong>进行归一化；LN 则取<strong>同一个样本的不同通道</strong>进行归一化。

## 1. 批归一化（Batch Normalization，BN）

### 1.1 BN 与批大小

BN 根据一个批次中的样本计算归一化统计量。当批大小很小时，例如一个批次中只有 4 个样本，这些样本的均值和方差可能无法代表整体数据分布，因此 BN 的效果会变差。

### 1.2 BN 与循环神经网络

<p align="center">
  {{< img src="RNN.jpg" alt="RNN" >}}
</p>

循环神经网络（Recurrent Neural Network，RNN）的一个批次中，各个样本的序列长度通常不同。当计算到较后的时间步时，例如上图中的 \(t>4\)，可能只剩一个样本仍有数据。此时得到的统计量无法代表整体分布，BN 的效果并不好。

另外，如果测试序列比所有训练序列都长，较后时间步没有训练阶段保存的统计量，也会给 BN 的使用带来困难。

## 2. 层归一化（Layer Normalization，LN）

### 2.1 MLP 中的 LN

先看多层感知机（Multi-Layer Perceptron，MLP）中的 LN。设 \(H\) 是一层中的隐藏节点数，\(l\) 是层编号，可以计算归一化统计量 \(\mu\) 和 \(\sigma\)：

\[
\mu^{l} = \frac{1}{H} \sum_{i=1}^{H} a^l_i ~~~~~~~
\sigma^{l} = \sqrt{\frac{1}{H} \sum_{i=1}^{H}(a^l_i-\mu^l)^2}
\]

这些统计量的计算与批次中的样本数量无关，只取决于当前样本的隐藏节点。通过 \(\mu^{l}\) 和 \(\sigma^{l}\)
可以得到归一化后的值：

\[
\hat{a}^l = \frac{a^l-\mu^l}{\sqrt{(\sigma^l)^2+\epsilon}} \tag{1}
\]

其中 \(\epsilon\) 是一个很小的常数，用于避免除零。

LN 还使用增益（gain）\(g\) 和偏置（bias）\(b\) 恢复模型所需的尺度与平移。假设激活函数为 \(f\)，最终输出为：

\[
h^l = f(g^l \odot \hat{a}^l + b^l) \tag{2}
\]

合并公式（1）和公式（2），并省略层编号 \(l\)，得到：

\[
h=f(\frac{g}{\sqrt{\sigma^2+\epsilon}} \odot (a-\mu) + b)
\]

### 2.2 RNN 中的 LN

对于 RNN 在时间步 \(t\) 的节点，其输入是 \(t-1\) 时刻的隐藏状态 \(h^{t-1}\) 和当前输入 \(x^t\)，可以表示为：

\[
a^t = W_{hh}h^{t-1}+W_{xh}x^t.
\]

随后可以对 \(a^t\) 使用与 MLP 中相同的归一化过程：

\[
h^t=f(\frac{g}{\sqrt{\sigma^2+\epsilon}} \odot (a^t-\mu^t) + b) ~~~~~~
\mu^{t} = \frac{1}{H} \sum_{i=1}^{H} a^t_i ~~~~~~~
\sigma^{t} = \sqrt{\frac{1}{H} \sum_{i=1}^{H}(a^t_i-\mu^t)^2}
\]

## 3. 均方根归一化（Root Mean Square Layer Normalization，RMSNorm）

RMSNorm 不减去均值，只根据输入的均方根缩放特征。对于 \(d\) 维输入 \(x\)，其计算为

\[
\begin{aligned}
\operatorname{RMS}(x)
&=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},\\
y_i
&=g_i\frac{x_i}{\operatorname{RMS}(x)}.
\end{aligned}
\]

其中 \(g_i\) 是可学习的增益，通常与特征维度等长。与 LN 相比，RMSNorm 不计算均值，也不执行中心化，只控制输入的整体幅度。因此，它保留了 LN 对批大小不敏感的特点，同时减少了均值计算和减法操作。

## 参考文献

[1] Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. [Online]. Available: https://arxiv.org/abs/1910.07467
