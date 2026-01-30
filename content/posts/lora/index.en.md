---
title: "LoRA: 大模型低秩适配方法详解"
date: 2025-06-18T10:00:00+08:00
series:
    main: "Large Language Model"
    subseries: "Fine-tuning"
categories: ["Deep Learning Skills", "Large Language Model"]
tags: ["LoRA", "fine-tuning", "LLM"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "详细解读 LoRA (Low-Rank Adaptation) 方法在大模型微调中的应用"
UseHugoToc: true
cover:
    image: "lora-cover.png" 
    alt: "LoRA Architecture" 
    caption: "LoRA Architecture" 
    relative: false
    hidden: true
---

## LoRA

<p align="center">
  {{< img src="lora_overview.png" alt="lora_overview" width="60%" >}}
</p>

> LORA是一种低资源**微调**大模型方法，出自论文[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)。 
> 使用LORA，训练参数仅为整体参数的万分之一、GPU显存使用量减少2/3且不会引入额外的推理耗时.

### 核心思想

1. **低秩分解**: 不直接更新预训练模型的权重,而是学习低秩分解矩阵
2. **参数效率**: 显著减少需要训练的参数数量
3. **原始权重保持**: 保持预训练模型的权重不变,只训练额外的低秩矩阵

### 技术原理

#### 1. 高效微调基本原理

- **full fine-tuing**

    以语言模型为例，在微调过程中模型加载预训练参数$\Phi_0$进行初始化,并通过最大化条件语言模型概率进行参数更新$\Phi_0 + \Delta \Phi$，即：

    $$
    \max_{\Phi} \sum_{(x,y)\in\mathcal{Z}} \sum^{\lvert y \rvert}_{t=1} \log{(P^{\Phi}(y_t|x,y < t))}
    $$

    这种微调方式主要的缺点是我们学习到的参数增量$\Delta \Phi$的维度和预训练参数$\Phi_0$是一致的，这种微调方式所需的资源很多，一般被称为**full fine-tuing**。

- **高效微调**

    研究者认为可以用更少的参数表示上述要学习的参数增量$\Delta \Phi = \Delta \Phi(\Theta)$，其中$\lvert \Theta \rvert \ll \lvert \Phi_0 \rvert$，原先寻找的优化目标变为寻找：

    $$
    \max_{\Theta} \sum_{(x,y)\in\mathcal{Z}} \sum^{\lvert y \rvert}_{t=1} \log{(P^{\Phi_0 + \Delta \Phi(\Theta)}(y_t|x,y < t))}
    $$


    这种仅微调一部分参数的方法称为**高效微调**。针对高效微调，研究者有很多的实现方式(如Adapter、prefixtuing等)。本文作者旨在使用一个低秩矩阵来编码，相比于其他方法，LORA不会增加推理耗时且更便于优化。

#### 2. 数学表达

> [Aghajanyan的研究](https://arxiv.org/abs/2012.13255)表明：预训练模型拥有极小的内在维度(instrisic dimension)，即存在一个极低维度的参数，微调它和在全参数空间中微调能起到相同的效果。

给定原始权重矩阵 $W_{\theta} \in \mathbb{R}^{d \times k}$, LoRA 通过以下方式进行参数更新:

$$
W = W_{\theta} + BA
$$

其中:
- $B \in \mathbb{R}^{d \times r}$，初始化为0矩阵
- $A \in \mathbb{R}^{r \times k}$，初始化为高斯分布矩阵
- $r \ll \min{(k, d)}$，r 是秩 (rank),通常远小于 d 和 k
- 初始化$W = W_{\theta} + \mathbf{0} \times A = W_{\theta}$

#### 3. 低秩分解的优势

- **参数量减少**: 从 $d \times k$ 减少到 $r \times (d+k)$
- **内存效率**: 可以使用更小的显存进行训练
- **计算效率**: 降低了计算复杂度

---

<div class="zhihu-ref">
  <div class="zhihu-ref-title">参考文献</div>
  <ol>
    <li><a href="https://arxiv.org/abs/2106.09685" target="_blank">LoRA: Low-Rank Adaptation of Large Language Models</a></li>
    <li><a href="https://zhuanlan.zhihu.com/p/646791309" target="_blank">LORA微调系列(一)：LORA和它的基本原理</a></li>
  </ol>
</div>