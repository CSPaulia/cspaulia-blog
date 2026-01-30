---
title: "LoRA: Low-Rank Adaptation for LLM Fine-tuning"
date: 2025-06-18T10:00:00+08:00
series:
    main: "Large Language Model"
    subseries: "Fine-tuning"
categories: ["Deep Learning Skills", "Large Language Model"]
tags: ["LoRA", "fine-tuning"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "A detailed walkthrough of LoRA (Low-Rank Adaptation) for parameter-efficient LLM fine-tuning"
UseHugoToc: true
cover:
    image: "lora-cover.png" 
    alt: "LoRA architecture" 
    caption: "LoRA architecture" 
    relative: false
    hidden: true
---

## LoRA

<p align="center">
  {{< img src="lora_overview.png" alt="lora_overview" width="60%" >}}
</p>

> LoRA is a parameter-efficient fine-tuning method for large language models, introduced in the paper
> [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685).
> With LoRA, the number of trainable parameters can be reduced to around $10^{-4}$ of the full model; GPU memory usage can drop by about 2/3, and it does not add extra inference latency.

### Key ideas

1. **Low-rank decomposition**: instead of updating the full pretrained weights, learn a low-rank update
2. **Parameter efficiency**: drastically reduces the number of trainable parameters
3. **Keep base weights frozen**: the original pretrained weights stay unchanged; only the added low-rank matrices are trained

### How it works

#### 1. The idea of parameter-efficient fine-tuning

- **Full fine-tuning**

  For a language model, we initialize the model with pretrained parameters $\Phi_0$ and update parameters by maximizing the conditional language modeling likelihood, i.e., learn $\Phi_0 + \Delta \Phi$:

    $$
    \max_{\Phi} \sum_{(x,y)\in\mathcal{Z}} \sum^{\lvert y \rvert}_{t=1} \log{(P^{\Phi}(y_t|x,y < t))}
    $$

    The main downside is that the learned update $\Delta \Phi$ has the same dimensionality as the original parameters $\Phi_0$, which is expensive in compute and memory. This is typically called **full fine-tuning**.

  - **Parameter-efficient fine-tuning**

    A common idea is to represent the update with a much smaller set of parameters: $\Delta \Phi = \Delta \Phi(\Theta)$, where $\lvert \Theta \rvert \ll \lvert \Phi_0 \rvert$. The optimization objective becomes:

    $$
    \max_{\Theta} \sum_{(x,y)\in\mathcal{Z}} \sum^{\lvert y \rvert}_{t=1} \log{(P^{\Phi_0 + \Delta \Phi(\Theta)}(y_t|x,y < t))}
    $$


  This family of methods is called **parameter-efficient fine-tuning** (PEFT). There are many approaches (e.g., Adapters, prefix-tuning, etc.). LoRA encodes the update using low-rank matrices; compared with some alternatives, it does not add inference latency and is easy to optimize.

#### 2. 数学表达

> The work by [Aghajanyan et al.](https://arxiv.org/abs/2012.13255) suggests that pretrained models can have a small intrinsic dimension: there may exist a very low-dimensional subspace such that fine-tuning within it can achieve comparable performance to full-space fine-tuning.

Given an original weight matrix $W_{\theta} \in \mathbb{R}^{d \times k}$, LoRA updates it as:

$$
W = W_{\theta} + BA
$$

where:
- $B \in \mathbb{R}^{d \times r}$，初始化为0矩阵
- $A \in \mathbb{R}^{r \times k}$，初始化为高斯分布矩阵
- $r \ll \min{(k, d)}$，r 是秩 (rank),通常远小于 d 和 k
- 初始化$W = W_{\theta} + \mathbf{0} \times A = W_{\theta}$

In words:
- $B \in \mathbb{R}^{d \times r}$ is initialized as a zero matrix
- $A \in \mathbb{R}^{r \times k}$ is initialized from a Gaussian distribution
- $r \ll \min{(k, d)}$ is the rank, typically much smaller than $d$ and $k$
- initially $W = W_{\theta} + \mathbf{0} \times A = W_{\theta}$

#### 3. Why low-rank helps

- **Fewer parameters**: from $d \times k$ down to $r \times (d+k)$
- **Better memory efficiency**: enables training with smaller GPU memory
- **Better compute efficiency**: reduces overall compute cost

---

<div class="zhihu-ref">
  <div class="zhihu-ref-title">References</div>
  <ol>
    <li><a href="https://arxiv.org/abs/2106.09685" target="_blank">LoRA: Low-Rank Adaptation of Large Language Models</a></li>
    <li><a href="https://zhuanlan.zhihu.com/p/646791309" target="_blank">LORA微调系列(一)：LORA和它的基本原理</a></li>
  </ol>
</div>