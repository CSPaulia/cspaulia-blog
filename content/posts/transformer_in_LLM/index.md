---
title: "LM Architecture and Training"
date: 2025-10-10T10:00:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "Large Language Model"
    subseries: "Architecture and Training"
draft: false
categories: ["Large Language Model"]
tags: ["LLM", "Training"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "关于语言模型架构和超参数的笔记"
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: true
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "LLMs.jpg" # image path/url
    alt: "LM Architecture and Training" # alt text
    caption: "LM Architecture and Training" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

一篇非常棒的[博客](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)，它总结了几种流行的大模型之间的架构差异

## 1. 原始Transformer vs 现代变体

下表总结了原始Transformer（Vaswani et al., 2017）与现代大语言模型（LLM）中的主流Transformer变体在架构和训练细节上的主要区别：

| 方面                | 原始Transformer (2017)         | 现代LLM中的变体 |
|---------------------|---------------------------------|---------------------------------------------|
| 归一化顺序          | 后归一化（Post-LN）             | 前归一化（Pre-LN）                          |
| 激活函数            | ReLU                            | SwiGLU（GELU、SiLU、Swish等）                         |
| Dropout             | 广泛使用                        | 训练大模型时常常减小或去除                  |
| 归一化类型          | LayerNorm                       | RMSNorm（LayerNorm、ScaleNorm等）             |
| 线性层              | 添加偏置                        | 不添加偏置                                   |
| 注意力头            | 多头注意力（固定头数）           | GQA、MQA等           |
| 位置编码            | 绝对位置编码（sinusoidal）       | RoPE等      |
| 其它                | -                               | FlashAttention, MoE, 分层并行等              |

### 1.1. 前归一化 vs 后归一化

几乎所有的现代语言模型均采用前归一化（除了BERT），能使训练更加稳定

左图为前归一化（pre-norm），右图为后归一化（post-norm）

<img src="pre-post-norm.png" alt="pre-vs-post" width="300"/>

**新！**：左图为前归一化（pre-norm），右图为双归一化（'double' norm，使用者包括 Grok，Gemma 2）

<img src="pre-post-norm.png" alt="pre-vs-post" width="300"/>

**新！**：Olom 2 仅使用非残差部分的后归一化

### 1.2. LayerNorm vs RMSNorm

原始 Transformer：**LayerNorm** (GPT3/2/1，OPT，GPT-J，BLOOM)

$$
y = \frac{x - \textbf{E}[x]}{\sqrt{\textbf{Var}[x] + \epsilon}} * \gamma + \beta
$$

现代语言模型：**RMSNorm**（LLaMA-family，PaLM，T5）

$$
y = \frac{x}{\sqrt{\lVert x \rVert^2_2 + \epsilon}} * \gamma
$$

RMSNorm的优势：运行速度更快，而并不影响精度
- 更少的 Operations（无需计算均值）
- 更少的参数（没有偏置项需要存储）

### 1.3. FFN：有偏置 vs 无偏置

原始 Transformer：有偏置

$$
\textbf{FFN}(x) = \max(0,xW_1+b_1)W_2+b_2
$$

现代语言模型：无偏置

$$
\textbf{FFN}(x) = \sigma(xW_1)W_2
$$

无偏置的优势：更小的存储开销以及稳定的优化

### 1.4. 激活函数

| Activation | Model |
| :----------: | :-----: |
| ReLU | Original transformer, T5, Gopher, Chinchilla, OPT |
| GeLU | GPT1/2/3, GPTJ, GPT-Neox, BLOOM |
| GeGLU | T5 v1.1, mT5, LaMDA, Phi3, Gemma 2, Gemma 3 |
| SwiGLU | LLaMa 1/2/3, PaLM, Mistral, OlMo, most models post 2023 |

激活函数的介绍详见 [Post](../activation/)