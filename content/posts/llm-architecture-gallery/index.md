---
title: "LLM 架构图谱 (2024–2026)"
date: 2026-04-28T11:30:03+08:00
series:
    main: "大语言模型"
    subseries: "架构与训练"
categories: ["大语言模型"]
tags: ["架构", "注意力", "MoE"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "2024–2026 年主流开源 LLM 架构一览，含注意力机制、MoE、SSM 等维度对比。"
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
    image: "<image path/url>"
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

基于 [Sebastian Raschka 的 LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 整理，覆盖 2024 至 2026 年发布的 38 个开源/开放权重模型。注意力列和架构列链接到本站详细技术文章。

> **图例**：D = Dense（密集），M = MoE（稀疏混合专家），H = Hybrid（混合架构）。参数量记为 总参数量 / 激活参数量（MoE 模型）。
> Norm 列注明类型及放置位置（pre = 子层前，post = 子层后）。

## 2024

| 模型 | 开发者 | 参数量 | 架构 | 注意力 | Norm | 位置编码 | 上下文 |
|------|--------|--------|------|--------|------|---------|--------|
| Llama 3 8B | Meta | 8B | D | [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 8K |
| OLMo 2 7B | Ai2 | 7B | D | MHA + QK-Norm | RMSNorm, post | RoPE | 4K |
| DeepSeek V3 | DeepSeek | 671B/37B | [MoE](../mixture-of-experts/) | MLA | RMSNorm, pre | RoPE | 128K |

## 2025

| 模型 | 开发者 | 参数量 | 架构 | 注意力 | Norm | 位置编码 | 上下文 |
|------|--------|--------|------|--------|------|---------|--------|
| DeepSeek R1 | DeepSeek | 671B/37B | [MoE](../mixture-of-experts/) | MLA | RMSNorm, pre | RoPE | 128K |
| Gemma 3 27B | Google | 27B | D | [GQA](../attention_in_llm/) + QK-Norm + [5:1 SWA](../attention_in_llm/#sliding-window) | RMSNorm, pre+post | RoPE | 128K |
| Mistral Small 3.1 24B | Mistral | 24B | D | [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 128K |
| Llama 4 Maverick | Meta | 400B/17B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 256K |
| Qwen3 235B-A22B | Alibaba | 235B/22B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| Qwen3 32B | Alibaba | 32B | D | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| Qwen3 8B | Alibaba | 8B | D | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| Qwen3 4B | Alibaba | 4B | D | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| SmolLM3 3B | HuggingFace | 3B | D | [GQA](../attention_in_llm/) + 周期 NoPE 层 | RMSNorm, pre | RoPE + NoPE | 32K |
| Kimi K2 | Moonshot | 1T/32B | [MoE](../mixture-of-experts/) | MLA | RMSNorm, pre | RoPE | 128K |
| GLM-4.5 355B | Zhipu | 355B/32B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| GPT-OSS 120B | OpenAI | 120B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + 交替 [SWA](../attention_in_llm/#sliding-window)/全局 | RMSNorm, pre | RoPE | 256K |
| GPT-OSS 20B | OpenAI | 20B/3.6B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + 交替 [SWA](../attention_in_llm/#sliding-window)/全局 | RMSNorm, pre | RoPE | 256K |
| Grok 2.5 270B | xAI | 270B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 128K |
| Qwen3 Next 80B-A3B | Alibaba | 80B/3B | H | 3:1 DeltaNet + Gated Attention | RMSNorm, pre | RoPE | 128K |
| MiniMax M2 230B | MiniMax | 230B/10B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm + 部分 RoPE | RMSNorm, pre | 部分 RoPE | 128K |
| Kimi Linear 48B-A3B | Moonshot | 48B/3B | H | 3:1 Delta Attention + MLA | RMSNorm, pre | RoPE | 128K |
| OLMo 3 32B | Ai2 | 32B | D | [GQA](../attention_in_llm/) + QK-Norm + [3:1 SWA](../attention_in_llm/#sliding-window) | RMSNorm, post | RoPE | 64K |
| OLMo 3 7B | Ai2 | 7B | D | MHA + QK-Norm + [3:1 SWA](../attention_in_llm/#sliding-window) | RMSNorm, post | RoPE | 64K |
| DeepSeek V3.2 | DeepSeek | 671B/37B | [MoE](../mixture-of-experts/) | MLA + [DSA](../attention_in_llm/#dsa) | RMSNorm, pre | RoPE | 128K |
| Mistral 3 Large | Mistral | 673B/41B | [MoE](../mixture-of-experts/) | MLA | RMSNorm, pre | RoPE | 256K |
| Nemotron 3 Nano 30B-A3B | NVIDIA | 30B/3B | H | 主 [Mamba-2](../mamba/) + 少量 [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 128K |
| Xiaomi MiMo-V2 309B | Xiaomi | 309B/15B | [MoE](../mixture-of-experts/) | [5:1 SWA](../attention_in_llm/#sliding-window)/全局 | RMSNorm, pre | RoPE | 128K |
| GLM-4.7 355B | Zhipu | 355B/32B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |

## 2026

| 模型 | 开发者 | 参数量 | 架构 | 注意力 | Norm | 位置编码 | 上下文 |
|------|--------|--------|------|--------|------|---------|--------|
| Step 3.5 Flash 196B | StepFun | 196B/11B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + [3:1 SWA](../attention_in_llm/#sliding-window) | RMSNorm, pre | RoPE | 128K |
| Nanbeige 4.1 3B | Nanbeige | 3B | D | [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 128K |
| GLM-5 744B | Zhipu | 744B/40B | [MoE](../mixture-of-experts/) | MLA + [DSA](../attention_in_llm/#dsa) | RMSNorm, pre | RoPE | 128K |
| MiniMax M2.5 230B | MiniMax | 230B/10B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| Tiny Aya 3.35B | Cohere | 3.35B | D | [GQA](../attention_in_llm/) + [3:1 SWA](../attention_in_llm/#sliding-window) | RMSNorm, pre | RoPE + NoPE | 128K |
| Ling 2.5 1T | Ling | 1T/63B | H | 7:1 Lightning Attention + MLA | RMSNorm, pre | RoPE | 128K |
| Qwen3.5 397B | Alibaba | 397B/17B | H | 3:1 DeltaNet + Gated Attention | RMSNorm, pre | RoPE | 128K |
| Arcee Trinity 400B | Arcee | 400B/13B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + Gated Attention + [3:1 SWA](../attention_in_llm/#sliding-window) | RMSNorm, pre+post | RoPE + NoPE | 128K |
| Sarvam 105B | Sarvam | 105B | [MoE](../mixture-of-experts/) | MLA + KV LayerNorm | RMSNorm, pre | NoPE + RoPE | 128K |
| Sarvam 30B | Sarvam | 30B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| Nemotron 3 Super 120B-A12B | NVIDIA | 120B/12B | H | 主 [Mamba-2](../mamba/) + 少量 [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 128K |

## 架构选型趋势

**收敛项**（绝大多数模型采用）：

- **GQA 成为默认**：除 OLMo 7B 保留 MHA 外，所有 Attention 模型使用 GQA 或其变体 → [注意力机制详解](../attention_in_llm/#sparse-attention)
- **RoPE 位置编码**：几乎统一，部分模型混合 NoPE 层
- **RMSNorm + SwiGLU**：归一化和激活的事实标准 → [归一化](../norm/)
- **QK-Norm**：2025 年起大量采用 → [归一化](../norm/)

**分化项**（技术方案多元）：

| 方向 | 代表模型 | 详情 |
|------|---------|------|
| MLA (多头潜在注意力) | DeepSeek V3/R1/V3.2, Kimi K2, Mistral 3 Large, GLM-5 | KV 缓存极致压缩 → [DSA 章节](../attention_in_llm/#dsa) |
| Sliding Window + Global | Gemma 3, GPT-OSS, OLMo 3, Step 3.5 | 局部+全局混合 → [滑动窗口](../attention_in_llm/#sliding-window) |
| 动态稀疏选择 | DeepSeek V3.2 (DSA) | 内容感知 Top-k → [DSA](../attention_in_llm/#dsa) |
| 线性/Delta 注意力 | Qwen3 Next/3.5, Kimi Linear, Ling 2.5 | 线性复杂度替代方案 |
| Mamba-2 / SSM | Nemotron 3 Nano/Super | 状态空间替代注意力 → [Mamba](../mamba/) |
| Hybrid Attention+SSM | Nemotron 3, Qwen3.5 | 混合范式 → [Mamba](../mamba/) · [注意力](../attention_in_llm/) |

> **核心判断**：2024–2026 年的主流选择是 **MoE + GQA/MLA + RoPE + RMSNorm + SwiGLU**，但注意力机制正在从"纯 Attention"向"Attention + 线性替代 + SSM"的混合方向分裂。

---

*数据来源：[LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) by Sebastian Raschka*
