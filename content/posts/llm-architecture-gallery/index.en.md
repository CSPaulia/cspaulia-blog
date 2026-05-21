---
title: "LLM Architecture Landscape (2024–2026)"
date: 2026-04-28T11:30:03+08:00
series:
    main: "Large Language Model"
    subseries: "Architecture and Training"
categories: ["大语言模型"]
tags: ["架构", "注意力", "MoE"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "An overview of mainstream open-source LLM architectures from 2024–2026, including comparisons of attention mechanisms, MoE, SSM, and other dimensions."
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

Compiled from [Sebastian Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), covering 38 open-source/open-weight models released from 2024 to 2026. The Attention and Architecture columns link to detailed technical articles on this site.

> **Legend**: D = Dense, M = MoE (Sparse Mixture of Experts), H = Hybrid. Parameter count is denoted as total parameters / activated parameters (for MoE models).
> The Norm column indicates type and placement (pre = before sublayer, post = after sublayer).

## 2024

| Model | Developer | Parameters | Architecture | Attention | Norm | Position Encoding | Context |
|-------|-----------|------------|-------------|-----------|------|-------------------|---------|
| Llama 3 8B | Meta | 8B | D | [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 8K |
| OLMo 2 7B | Ai2 | 7B | D | MHA + QK-Norm | RMSNorm, post | RoPE | 4K |
| DeepSeek V3 | DeepSeek | 671B/37B | [MoE](../mixture-of-experts/) | MLA | RMSNorm, pre | RoPE | 128K |

## 2025

| Model | Developer | Parameters | Architecture | Attention | Norm | Position Encoding | Context |
|-------|-----------|------------|-------------|-----------|------|-------------------|---------|
| DeepSeek R1 | DeepSeek | 671B/37B | [MoE](../mixture-of-experts/) | MLA | RMSNorm, pre | RoPE | 128K |
| Gemma 3 27B | Google | 27B | D | [GQA](../attention_in_llm/) + QK-Norm + [5:1 SWA](../attention_in_llm/#sliding-window) | RMSNorm, pre+post | RoPE | 128K |
| Mistral Small 3.1 24B | Mistral | 24B | D | [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 128K |
| Llama 4 Maverick | Meta | 400B/17B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 256K |
| Qwen3 235B-A22B | Alibaba | 235B/22B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| Qwen3 32B | Alibaba | 32B | D | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| Qwen3 8B | Alibaba | 8B | D | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| Qwen3 4B | Alibaba | 4B | D | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| SmolLM3 3B | HuggingFace | 3B | D | [GQA](../attention_in_llm/) + periodic NoPE layers | RMSNorm, pre | RoPE + NoPE | 32K |
| Kimi K2 | Moonshot | 1T/32B | [MoE](../mixture-of-experts/) | MLA | RMSNorm, pre | RoPE | 128K |
| GLM-4.5 355B | Zhipu | 355B/32B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |
| GPT-OSS 120B | OpenAI | 120B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + alternating [SWA](../attention_in_llm/#sliding-window)/global | RMSNorm, pre | RoPE | 256K |
| GPT-OSS 20B | OpenAI | 20B/3.6B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + alternating [SWA](../attention_in_llm/#sliding-window)/global | RMSNorm, pre | RoPE | 256K |
| Grok 2.5 270B | xAI | 270B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 128K |
| Qwen3 Next 80B-A3B | Alibaba | 80B/3B | H | 3:1 DeltaNet + Gated Attention | RMSNorm, pre | RoPE | 128K |
| MiniMax M2 230B | MiniMax | 230B/10B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm + partial RoPE | RMSNorm, pre | Partial RoPE | 128K |
| Kimi Linear 48B-A3B | Moonshot | 48B/3B | H | 3:1 Delta Attention + MLA | RMSNorm, pre | RoPE | 128K |
| OLMo 3 32B | Ai2 | 32B | D | [GQA](../attention_in_llm/) + QK-Norm + [3:1 SWA](../attention_in_llm/#sliding-window) | RMSNorm, post | RoPE | 64K |
| OLMo 3 7B | Ai2 | 7B | D | MHA + QK-Norm + [3:1 SWA](../attention_in_llm/#sliding-window) | RMSNorm, post | RoPE | 64K |
| DeepSeek V3.2 | DeepSeek | 671B/37B | [MoE](../mixture-of-experts/) | MLA + [DSA](../attention_in_llm/#dsa) | RMSNorm, pre | RoPE | 128K |
| Mistral 3 Large | Mistral | 673B/41B | [MoE](../mixture-of-experts/) | MLA | RMSNorm, pre | RoPE | 256K |
| Nemotron 3 Nano 30B-A3B | NVIDIA | 30B/3B | H | Primarily [Mamba-2](../mamba/) + few [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 128K |
| Xiaomi MiMo-V2 309B | Xiaomi | 309B/15B | [MoE](../mixture-of-experts/) | [5:1 SWA](../attention_in_llm/#sliding-window)/global | RMSNorm, pre | RoPE | 128K |
| GLM-4.7 355B | Zhipu | 355B/32B | [MoE](../mixture-of-experts/) | [GQA](../attention_in_llm/) + QK-Norm | RMSNorm, pre | RoPE | 128K |

## 2026

| Model | Developer | Parameters | Architecture | Attention | Norm | Position Encoding | Context |
|-------|-----------|------------|-------------|-----------|------|-------------------|---------|
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
| Nemotron 3 Super 120B-A12B | NVIDIA | 120B/12B | H | Primarily [Mamba-2](../mamba/) + few [GQA](../attention_in_llm/) | RMSNorm, pre | RoPE | 128K |
| DeepSeek V4 Flash | DeepSeek | 284B/13B | [MoE](../mixture-of-experts/) | [CSA](../attention_in_llm/index.md#csa) + [HCA](../attention_in_llm/index.md#hca) | RMSNorm, pre | RoPE | 128K |
| DeepSeek V4 Pro | DeepSeek | 1.6T/49B | [MoE](../mixture-of-experts/) | MLA + [DSA](../attention_in_llm/index.md#dsa) + [HCA](../attention_in_llm/index.md#hca) | RMSNorm, pre | RoPE | 128K |

## Architecture Selection Trends

**Convergence** (adopted by the vast majority of models):

- **GQA becomes default**: Except for OLMo 7B which retains MHA, all Attention models use GQA or its variants → [Attention Mechanism Details](../attention_in_llm/#sparse-attention)
- **RoPE position encoding**: Nearly unified, some models mix NoPE layers
- **RMSNorm + SwiGLU**: De facto standard for normalization and activation → [Normalization](../norm/)
- **QK-Norm**: Widely adopted from 2025 onwards → [Normalization](../norm/)

**Divergence** (diverse technical approaches):

| Direction | Representative Models | Details |
|-----------|----------------------|---------|
| MLA (Multi-head Latent Attention) | DeepSeek V3/R1/V3.2, Kimi K2, Mistral 3 Large, GLM-5 | Extreme KV cache compression → [DSA Section](../attention_in_llm/#dsa) |
| Sliding Window + Global | Gemma 3, GPT-OSS, OLMo 3, Step 3.5 | Local + Global mixture → [Sliding Window](../attention_in_llm/#sliding-window) |
| Dynamic Sparse Selection | DeepSeek V3.2 (DSA) | Content-aware Top-k → [DSA](../attention_in_llm/#dsa) |
| Linear/Delta Attention | Qwen3 Next/3.5, Kimi Linear, Ling 2.5 | Linear complexity alternatives |
| Mamba-2 / SSM | Nemotron 3 Nano/Super | State space replaces attention → [Mamba](../mamba/) |
| Hybrid Attention+SSM | Nemotron 3, Qwen3.5 | Hybrid paradigm → [Mamba](../mamba/) · [Attention](../attention_in_llm/) |

> **Core Judgment**: The mainstream choice in 2024–2026 is **MoE + GQA/MLA + RoPE + RMSNorm + SwiGLU**, but attention mechanisms are diverging from "pure Attention" towards a hybrid of "Attention + linear alternatives + SSM".

---

*Data source: [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) by Sebastian Raschka*