---
title: "LLM Architecture Gallery"
date: 2026-04-28T11:30:03+08:00
series:
    main: "Large Language Model"
    subseries: "Architecture and Training"
categories: ["大语言模型"]
tags: ["architecture", "attention", "MoE"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "A comparison of open/open-weight LLM architectures, attention mechanisms, context windows, and publicly documented training parallelism."
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

This page is based on Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/). As of 2026-07-16, the upstream gallery lists 85 entries. Rows are sorted by release date, and model names link directly when upstream provides a technical report or configuration. The former unverified Norm and positional-encoding guesses have been removed in favor of fields explicitly maintained upstream.

> **Legend**: D = Dense, M = Sparse Mixture of Experts (MoE), H = Hybrid, and R = Recurrent. Parameters are total / active per token; `Undisclosed` means no specific training-parallelism information was found after checking model reports, release posts, and training recipes.
>
> **Training field**: `Original` is a configuration disclosed by the model publisher; `Recipe` is a public recipe for the same open architecture, not the original cluster; `Strategy` discloses a parallelism method but not its degrees.
>
> **Parallelism**: Tensor Parallelism (TP), Sequence Parallelism (SP), Expert Parallelism (EP), Pipeline Parallelism (PP), and Context Parallelism (CP). ZeRO-1 is stage 1 of the Zero Redundancy Optimizer.

Further reading: [attention mechanisms](../attention_in_llm/) (MHA, GQA, MQA, MLA, sliding windows, DSA/CSA/HCA) · [Transformer components](../transformer_in_LLM/) (RoPE, RMSNorm, SwiGLU) · [MoE](../mixture-of-experts/) · [Mamba / linear attention](../mamba/) · [distributed training parallelism](../parallelization/).

## Model comparison

<div style="overflow-x: auto;">
  <table style="font-size: 0.76rem;">
    <thead>
      <tr>
        <th>Release date</th>
        <th>Model (links to report or config)</th>
        <th>Parameters (total / active)</th>
        <th>Architecture</th>
        <th>Attention / sequence module</th>
        <th>Context</th>
        <th>Training parallelism</th>
      </tr>
    </thead>
    <tbody>
<tr>
  <td>2019-11-05</td>
  <td><a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">GPT-2 XL 1.5B</a></td>
  <td>1.5B</td>
  <td>D</td>
  <td>MHA with learned absolute positional embeddings</td>
  <td>1,024</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2024-04-18</td>
  <td><a href="https://arxiv.org/pdf/2407.21783">Llama 3 8B</a></td>
  <td>8B</td>
  <td>D</td>
  <td>GQA with RoPE</td>
  <td>8,192</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2024-09-25</td>
  <td>Llama 3.2 1B</td>
  <td>1B</td>
  <td>D</td>
  <td>GQA</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2024-09-25</td>
  <td>Llama 3.2 3B</td>
  <td>3B</td>
  <td>D</td>
  <td>GQA</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2024-11-25</td>
  <td><a href="https://arxiv.org/pdf/2501.00656">OLMo 2 7B</a></td>
  <td>7B</td>
  <td>D</td>
  <td>MHA with QK-Norm</td>
  <td>4,096</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2024-12-12</td>
  <td><a href="https://arxiv.org/pdf/2412.08905">Phi-4</a></td>
  <td>14B</td>
  <td>D</td>
  <td>GQA with RoPE</td>
  <td>16,384</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2024-12-26</td>
  <td><a href="https://arxiv.org/pdf/2412.19437">DeepSeek V3</a></td>
  <td>671B / 37B</td>
  <td>M</td>
  <td>MLA</td>
  <td>128,000</td>
  <td><a href="https://arxiv.org/abs/2412.19437">Original</a>: ZeRO-1 · PP16 · EP64 (8 nodes; DualPipe)</td>
</tr>
<tr>
  <td>2025-01-20</td>
  <td><a href="https://arxiv.org/pdf/2501.12948">DeepSeek R1</a></td>
  <td>671B / 37B</td>
  <td>M</td>
  <td>MLA</td>
  <td>128,000</td>
  <td><a href="https://arxiv.org/abs/2501.12948">Strategy</a>: based on V3; R1 post-training degrees undisclosed</td>
</tr>
<tr>
  <td>2025-03-11</td>
  <td><a href="https://arxiv.org/pdf/2503.19786">Gemma 3 27B</a></td>
  <td>27B</td>
  <td>D</td>
  <td>GQA with QK-Norm and 5:1 sliding-window/global attention</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-03-17</td>
  <td><a href="https://arxiv.org/abs/2503.13427">xLSTM 7B</a></td>
  <td>7B</td>
  <td>R</td>
  <td>No self-attention; mLSTM recurrent layers with matrix memory</td>
  <td>No explicit limit</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-03-18</td>
  <td><a href="https://mistral.ai/news/mistral-small-3-1">Mistral Small 3.1 24B</a></td>
  <td>24B</td>
  <td>D</td>
  <td>Standard GQA</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-04-05</td>
  <td><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">Llama 4 Maverick</a></td>
  <td>400B / 17B</td>
  <td>M</td>
  <td>GQA</td>
  <td>1,000,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-04-28</td>
  <td><a href="https://arxiv.org/pdf/2505.09388">Qwen3 8B</a></td>
  <td>8B</td>
  <td>D</td>
  <td>GQA with QK-Norm</td>
  <td>128,000</td>
  <td><a href="https://www.usenix.org/conference/osdi26/presentation/hu-weifang">Strategy</a>: Qwen3 production pretraining uses Tessera PP/MoE communication overlap; degrees for this size undisclosed</td>
</tr>
<tr>
  <td>2025-04-28</td>
  <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json">Qwen3 0.6B</a></td>
  <td>0.6B</td>
  <td>D</td>
  <td>GQA</td>
  <td>32,768</td>
  <td><a href="https://www.usenix.org/conference/osdi26/presentation/hu-weifang">Strategy</a>: Qwen3 production pretraining uses Tessera PP/MoE communication overlap; degrees for this size undisclosed</td>
</tr>
<tr>
  <td>2025-04-28</td>
  <td><a href="https://arxiv.org/pdf/2505.09388">Qwen3 235B-A22B</a></td>
  <td>235B / 22B</td>
  <td>M</td>
  <td>GQA with QK-Norm</td>
  <td>128,000</td>
  <td><a href="https://www.usenix.org/conference/osdi26/presentation/hu-weifang">Strategy</a>: Qwen3 production pretraining uses Tessera PP/MoE communication overlap; degrees for this size undisclosed</td>
</tr>
<tr>
  <td>2025-04-28</td>
  <td><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json">Qwen3 30B-A3B</a></td>
  <td>30B / 3B</td>
  <td>M</td>
  <td>GQA</td>
  <td>128,000</td>
  <td><a href="https://www.usenix.org/conference/osdi26/presentation/hu-weifang">Strategy</a>: Qwen3 production pretraining uses Tessera PP/MoE communication overlap; degrees for this size undisclosed</td>
</tr>
<tr>
  <td>2025-04-28</td>
  <td><a href="https://arxiv.org/pdf/2505.09388">Qwen3 32B</a></td>
  <td>32B</td>
  <td>D</td>
  <td>GQA with QK-Norm</td>
  <td>128,000</td>
  <td><a href="https://www.usenix.org/conference/osdi26/presentation/hu-weifang">Strategy</a>: Qwen3 production pretraining uses Tessera PP/MoE communication overlap; degrees for this size undisclosed</td>
</tr>
<tr>
  <td>2025-04-28</td>
  <td><a href="https://arxiv.org/pdf/2505.09388">Qwen3 4B</a></td>
  <td>4B</td>
  <td>D</td>
  <td>GQA with QK-Norm</td>
  <td>32,768</td>
  <td><a href="https://www.usenix.org/conference/osdi26/presentation/hu-weifang">Strategy</a>: Qwen3 production pretraining uses Tessera PP/MoE communication overlap; degrees for this size undisclosed</td>
</tr>
<tr>
  <td>2025-06-19</td>
  <td><a href="https://huggingface.co/blog/smollm3">SmolLM3 3B</a></td>
  <td>3B</td>
  <td>D</td>
  <td>GQA with periodic NoPE layers</td>
  <td>131,072</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-07-10</td>
  <td><a href="https://arxiv.org/pdf/2507.20534">Kimi K2</a></td>
  <td>1T / 32B</td>
  <td>M</td>
  <td>MLA</td>
  <td>128,000</td>
  <td><a href="https://arxiv.org/abs/2507.20534">Original</a>: ZeRO-1 · PP16 (VPP) · EP16; EP communication overlaps with 1F1B</td>
</tr>
<tr>
  <td>2025-07-28</td>
  <td><a href="https://arxiv.org/pdf/2508.06471">GLM-4.5-Air</a></td>
  <td>106B / 12B</td>
  <td>M</td>
  <td>GQA</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-07-28</td>
  <td><a href="https://arxiv.org/pdf/2508.06471">GLM-4.5 355B</a></td>
  <td>355B / 32B</td>
  <td>M</td>
  <td>GQA with QK-Norm</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-07-31</td>
  <td><a href="https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/config.json">Qwen3 Coder Flash 30B-A3B</a></td>
  <td>30B / 3.3B</td>
  <td>M</td>
  <td>GQA</td>
  <td>256,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-08-04</td>
  <td><a href="https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf">GPT-OSS 120B</a></td>
  <td>117B / 5.1B</td>
  <td>M</td>
  <td>GQA with alternating sliding-window and global layers</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-08-04</td>
  <td><a href="https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf">GPT-OSS 20B</a></td>
  <td>21B / 3.6B</td>
  <td>M</td>
  <td>GQA with alternating sliding-window and global layers</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-08-14</td>
  <td><a href="https://arxiv.org/pdf/2503.19786">Gemma 3 270M</a></td>
  <td>270M</td>
  <td>D</td>
  <td>Multi-query attention with QK-Norm and 5:1 sliding-window/global attention</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-08-22</td>
  <td><a href="https://huggingface.co/xai-org/grok-2/blob/main/config.json">Grok 2.5 270B</a></td>
  <td>270B</td>
  <td>M</td>
  <td>GQA</td>
  <td>131,072</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-09-09</td>
  <td><a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json">Qwen3 Next 80B-A3B</a></td>
  <td>80B / 3B</td>
  <td>H+M</td>
  <td>3:1 Gated DeltaNet and Gated Attention</td>
  <td>262,144</td>
  <td><a href="https://www.usenix.org/conference/osdi26/presentation/hu-weifang">Strategy</a>: Qwen3-Next production pretraining uses Tessera PP/MoE communication overlap; degrees undisclosed</td>
</tr>
<tr>
  <td>2025-10-23</td>
  <td><a href="https://arxiv.org/abs/2605.26494">MiniMax M2 230B</a></td>
  <td>230B / 10B</td>
  <td>M</td>
  <td>GQA with QK-Norm and partial RoPE</td>
  <td>196,608</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-10-30</td>
  <td><a href="https://arxiv.org/pdf/2510.26692">Kimi Linear 48B-A3B</a></td>
  <td>48B / 3B</td>
  <td>H+M</td>
  <td>3:1 Kimi Delta Attention and MLA</td>
  <td>1,000,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-11-20</td>
  <td><a href="https://arxiv.org/pdf/2512.13961">OLMo 3 32B</a></td>
  <td>32B</td>
  <td>D</td>
  <td>GQA with QK-Norm and 3:1 sliding-window/global attention</td>
  <td>65,536</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-11-20</td>
  <td><a href="https://arxiv.org/pdf/2512.13961">OLMo 3 7B</a></td>
  <td>7B</td>
  <td>D</td>
  <td>MHA with QK-Norm and 3:1 sliding-window/global attention</td>
  <td>65,536</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-11-26</td>
  <td><a href="https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf">INTELLECT-3</a></td>
  <td>106B / 12B</td>
  <td>M</td>
  <td>GQA</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-12-01</td>
  <td><a href="https://arxiv.org/pdf/2512.02556">DeepSeek V3.2</a></td>
  <td>671B / 37B</td>
  <td>M</td>
  <td>MLA with DeepSeek Sparse Attention</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-12-02</td>
  <td><a href="https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512/blob/main/params.json">Mistral Large 3</a></td>
  <td>673B / 41B</td>
  <td>M</td>
  <td>MLA</td>
  <td>262,144</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-12-04</td>
  <td><a href="https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf">Nemotron 3 Nano 30B-A3B</a></td>
  <td>30B / 3B</td>
  <td>H+M</td>
  <td>Mostly Mamba-2 with a few GQA layers</td>
  <td>1,000,000</td>
  <td><a href="https://docs.nvidia.com/nemotron/latest/nemotron/nano3/pretrain.html">Original</a>: main TP8+SP · EP8 · PP1 · CP1; long-context TP8+SP · EP8 · PP4 · CP8</td>
</tr>
<tr>
  <td>2025-12-16</td>
  <td><a href="https://arxiv.org/pdf/2601.02780">Xiaomi MiMo-V2-Flash 309B</a></td>
  <td>309B / 15B</td>
  <td>M</td>
  <td>5:1 sliding-window/global attention</td>
  <td>262,144</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2025-12-22</td>
  <td><a href="https://arxiv.org/pdf/2508.06471">GLM-4.7 355B</a></td>
  <td>355B / 32B</td>
  <td>M</td>
  <td>GQA with QK-Norm</td>
  <td>202,752</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-01-27</td>
  <td><a href="https://arxiv.org/pdf/2602.02276">Kimi K2.5</a></td>
  <td>1T / 32B</td>
  <td>M</td>
  <td>MLA</td>
  <td>256,000</td>
  <td><a href="https://www.kimi.com/blog/kimi-k2-5">Strategy</a>: continued pretraining from K2; degrees for this phase undisclosed</td>
</tr>
<tr>
  <td>2026-01-27</td>
  <td><a href="https://arxiv.org/pdf/2602.17004">Arcee AI Trinity Large 400B</a></td>
  <td>400B / 13B</td>
  <td>M</td>
  <td>GQA with gated attention and 3:1 sliding-window/global attention</td>
  <td>512,000</td>
  <td><a href="https://arxiv.org/abs/2602.17004">Original</a>: HSDP/FSDP + EP; degrees undisclosed</td>
</tr>
<tr>
  <td>2026-01-28</td>
  <td><a href="https://arxiv.org/abs/2601.21204">LongCat-Flash-Lite 68.5B-A3B</a></td>
  <td>68.5B / ~3B</td>
  <td>M</td>
  <td>MLA with RoPE + NoPE</td>
  <td>256,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-02-01</td>
  <td><a href="https://arxiv.org/pdf/2602.10604">Step 3.5 Flash 196B</a></td>
  <td>196B / 11B</td>
  <td>M</td>
  <td>GQA with 3:1 sliding-window attention</td>
  <td>262,144</td>
  <td><a href="https://arxiv.org/abs/2602.10604">Original</a>: ZeRO-1 · PP8 (VPP) · EP8; decoupled attention/MoE parallelism</td>
</tr>
<tr>
  <td>2026-02-10</td>
  <td><a href="https://arxiv.org/pdf/2602.13367">Nanbeige 4.1 3B</a></td>
  <td>3B</td>
  <td>D</td>
  <td>GQA</td>
  <td>262,144</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-02-11</td>
  <td><a href="https://arxiv.org/pdf/2602.15763">GLM-5 744B</a></td>
  <td>744B / 40B</td>
  <td>M</td>
  <td>MLA with DeepSeek Sparse Attention</td>
  <td>202,752</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-02-12</td>
  <td><a href="https://arxiv.org/abs/2605.26494">MiniMax M2.5 230B</a></td>
  <td>230B / 10B</td>
  <td>M</td>
  <td>GQA with QK-Norm</td>
  <td>196,608</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-02-13</td>
  <td><a href="https://arxiv.org/pdf/2603.11510">Tiny Aya 3.35B</a></td>
  <td>3.35B</td>
  <td>D</td>
  <td>GQA with 3:1 sliding-window attention</td>
  <td>8,192</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-02-15</td>
  <td><a href="https://huggingface.co/inclusionAI/Ling-2.5-1T/blob/main/config.json">Ling 2.5 1T</a></td>
  <td>1T / 63B</td>
  <td>H+M</td>
  <td>Lightning Attention plus MLA</td>
  <td>256,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-02-16</td>
  <td><a href="https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json">Qwen3.5 397B</a></td>
  <td>397B / 17B</td>
  <td>H+M</td>
  <td>3:1 Gated DeltaNet and Gated Attention</td>
  <td>262,144</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-03-03</td>
  <td><a href="https://www.sarvam.ai/blogs/sarvam-30b-105b">Sarvam 30B</a></td>
  <td>30B / 2.4B</td>
  <td>M</td>
  <td>GQA with QK-Norm</td>
  <td>131,072</td>
  <td><a href="https://www.nvidia.com/en-us/case-studies/sarvam-sovereign-ai/">Strategy</a>: Megatron-LM 6D parallelism on 4,096+ H100s; degrees undisclosed</td>
</tr>
<tr>
  <td>2026-03-03</td>
  <td><a href="https://www.sarvam.ai/blogs/sarvam-30b-105b">Sarvam 105B</a></td>
  <td>105B / 10.3B</td>
  <td>M</td>
  <td>MLA with KV LayerNorm and NoPE + RoPE</td>
  <td>131,072</td>
  <td><a href="https://www.nvidia.com/en-us/case-studies/sarvam-sovereign-ai/">Strategy</a>: Megatron-LM 6D parallelism on 4,096+ H100s; degrees undisclosed</td>
</tr>
<tr>
  <td>2026-03-11</td>
  <td><a href="https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf">Nemotron 3 Super 120B-A12B</a></td>
  <td>120B / 12B</td>
  <td>H+M</td>
  <td>Mostly Mamba-2 with a few GQA layers</td>
  <td>1,000,000</td>
  <td><a href="https://docs.nvidia.com/nemotron/latest/nemotron/super3/pretrain.html">Original</a>: main TP4+SP · EP8 · PP1 · CP1; long-context TP2+SP · EP64 · CP64</td>
</tr>
<tr>
  <td>2026-03-16</td>
  <td><a href="https://mistral.ai/news/mistral-small-4">Mistral Small 4</a></td>
  <td>119B / 6.63B</td>
  <td>M</td>
  <td>MLA</td>
  <td>256,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-03-16</td>
  <td><a href="https://huggingface.co/blog/nvidia/nemotron-3-nano-4b">Nemotron 3 Nano 4B</a></td>
  <td>4B</td>
  <td>H</td>
  <td>GQA with only 4 attention layers</td>
  <td>262,144</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-03-18</td>
  <td><a href="https://arxiv.org/abs/2605.26494">MiniMax M2.7 230B</a></td>
  <td>230B / 10B</td>
  <td>M</td>
  <td>GQA with QK-Norm</td>
  <td>196,608</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-02</td>
  <td><a href="https://ai.google.dev/gemma/docs/core/model_card_4">Gemma 4 E2B</a></td>
  <td>5.1B (2.3B effective)</td>
  <td>D</td>
  <td>Multi-query attention with QK-Norm, unified K/V on global layers, p-RoPE on global layers, and 4:1 sliding-window/global attention</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-02</td>
  <td><a href="https://ai.google.dev/gemma/docs/core/model_card_4">Gemma 4 26B-A4B</a></td>
  <td>25.2B / 3.8B</td>
  <td>M</td>
  <td>GQA with QK-Norm, unified K/V on global layers, p-RoPE on global layers, and 5:1 sliding-window/global attention</td>
  <td>256,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-02</td>
  <td><a href="https://ai.google.dev/gemma/docs/core/model_card_4">Gemma 4 31B</a></td>
  <td>30.7B</td>
  <td>D</td>
  <td>GQA with QK-Norm, unified K/V on global layers, p-RoPE on global layers, and 5:1 sliding-window/global attention</td>
  <td>256,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-02</td>
  <td><a href="https://ai.google.dev/gemma/docs/core/model_card_4">Gemma 4 E4B</a></td>
  <td>8B (4.5B effective)</td>
  <td>D</td>
  <td>GQA with QK-Norm, unified K/V on global layers, p-RoPE on global layers, and 5:1 sliding-window/global attention</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-07</td>
  <td><a href="https://arxiv.org/pdf/2602.15763">GLM-5.1</a></td>
  <td>744B / 40B</td>
  <td>M</td>
  <td>MLA with DeepSeek Sparse Attention</td>
  <td>202,752</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-15</td>
  <td><a href="https://qwen.ai/blog?id=qwen3.6-35b-a3b">Qwen3.6 35B-A3B</a></td>
  <td>35B / 3B</td>
  <td>H+M</td>
  <td>3:1 Gated DeltaNet and Gated Attention</td>
  <td>262,144</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-20</td>
  <td><a href="https://www.kimi.com/blog/kimi-k2-6.html">Kimi K2.6</a></td>
  <td>1T / 32B</td>
  <td>M</td>
  <td>MLA</td>
  <td>256,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-22</td>
  <td><a href="https://mimo.xiaomi.com/mimo-v2-5">Xiaomi MiMo-V2.5 310B</a></td>
  <td>310B / 15B</td>
  <td>M</td>
  <td>5:1 sliding-window/global attention</td>
  <td>1,048,576</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-22</td>
  <td><a href="https://qwen.ai/blog?id=qwen3.6-27b">Qwen3.6 27B</a></td>
  <td>27B</td>
  <td>H</td>
  <td>3:1 Gated DeltaNet and Gated Attention</td>
  <td>262,144</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-22</td>
  <td><a href="https://mimo.xiaomi.com/mimo-v2-5">Xiaomi MiMo-V2.5-Pro 1.02T</a></td>
  <td>1.02T / 42B</td>
  <td>M</td>
  <td>GQA with 6:1 sliding-window/global attention</td>
  <td>1,048,576</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-23</td>
  <td><a href="https://www.tencent.com/en-us/articles/2202320.html">Tencent Hy3-preview 295B-A21B</a></td>
  <td>295B / 21B</td>
  <td>M</td>
  <td>GQA with QK-Norm</td>
  <td>262,144</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-23</td>
  <td><a href="https://huggingface.co/inclusionAI/Ling-2.6-1T">Ling 2.6 1T</a></td>
  <td>1T / 63B</td>
  <td>H+M</td>
  <td>Lightning Attention plus MLA</td>
  <td>262,144</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-24</td>
  <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf">DeepSeek V4-Pro</a></td>
  <td>1.6T / 49B</td>
  <td>M</td>
  <td>MLA-style CSA/HCA with mHC</td>
  <td>1,048,576</td>
  <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/DeepSeek_V4.pdf">Strategy</a>: CP for long context; EP communication overlaps with DualPipe 1F1B; degrees undisclosed</td>
</tr>
<tr>
  <td>2026-04-24</td>
  <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/DeepSeek_V4.pdf">DeepSeek V4-Flash</a></td>
  <td>284B / 13B</td>
  <td>M</td>
  <td>MLA-style CSA/HCA with mHC</td>
  <td>1,048,576</td>
  <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/DeepSeek_V4.pdf">Strategy</a>: CP for long context; EP communication overlaps with DualPipe 1F1B; degrees undisclosed</td>
</tr>
<tr>
  <td>2026-04-28</td>
  <td><a href="https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf">Laguna XS.2</a></td>
  <td>33B / 3B</td>
  <td>M</td>
  <td>Gated GQA with QK-Norm and 3:1 sliding-window/global attention</td>
  <td>131,072</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-04-29</td>
  <td><a href="https://huggingface.co/blog/ibm-granite/granite-4-1">Granite 4.1 30B</a></td>
  <td>30B</td>
  <td>D</td>
  <td>GQA with RoPE</td>
  <td>131,072</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-05-06</td>
  <td><a href="https://arxiv.org/abs/2605.05365">ZAYA1-8B</a></td>
  <td>8.4B / 760M</td>
  <td>M</td>
  <td>CCA with 4:1 GQA, RoPE, and Q/K L2 norm</td>
  <td>131,072</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-05-20</td>
  <td><a href="https://cohere.com/blog/command-a-plus">Command A+ 218B-A25B</a></td>
  <td>218B / 25B</td>
  <td>M</td>
  <td>16:1 GQA with 3:1 sliding-window/global attention</td>
  <td>128K input, 64K output</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-05-28</td>
  <td><a href="https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb">LFM2.5 1.2B</a></td>
  <td>1.2B</td>
  <td>H</td>
  <td>LIV convolution blocks plus GQA</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-05-28</td>
  <td><a href="https://www.liquid.ai/blog/lfm2-5-8b-a1b">LFM2.5 8B-A1B</a></td>
  <td>8.3B / 1.5B</td>
  <td>H+M</td>
  <td>LIV convolution blocks plus GQA and MoE</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-05-28</td>
  <td><a href="https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind">LFM2.5 350M</a></td>
  <td>350M</td>
  <td>H</td>
  <td>LIV convolution blocks plus GQA</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-06-01</td>
  <td><a href="https://arxiv.org/abs/2605.31268">JetBrains Mellum2 Thinking 12B-A2.5B</a></td>
  <td>12B / 2.5B</td>
  <td>M</td>
  <td>GQA with 3:1 sliding-window/full attention</td>
  <td>131,072</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-06-03</td>
  <td><a href="https://developers.googleblog.com/gemma-4-12b-the-developer-guide/">Gemma 4 12B</a></td>
  <td>12B</td>
  <td>D</td>
  <td>GQA with QK-Norm, unified K/V on global layers, p-RoPE on global layers, and 5:1 sliding-window/global attention</td>
  <td>128,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-06-04</td>
  <td><a href="https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf">Nemotron 3 Ultra 550B-A55B</a></td>
  <td>550B / 55B</td>
  <td>H+M</td>
  <td>Mostly Mamba-2 with a few GQA layers</td>
  <td>1,000,000</td>
  <td><a href="https://docs.nvidia.com/nemotron/nightly/nemotron/ultra3/pretrain.html">Original</a>: TP2 · PP12 · EP32 · CP1 (ETP1)</td>
</tr>
<tr>
  <td>2026-06-05</td>
  <td><a href="https://huggingface.co/blog/CohereLabs/introducing-north-mini-code">North Mini Code 30B-A3B</a></td>
  <td>30B / 3B</td>
  <td>M</td>
  <td>8:1 GQA with 3:1 sliding-window/global attention</td>
  <td>256K input, 64K output</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-06-12</td>
  <td><a href="https://huggingface.co/moonshotai/Kimi-K2.7-Code">Kimi K2.7 Code</a></td>
  <td>1T / 32B</td>
  <td>M</td>
  <td>MLA</td>
  <td>256,000</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-06-13</td>
  <td><a href="https://arxiv.org/abs/2606.13392">MiniMax M3 428B</a></td>
  <td>428B / 23B</td>
  <td>M</td>
  <td>GQA with QK-Norm and MiniMax Sparse Attention</td>
  <td>1,048,576</td>
  <td><a href="https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/minimax/minimax-m3.html">Recipe</a>: Megatron Bridge baseline TP2 · PP4 · EP32 (not the original cluster)</td>
</tr>
<tr>
  <td>2026-06-15</td>
  <td><a href="https://arxiv.org/abs/2606.16140">VibeThinker-3B</a></td>
  <td>3B</td>
  <td>D</td>
  <td>8:1 GQA with RoPE</td>
  <td>131,072</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-06-17</td>
  <td><a href="https://arxiv.org/pdf/2602.15763">GLM-5.2</a></td>
  <td>744B / 40B</td>
  <td>M</td>
  <td>MLA with DeepSeek Sparse Attention and IndexShare</td>
  <td>1,048,576</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-07-10</td>
  <td><a href="https://arxiv.org/abs/2607.09424">Soofi S 30B-A3B</a></td>
  <td>31.6B / 3.2B</td>
  <td>H+M</td>
  <td>Mostly Mamba-2 with 6 GQA layers and no positional embeddings (NoPE)</td>
  <td>1,048,576</td>
  <td>Undisclosed</td>
</tr>
<tr>
  <td>2026-07-15</td>
  <td><a href="https://thinkingmachines.ai/news/introducing-inkling/">Inkling</a></td>
  <td>975B / 41B</td>
  <td>M</td>
  <td>GQA with QK-Norm and learned relative-position bias in a 5:1 sliding-window/global pattern</td>
  <td>1,048,576</td>
  <td>Undisclosed</td>
</tr>
    </tbody>
  </table>
</div>


## Observations

- MoE remains the common path for parameter scaling. Long-context models also combine local/global, sparse, or hybrid sequence modules to reduce attention cost.
- KV-cache optimization has not converged on one method: GQA/MQA reduce KV heads, MLA uses low-rank compression, and DSA/CSA/HCA further sparsify or compress long-context computation.
- Mamba, DeltaNet, Lightning Attention, and convolution modules appear in a subset of models. They should be understood through the layer ratios in each report, not as a single replacement for GQA or MLA.

## References

[1] S. Raschka, "LLM Architecture Gallery." [Online]. Available: https://sebastianraschka.com/llm-architecture-gallery/.

[2] S. Raschka, "llm-architecture-gallery: source metadata." [Online]. Available: https://github.com/rasbt/llm-architecture-gallery/blob/main/models.yml.

[3] DeepSeek-AI, "DeepSeek-V3 Technical Report," 2024. [Online]. Available: https://arxiv.org/abs/2412.19437.

[4] Moonshot AI, "Kimi K2 Technical Report," 2025. [Online]. Available: https://arxiv.org/abs/2507.20534.

[5] W. Hu et al., "Tessera: Automated Pipeline Parallelism for Production-Scale LLM Pretraining," OSDI 2026. [Online]. Available: https://www.usenix.org/conference/osdi26/presentation/hu-weifang.

[6] NVIDIA, "Nemotron 3 pretraining recipes: Nano, Super, and Ultra." [Online]. Available: https://docs.nvidia.com/nemotron/latest/nemotron/nano3/pretrain.html; https://docs.nvidia.com/nemotron/latest/nemotron/super3/pretrain.html; https://docs.nvidia.com/nemotron/nightly/nemotron/ultra3/pretrain.html.

[7] NVIDIA, "MiniMax-M3 with Megatron Bridge." [Online]. Available: https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/minimax/minimax-m3.html.

[8] StepFun, "Step 3.5 Flash Technical Report," 2026. [Online]. Available: https://arxiv.org/abs/2602.10604.

[9] NVIDIA, "Sarvam: building sovereign AI in India." [Online]. Available: https://www.nvidia.com/en-us/case-studies/sarvam-sovereign-ai/.

[10] Arcee AI, "Trinity Large Technical Report," 2026. [Online]. Available: https://arxiv.org/abs/2602.17004.
