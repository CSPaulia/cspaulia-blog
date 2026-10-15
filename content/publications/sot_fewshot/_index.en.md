---
title: "Joint Modeling of Typicality and Uncertainty for SoT-Based Few-Shot LLM Reasoning"
date: 2026-05-06
venue: ICASSP 2026
authors:
    - Bole Wang
    - Ping Wei
    - Ziyang Ren
    - Shengze Li
    - Jieyuan Zhang
    - Shuaijia Chen
selected: true
paper: https://ieeexplore.ieee.org/document/11464177
summary: Joint modeling of typicality and uncertainty for Sketch-of-Thought based few-shot LLM reasoning.
cover:
    image: cover.png
---

Chain-of-Thought (CoT) and its variants have advanced large language model (LLM) reasoning. Traditional methods construct prompts for LLMs by randomly sampling question examples as demonstrations, which fail to guarantee effective prompting. To address these limitations, we propose a joint modeling framework that integrates typicality and uncertainty to determine demonstrations for effective LLM prompting. We utilize an aggregator to calculate the typicality scores of each question and compute the uncertainty scores from the answer distributions produced by the LLM through multi-turn question answering. We design three strategies to integrate typicality and uncertainty: an entropy-weighted method for adaptive score balancing and two categorization-first approaches that enhance prompt relevance and reasoning performance. Based on the integrated result, the valuable questions are utilized to construct effective prompts in the SoT paradigm. Experiments on multiple datasets demonstrate that our method successfully improves answer accuracy while reducing token costs.
