---
title: "GPT Series"
date: 2025-06-18T20:00:00+08:00
series:
    main: "Large Language Model"
    subseries: "Mainstream Series"
categories: ["Large Language Model"]
tags: ["GPT", "Pre-training", "LLM"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "A deep dive into the GPT family of pretrained language models."
UseHugoToc: true
cover:
    image: "gpt-cover.png" # image path/url
    alt: "GPT Architecture" # alt text
    caption: "GPT Architecture" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
    hiddenInList: false # hide on list pages and home
    # class: "post-cover"
---

## GPT-1

### Motivation

It was unclear which training objective could best learn text representations that transfer well across tasks. (Intuitively: for different NLP tasks, we don't know which objective works best.)

### Method

- A semi-supervised setup
- Transformer (a more structured memory for handling long-range dependencies)  strong transfer performance

#### 1. Unsupervised pre-training

- **Objective:**

    Given an unsupervised corpus of tokens $U = \\{u_1, \cdots , u_n\\}$,

    $$
    L_1(U) = \sum_i \log{P(u_i | u_{i-k}, \cdots, u_{i-1}; \Theta)},\quad i \in \\{1,\cdots, n\\}
    $$

    - $k$ is the context window size; a neural network with parameters $\Theta$ models the conditional probability.
    - $P(u_i | u_{i-k}, \cdots, u_{i-1}; \Theta)$ is the probability of token $u_i$ given the previous $k$ tokens.

    Since GPT is trained with an unsupervised language modeling objective, it learns to predict the next token given the context. Maximizing the summed log-likelihood above yields an objective that can be applied broadly.

- **Architecture:**

    A multi-layer Transformer decoder

    $$
    \begin{aligned}
    h_0 &= UW_e + W_p \\\\
    h_i &= \text{transformer block}(h_{i-1}) \\\\
    P(u) &= \text{softmax}(h_n W_e^T)
    \end{aligned}
    $$

    - $W_e$ is the token embedding matrix
    - $W_p$ is the position embedding matrix

#### 2. Supervised fine-tuning

Assume a labeled dataset that contains:

- an input token sequence $x^1, \cdots, x^m$
- a label $y$

Let $h_l^m$ be the activation from the last Transformer block. Then:

$$
P(y|x^1, \dots, x^m) = \text{softmax}(h_l^m W_y)
$$

$$
L_2(\hat{C}) = \sum_{(x, y)} \log P(y|x^1, \dots, x^m)
$$

Similar to $L_1$, but now maximizing the label likelihood. The final objective is:

$$
L_3(\hat{C}) = L_2(\hat{C}) + \lambda * L_1(\hat{C})
$$

So the fine-tuning objective combines $L_1$ and $L_2$.

#### 3. Task-specific input transformations

<p align="center">
  {{< img src="gpt-tasks.png" alt="gpt-tasks" >}}
</p>

For similarity tasks, because GPT is a unidirectional model (generating tokens left-to-right), the order of Text 1 and Text 2 matters. You can feed different orderings and average the resulting similarity scores.

## GPT-2

### Motivation

A common way to build ML systems is to collect a dataset for training. However, training on a single domain-specific dataset often harms generalization.

### Method

- **Multitask learning:** language models can perform downstream tasks in a zero-shot setting without changing parameters or architecture.
- Pre-training + supervised fine-tuning
- Model the objective as $p(\text{output}|\text{input}, \text{task})$, written as a sequence: {task (as a prompt), input, output}

    - Example 1 (translation): (translate to French, English text, French text)
    - Example 2 (reading comprehension): (answer the question, document, question, answer)

#### Training data

- Collected Reddit posts with at least 3 karma: 45M links  8M documents, ~40GB of text

#### Model

Roughly similar to GPT.

## GPT-3

### Motivation

Even when tasks are unknown in advance, most language models still require task-specific datasets and task-specific fine-tuning.

- Need large labeled datasets tailored to each task
- Strong performance on a fine-tuning set does not necessarily mean strong generalization

### Method

- **Meta-learning:** train a model that generalizes well
- **In-context learning:** given a few training examples at inference time, do not update the model weights (intuitively: include demonstrations in the prompt)

    - zero-shot
    - one-shot
    - few-shot

<p align="center">
  {{< img src="gpt-3-tasks.png" alt="gpt-3-tasks" >}}
</p>

#### Model & architecture

- Same model and architecture as GPT-2
- Sparse Transformer
- 8 different sizes

<p align="center">
  {{< img src="gpt-3-models.png" alt="gpt-3-models" >}}
</p>

---

<div class="zhihu-ref">
  <div class="zhihu-ref-title">References</div>
  <ol>
    <li><a href="https://www.bilibili.com/video/BV1AF411b7xQ?spm_id_from=333.788.videopod.sections&vd_source=9e4f1724ef60547fa31e3c8270245ff8" target="_blank">GPT, GPT-2, GPT-3 paper walkthrough (Bilibili)</a></li>
    <li><a href="https://www.mikecaptain.com/resources/pdf/GPT-1.pdf" target="_blank">Improving language understanding by generative pre-training</a></li>
    <li><a href="https://storage.prod.researchhub.com/uploads/papers/2020/06/01/language-models.pdf" target="_blank">Language models are unsupervised multitask learners</a></li>
    <li><a href="https://www.mikecaptain.com/resources/pdf/GPT-3.pdf" target="_blank">Language Models are Few-Shot Learners</a></li>
  </ol>
</div>