---
title: "Cross-Attention Mechanism"
date: 2025-05-21T22:04:00+08:00
# weight: 1
# aliases: ["/first"]
categories: ["Neural Network Modules"]
tags: ["Attention"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false # show table of contents
draft: false
hidemeta: false
comments: false
description: ""
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
    image: "cross_attention.png" # image path/url
    alt: "Cross Attention Diagram" # alt text
    caption: "Illustration of Cross Attention Mechanism" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Cross-Attention

### Overview

Cross-attention is an attention mechanism that fuses two embedding sequences:

- The two sequences must have the **same embedding dimension**.
- They can come from different modalities (e.g., text, images, audio).
- One sequence provides the **queries (Q)** and determines the **output length**.
- The other sequence provides the **keys (K)** and **values (V)**.

### Cross-attention vs. self-attention

The key difference is the input:

- **Self-attention** uses a single sequence to produce Q/K/V.
- **Cross-attention** uses two sequences: one for Q and the other for K/V.

![cross attention](cross_attention.png)

### Cross-attention algorithm

- Given two sequences $S_1$ and $S_2$
- Compute $K$ and $V$ from $S_1$
- Compute $Q$ from $S_2$
- Compute the attention matrix from $Q$ and $K$
- Apply attention to $V$
- The output sequence length matches $S_2$

$$
\pmb{\text{softmax}}((W_Q S_2)(W_K S_1)^\mathrm{T})W_V S_1
$$

