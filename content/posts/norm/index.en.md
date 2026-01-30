---
title: "100 Normalization Methods (Work in Progress)"
date: 2025-05-21T21:15:00+08:00
# weight: 1
# aliases: ["/first"]
categories: ["Deep Learning Skills"]
tags: ["Normalization"]
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
    alt: "Normalization methods overview" # alt text
    caption: "Normalization methods" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

### Layer Normalization

<p align="center">
  {{< img src="LNvsBN.jpg" alt="LN vs BN" >}}
</p>

In the figure above, $N$ denotes the sample axis, $C$ the channel axis, and $F$ the number of features per channel.
Batch Normalization (BN, right) normalizes using features from **the same channel across different samples**.
Layer Normalization (LN, left) normalizes using features from **different channels within the same sample**.

#### 1. Issues with BN

##### 1.1 BN and batch size

BN computes normalization statistics based on the **number of samples** in a batch.
When the batch is very small (e.g., only 4 samples), the mean and variance estimated from those samples may not represent the global data distribution well, so BN can perform poorly.

##### 1.2 BN and RNNs

<p align="center">
  {{< img src="RNN.jpg" alt="RNN" >}}
</p>

Within a batch, sequence lengths often differ.
For later time steps (e.g., $t>4$ in the figure), only a small number of sequences may still have valid tokens.
Statistics computed from so few samples are not representative of the overall distribution, so BN tends to work poorly in this setting.

Also, at inference time, if we encounter a test sequence longer than any training sequence, we may not have the corresponding saved normalization statistics for those time steps, which makes BN hard to apply.

#### 2. LayerNorm in detail

##### 2.1 LN in an MLP

Consider LN in an MLP.
Let $H$ be the number of hidden units in a layer, and $l$ be the layer index.
LN computes the normalization statistics $\mu$ and $\sigma$ as:

$$
\mu^{l} = \frac{1}{H} \sum_{i=1}^{H} a^l_i ~~~~~~~
\sigma^{l} = \sqrt{\frac{1}{H} \sum_{i=1}^{H}(a^l_i-\mu^l)^2}
$$

Note that these statistics do **not** depend on batch size; they only depend on the number of hidden units.
If $H$ is sufficiently large, the estimated statistics can still be stable.
The normalized activation is:

$$
\hat{a}^l = \frac{a^l-\mu^l}{\sqrt{(\sigma^l)^2+\epsilon}} \tag{1}
$$

where $\epsilon$ is a small constant to avoid division by zero.

LN also uses learnable parameters to preserve representational capacity: gain $g$ and bias $b$.
With activation function $f$, the LN output is:

$$
h^l = f(g^l \odot \hat{a}^l + b^l) \tag{2}
$$

Combining (1) and (2) and omitting the layer index $l$:

$$
h=f(\frac{g}{\sqrt{\sigma^2+\epsilon}} \odot (a-\mu) + b)
$$

##### 2.2 LN in an RNN

For an RNN at time step $t$, the input is the previous hidden state $h^t$ at time step $t-1$ and the current input $\text{x}_t$.
It can be written as:

$$
	ext{a}^t = W_{hh}h^{t-1}+W_{xh}\text{x}^{t}
$$

Then we can apply the same normalization procedure on $\text{a}^t$ as above:

$$
h^t=f(\frac{g}{\sqrt{\sigma^2+\epsilon}} \odot (a^t-\mu^t) + b) ~~~~~~
\mu^{t} = \frac{1}{H} \sum_{i=1}^{H} a^t_i ~~~~~~~
\sigma^{l} = \sqrt{\frac{1}{H} \sum_{i=1}^{H}(a^t_i-\mu^t)^2}
$$


