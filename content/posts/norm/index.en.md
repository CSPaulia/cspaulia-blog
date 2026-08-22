---
title: "A Collection of N Normalization Methods"
date: 2025-05-21T21:15:00+08:00
series:
    main: "Deep Learning Foundations"
    subseries: "Normalization Methods"
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

<p align="center">
  {{< img src="LNvsBN.jpg" alt="LN vs BN" >}}
</p>

In the figure above, \(N\) denotes the sample axis, \(C\) the channel axis, and \(F\) the number of features per channel. BN normalizes <strong>the same channel across different samples</strong>, while LN normalizes <strong>different channels within the same sample</strong>.

## 1. Batch Normalization (BN)

### 1.1 BN and Batch Size

BN computes normalization statistics from the samples in a batch. When the batch is very small—for example, only four samples—the estimated mean and variance may not represent the overall data distribution, so BN can perform poorly.

### 1.2 BN and Recurrent Neural Networks

<p align="center">
  {{< img src="RNN.jpg" alt="RNN" >}}
</p>

Sequence lengths often differ within a batch of recurrent neural network (RNN) inputs. At later time steps, such as \(t>4\) in the figure, only one sample may still contain data. Statistics computed from that sample cannot represent the overall distribution, so BN is ineffective in this setting.

In addition, if a test sequence is longer than every training sequence, the later time steps have no statistics saved during training, which makes BN difficult to apply.

## 2. Layer Normalization (LN)

### 2.1 LN in an MLP

Consider LN in a multilayer perceptron (MLP). Let \(H\) be the number of hidden units in a layer and \(l\) the layer index. The normalization statistics \(\mu\) and \(\sigma\) are

\[
\mu^{l} = \frac{1}{H} \sum_{i=1}^{H} a^l_i ~~~~~~~
\sigma^{l} = \sqrt{\frac{1}{H} \sum_{i=1}^{H}(a^l_i-\mu^l)^2}
\]

These statistics do not depend on the number of samples in the batch; they only depend on the hidden units of the current sample. Using \(\mu^{l}\) and \(\sigma^{l}\), the normalized value is

\[
\hat{a}^l = \frac{a^l-\mu^l}{\sqrt{(\sigma^l)^2+\epsilon}} \tag{1}
\]

where \(\epsilon\) is a small constant that prevents division by zero.

LN also uses a learnable gain \(g\) and bias \(b\) to restore the scale and shift required by the model. Given an activation function \(f\), the final output is

\[
h^l = f(g^l \odot \hat{a}^l + b^l) \tag{2}
\]

Combining Equations (1) and (2), and omitting the layer index \(l\), gives

\[
h=f(\frac{g}{\sqrt{\sigma^2+\epsilon}} \odot (a-\mu) + b)
\]

### 2.2 LN in an RNN

For an RNN at time step \(t\), the input consists of the hidden state \(h^{t-1}\) from the preceding time step and the current input \(x^t\):

\[
a^t = W_{hh}h^{t-1}+W_{xh}x^t.
\]

The same normalization procedure used in the MLP can then be applied to \(a^t\):

\[
h^t=f(\frac{g}{\sqrt{\sigma^2+\epsilon}} \odot (a^t-\mu^t) + b) ~~~~~~
\mu^{t} = \frac{1}{H} \sum_{i=1}^{H} a^t_i ~~~~~~~
\sigma^{t} = \sqrt{\frac{1}{H} \sum_{i=1}^{H}(a^t_i-\mu^t)^2}
\]

## 3. Root Mean Square Layer Normalization (RMSNorm)

RMSNorm does not subtract the mean. Instead, it scales features according to the root mean square of the input. For a \(d\)-dimensional input \(x\), it is computed as

\[
\begin{aligned}
\operatorname{RMS}(x)
&=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},\\
y_i
&=g_i\frac{x_i}{\operatorname{RMS}(x)}.
\end{aligned}
\]

Here, \(g_i\) is a learnable gain that usually has the same dimensionality as the features. Unlike LN, RMSNorm neither computes the mean nor centers the input; it only controls the input's overall magnitude. It therefore retains LN's independence from batch size while eliminating the mean calculation and subtraction.

## References

[1] Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. [Online]. Available: https://arxiv.org/abs/1910.07467
