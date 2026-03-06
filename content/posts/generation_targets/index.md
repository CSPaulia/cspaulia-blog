---
title: "为生成模型构建训练目标"
date: 2026-02-03T11:10:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "生成模型"
    subseries: "模型训练"
categories: ["生成模型"]
tags: ["流匹配", "扩散模型"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "MIT 课程《Flow Matching and Diffusion》Lecture 2 笔记"
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: false # to disable highlightjs
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
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: true # only hide on current single page
    hiddenInList: true # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## 1. 训练目标

训练 = 找到参数 $\theta$，使得:

$$
X_0 \sim p_{init}, dX_t = u_t^{\theta}(X_t) dt~\text{or}~dX_t = u_t^{\theta}(X_t) dt + \sigma_t dW_t
$$

最终找到：

$$
X_1 \sim p_{data}
$$

在回归和分类任务中，训练目标往往是数据标签（label），然而在生成模型中，训练目标为向量场 $u_t^{\theta}$。因此，我们通过最小化均方差（MSE）来拟合向量场：

$$
L(\theta) = || u_t^{\theta}(x) - u_t^{target}(x) ||^2
$$

<img src="denoise.png" alt="Denoised Image" width="100%" />

## 2. 条件概率路径与边缘概率路径

**定义一 狄拉克测度**（Dirac measure，也可理解为点质量分布）：对 $z \in \mathbb{R}^d$，若 $X \sim \delta_z$，则 $X = z~a.s.$（即 $P(X = z) = 1$）。 

> 测度： 衡量集合大小的函数。
> - 长度测度：在 $(\mathbb R)$ 上，区间 $([a,b])$ 的测度是长度 $(b-a)$；
> - 面积测度：在 $(\mathbb R^2)$ 上，矩形 $([a,b]\times[c,d])$ 的测度是面积 $(b-a)(d-c)$；
> - 概率测度：在概率空间 $(\Omega, \mathcal{F}, P)$ 上，事件 $A \in \mathcal{F}$ 的测度是概率 $P(A)$。
>
> 狄拉克测度 $\delta_z$ 是一种特殊的概率测度，它将所有质量集中在单个点 $z$ 上：
> $$
> \delta_z(A)=
> \begin{cases}
> 1, & z\in A \\\\
> 0, & z\notin A \\\\
> \end{cases}
> $$

**定义二 条件概率路径（Conditional Probability Path）**：$\{P_t(\cdot|z), t \in [0,1]\}$，满足：
1. $P_t(\cdot|z)$ 是定义在 $\mathbb{R}^d$ 上的一个分布；
2. $P_0(\cdot|z) = P_{init},~P_1(\cdot|z) = \delta_z$。其中 $\delta_z$ 为狄拉克测度。

> 举例：高斯条件概率路径
> 
> $$
> P_t(\cdot|z) = \mathcal{N}(\alpha_t z, \sigma_t^2 I_d)
> $$
>
> 其中，我们令噪声调度函数（noise schedule）满足 $\alpha_t = t, \sigma_t = 1 - t$，则有 $\alpha_0 = 0, \sigma_0 = 1$，以及 $\alpha_1 = 1, \sigma_1 = 0$。
> 高斯条件概率路径如下图所示：
>
> <img src="distribution_variance.png" alt="Conditional Probability Path" width="100%" />

**定义三 边缘概率路径（Marginal Probability Path）**：已知 $z \sim P_{data}$，$x \sim P_t(\cdot|z)$，边缘概率路径 $\{P_t, t \in [0,1]\}$（该分布与 $z$ 无关）满足：
1. $P_t(x) = \int P_t(x|z) P_{data}(z) dz$；
2. $P_0 = P_{init},~P_1 = P_{data}$。

## 3. 条件向量场与边缘向量场

**定义四 条件向量场**（Conditional Vector Field）：$u_t^{target}(x|z), t \in [0,1], x,z \in \mathbb{R}^d$，使得：

$$
X_0 \sim P_{init}, \frac{d}{dt} X_t = u_t^{target}(X_t|z)
$$

可推出 $X_t$ 满足条件概率路径：

$$
X_t \sim P_t(\cdot|z), t \in [0,1]
$$

> $P_{init}$ 往往等于 $P_0(\cdot|z)$。

> 举例：高斯条件向量场
>
> 已知高斯条件概率路径为：
>
> $$
> P_t(\cdot|z) = \mathcal{N}(\alpha_t z, \sigma_t^2 I_d)
> $$
>
> 由于 $X_t \sim P_t(\cdot|z)$，因此 $X_t$ 可表示为：
>
> $$
> X_t = \alpha_t z + \sigma_t \epsilon,~\epsilon \sim \mathcal{N}(0, I_d)
> $$
>
> 对 $X_t$ 关于 $t$ 求导，得到高斯条件向量场：
>
> $$
> \frac{d}{dt} X_t = \dot{\alpha}_t z + \dot{\sigma}_t \epsilon = \dot{\alpha}_t z + \dot{\sigma}_t \frac{X_t - \alpha_t z}{\sigma_t} = \left(\dot{\alpha}_t - \frac{\dot{\sigma}_t}{\sigma_t}\alpha_t \right) z + \frac{\dot{\sigma}_t}{\sigma_t} X_t
> $$
>
> 即
>
> $$
> u_t^{target}(x|z) = \left(\dot{\alpha}_t - \frac{\dot{\sigma}_t}{\sigma_t}\alpha_t \right) z + \frac{\dot{\sigma}_t}{\sigma_t} x
> $$
>
> <img src="conditional_vector_field_2d.gif" alt="Conditional Vector Field" width="100%" />