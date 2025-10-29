---
title: "流匹配与扩散模型"
date: 2025-10-10T8:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "Generative Model"
    subseries: "Flow Matching"
categories: ["AIGC"]
tags: ["Flow Matching", "Diffusion"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "Notes for Flow Matching and Diffusion from MIT course."
# canonicalURL: "https://canonical.url/to/page"
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

## 1. 生成对象

### 1.1. 将生成对象表征为向量

1. 图像：
    - 尺度维度：高度 $H$ 和 宽度 $W$
    - 颜色维度：三维颜色通道 RGB

$$
z \in \mathbb{R}^{H \times W \times 3}
$$

2. 视频：
    - 时间维度：时间帧 $T$
    - 每一帧为图像

$$
z \in \mathbb{R}^{T \times H \times W \times 3}
$$

3. 分子结构：
    - N 个原子
    - 每个原子包含三个维度
  
$$
z \in \mathbb{R}^{N \times 3}
$$

**我们可以将我们想要生成的目标表征为向量：**

$$
z \in \mathbb{R}^{d}
$$

---

### 1.2. 生成——从数据分布中采样

**数据分布**：我们想要生成的对象的分布$p_{data}$

**概率密度**：$p_{data}: \mathbb{R}^d \to \mathbb{R} \ge 0, z \mapsto p_{data}$

{{< alert type="warning" title="注意" >}}
我们并不知道实际的概率密度
{{< /alert >}}

**生成**：意味着从数据分布中进行采样 $z \sim p_{data}$，采样 $z$ 如下：

<img src="Bao.jpg" alt="Baobao" width="300"/>

---

### 1.3. 数据集——由数据分布中的样本组成

**数据集**：数据分布中有限数量的样本的集合 $z_1, \cdots, z_N \sim p_{data}$

---

### 1.4. 条件生成

**条件生成**：意味着从条件数据分布中进行采样 $z \sim p_{data}(\cdot | y)$

---

## 2. Flow and Diffusion Model

### 2.1. Flow Models

<dl class="definition-list">
  <dt>轨迹（Trajectory）</dt>
  <dd>
    <span class="math">$X: [0, 1] \to \mathbb{R}^d, t \mapsto X_t$
    </span>
  </dd>
</dl>

<img src="Traj.png" alt="Trajectory" width="300"/>

<dl class="definition-list">
  <dt>向量场（Vector Feild）</dt>
  <dd>
    <span class="math">$u: R^d \times [0,1] \to \mathbb{R}^d, (x, t) \mapsto u_t(x)$
    </span>
    <br>其中 $x$ 表示点的位置，$u_t$ 表示向量方向
  </dd>
</dl>

<img src="vector_field.png" alt="Vector Field" width="300"/>

<dl class="definition-list">
  <dt>常微分方程（Ordinary Differential Equation, ODE）</dt>
  <dd>
    初始条件
    <span class="math">$X_0 = x_0$
    </span>
    <br>
    常微分方程/动力学方程
    <span class="math">$dX_t/dt = u_t(X_t)$
    </span>
  </dd>
</dl>

<img src="ode.png" alt="ODE" width="300"/>

> $dX_t/dt$ 即为轨迹的切线，可以理解为速度，因此ODE描述了粒子在向量场中的运动

<dl class="definition-list">
  <dt>流（Flow）</dt>
  <dd>
    <span class="math">$\phi: \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d, (t, x) \mapsto \phi_t(x)$
    </span>
    <br>其中 $\phi_t(x)$ 表示时间 $t$ 时刻，位置为 $x$ 的粒子沿着ODE轨迹运动到的新位置
    <br><span class="math">$\phi_0(x_0) = x_0$
    </span>
    <br><span class="math">$\frac{d}{dt}\phi_t(x_0) = u_t(\phi_t(x_0))$
  </dd>
</dl>

流本质是针对许多初始条件的常微分方程解的集合

<img src="flow_process.png" alt="Flow Process" width="80%"/>

Flow 可视化：

<img src="flow.gif" alt="Flow" width="300"/>