---
title: "流匹配与扩散模型"
date: 2025-10-10T8:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "Generative Model"
    subseries: "Flow Matching"
draft: true
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
disableHLJS: true # to disable highlightjs
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