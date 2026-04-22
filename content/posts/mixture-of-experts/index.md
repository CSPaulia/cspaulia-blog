---
title: "混合专家（Mixture of Experts，MoE）"
date: 2026-04-22T10:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "大语言模型"
    subseries: "架构与训练"
categories: ["大语言模型"]
tags: ["架构", "训练"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "关于混合专家（Mixture of Experts，MoE）的介绍和相关内容。"
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

## 1. 混合专家（Mixture of Experts，MoE）

<img src="moe_overall.png" alt="混合专家架构图"/>

- Dense FFN: 一个大 FFN，每个 token 都要跑它
- MoE FFN: $N$ 个 expert FFN，每个 token 只跑其中 1 个

### 1.1. MoE 的优点

1. 专家数量越多，模型性能越优，而 FLOPs 不变，可参考[图片](more_experts.png)[1]；
2. 在 MoE 模型的激活参数量与 Dense 模型一致的情况下， MoE 模型的训练速度更快，可参考[图片](moe_train.png)[2]；
3. MoE 模型可将专家分布在不同的设备上，单卡显存需求降低，可参考[图片](moe_parallelization.png)。


---

## 参考文献

[1] Fedus W, Zoph B, Shazeer N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity[J]. Journal of Machine Learning Research, 2022, 23(120): 1-39.

[2] Muennighoff N, Soldaini L, Groeneveld D, et al. Olmoe: Open mixture-of-experts language models[J]. arXiv preprint arXiv:2409.02060, 2024.