---
title: "My 1st post"
date: 2020-09-15T11:30:03+08:00
# weight: 1
# aliases: ["/first"]
# series:
#     main: "Large Language Model"
#     subseries: "Mainstream Series"
draft: true
categories: ["first"]
tags: ["first"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "Desc Text."
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

## Post Training 三阶段

> 来源于 InstructGPT[[1]](#ref1)

1. 收集数据并训练**监督**策略
    - 从提示词数据集中采样一个提示词
    - 标注者对数据进行标注（该标注即为期望输出）
    - 该标注数据用于对 LLM 进行监督训练
2. 收集**对比数据**并训练**奖励模型**
    - 采样一个提示词及多个模型可能输出的答案选项
    - 标注者从“最好”到“最差”对这些选项进行排序
    - 该标注数据用于对奖励模型进行训练
3. 在**训练好的奖励模型**的加持下利用**强化学习**优化策略
    - 从数据集中采用一个新的提示词
    - 利用策略生成一个输出
    - 奖励模型计算输出的奖励分数（Reward）
    - 根据奖励分数利用 PPO 等强化学习方法对策略进行更新

## SFT 数据集的构建

### 数据集中的问题

> 来自 FLAN 数据集的一个例子:
>
> What is this text about? OPTIONS: - World - Sport - Business - Science/Tech

- 自然的对话一般不包含这样的选项

> 来自 OpenAssistant 的例子:
>
> Question: Can you write a short introduction about the relevance of the term"monopsony" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research.
>
> Answer: "Monopsony" refers to a market structure where there is only one buyer for a particular good or service. In economics, this term is particularly relevant in the labor market, where a monopsony employer has significant power over the wages and working conditions of their employees. [...]. Overall, the concept of monopsony is essential to understanding the dynamics of labor markets and the impact of market power on workers. Further research is needed to understand the extent and impact of monopsonies on the economy and to develop policies to address this issue. References: Bivens, J., & Mishel, L.(2013). The Pay of Corporate Executives and Financial Professionals as Evidence of Rents in Top 1 Percent Incomes. Journal of Economic Perspectives, 27(3), 57-78.

- 人类标注者很难撰写如此**长而详细**的答案
- 输出中的符号（甚至风格）如何抉择
- 是否输出**参考文献**
    - 如何输出准确的参考文献：我们往往希望模型能够输出更加严谨、学术的输出，因此我们会在数据集的回答中加入参考文献，但是模型可能只学到了“输出内容要加上参考文献”，这会导致模型产生幻觉（Hallucination）
- 是否输出**复杂的知识**
- 数据集的规模大小
- 数据集的安全问题
- 如何决定模型输出的长短：通过下图你可以发现，不同指令数据的输入和输出长度区别是很大的

![instruction_dataset](./instrction_dataset.png)

### SFT 数据集构建经验

1. SFT 在模型已经具备某些能力的前提下，通过数据来“抽取”这些能力的表现效果最好；但如果试图用 SFT 去“添加”模型原本不具备的新行为，往往效果不佳
2. 并不是所有事实正确的数据都会提升模型表现，有时即使是高质量的事实数据，也可能干扰模型已有的分布或对齐，反而让性能下降
3. 某些类型的数据（例如安全性、遵循指令、风格等）哪怕只有少量，也能对模型带来巨大提升，不过，模型的长尾行为（覆盖面广、稀疏分布的场景）则更依赖于大量数据来改善

### 在预训练中使用指令微调（Instruction Tuning）

1. 在网页或预训练数据集上进行预训练
2. 将指令微调数据混入预训练中
3. 额外进行一个简短的指令微调

### 'Midtraining' / 'Two-phase Training'

![minicpm](./minicpm.png)

这种方案[[2]](#ref2)好似已被大部分 LLM 公司采纳（但没有详细的文档）：在 Stable 阶段采用纯预训练数据集训练（如上图左侧所示）；在 Decay 阶段采用预训练+指令微调混合数据集进行训练（如上图右侧所示）

### 

---

<div class="zhihu-ref">
  <div class="zhihu-ref-title">参考文献</div>
  <ol>
    <li id="ref1"><a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf" target="_blank">Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback[J]. Advances in neural information processing systems, 2022, 35: 27730-27744.</a></li>
    <li id="ref2"><a href="https://arxiv.org/pdf/2404.06395" target="_blank">Hu S, Tu Y, Han X, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies[J]. arXiv preprint arXiv:2404.06395, 2024.</a></li>
  </ol>
</div>