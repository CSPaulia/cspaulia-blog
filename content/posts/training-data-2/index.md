---
title: "大语言模型训练数据 II（Data II）：过滤、去重、数据混合与合成数据"
date: 2026-08-24T10:30:03+08:00
series:
  main: "大语言模型"
  subseries: "训练数据"
categories: ["大语言模型", "训练数据"]
tags: ["训练数据", "Training Data", "数据过滤", "去重", "数据混合", "合成数据"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "CS336 Lecture 14 学习笔记：训练数据的过滤、去重、数据混合与合成数据——从海量原始语料中挑出真正值得训练的部分。"
disableHLJS: false
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
    image: "the-pile.png"
    alt: "The Pile 的 22 个子数据集占比"
    caption: "The Pile 的构成：22 个子数据集按大小占比。"
    relative: true
---

这一讲回答"拿到原始数据之后，如何把它处理成真正可训练的数据"。本讲内容分为两部分：

- **数据管线（Data Pipeline）**：转换（Transformation）、过滤（Filtering）、去重（Deduplication）、混合（Mixing）。原始语料要依次经过这四个环节，才能变成高质量、低冗余、按比例配好的预训练数据；
- **合成数据（Synthetic Data）**：面向中期训练（Mid-training）与指令微调（Supervised Fine-Tuning，SFT）阶段，用模型生成的数据补充训练信号。

## 1. 转换（Transformation）：把原始格式变成纯文本

原始数据并不是现成的纯文本：网页是 HTML、论文是 PDF（arXiv）、代码是仓库目录。转换（Transformation）是数据管线的第一环，负责把这些格式提取成可训练的文本。

### 1.1 HTML 转文本：去除样板，方法选择影响下游精度

最常见的转换是 HTML 转文本：

- 去除样板内容（导航栏、广告等），只保留正文；
- 图片、表格要么丢弃、要么转写成文字。转换本质上有损，需要把二维版面线性化成文本流；
- 常用规则类工具：[trafilatura](https://trafilatura.readthedocs.io/en/latest/)、[resiliparse](https://resiliparse.chatnoir.eu/en/stable/)、[jusText](https://pypi.org/project/jusText/)、[lynx](https://lynx.invisible-island.net/) 等；
- 准确性很重要：抽取方式会直接影响下游任务精度，[DataComp-LM](https://arxiv.org/abs/2406.11794) 对此做了系统对比。

<figure>
  <img src="dclm-wet.png" alt="DataComp-LM 论文中不同文本抽取方式的下游任务准确率对比" loading="lazy">
  <figcaption>DataComp-LM 对比不同文本抽取方式：Common Crawl 自带的 WET 文本（12.2–12.5）明显低于 trafilatura 与 resiliparse（13.4–24.5）。图源：<a href="https://arxiv.org/abs/2406.11794">DataComp-LM 论文</a>。</figcaption>
</figure>

### 1.2 PDF 转文本：FinePDFs 的重新爬取、OCR 与清洗

PDF 比 HTML 难处理得多：HTML 有清晰的标签树，PDF 只有绘制指令——"在这里画一个字、在那里画一条线"，视觉上忠实但没有语义结构。[FinePDFs](https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog) 管线从 Common Crawl 的 PDF 出发：

- 重新爬取被截断的 PDF（PDF 文件大，爬虫下载常不完整）；
- 用 VLM 做 OCR（RolmOCR），或用 [Docling](https://github.com/docling-project/docling) 提取文本；
- 随后还有大量清理与过滤；
- 局限：大量版式信息（layout）在转换中丢失。

<figure>
  <img src="finepdfs.webp" alt="PDF 的源码结构与视觉版面对比" loading="lazy">
  <figcaption>PDF 的"解剖图"：左侧是源码结构（一连串绘制指令），右侧是视觉版面；PDF 保存的是外观而非结构，两者之间的鸿沟就是缺失的语义信息。图源：<a href="https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog">FinePDFs 博客</a>。</figcaption>
</figure>

## 2. 过滤（Filtering）：从原始数据中挑出与目标数据相似的部分

### 2.1 问题定义：目标数据 T 与原始数据 R

过滤的算法框架是：给定一些目标数据（Target Data）T 与大量原始数据（Raw Data）R，从 R 中找出与 T 相似的子集 T'。

<figure>
  <img src="raw-target-schema.png" alt="过滤的算法框架：从原始数据 R 中找出与目标数据 T 相似的子集 T'" loading="lazy">
  <figcaption>过滤的算法框架：给定目标数据 T 与原始数据 R，从 R 中找出与 T 相似的子集 T'。图源：CS336 Lecture 14 课件。</figcaption>
</figure>

### 2.2 三个应用与两个设计要求

过滤有三个典型应用：

- 语言识别（Language Identification）：英语 vs 其余语言；
- 质量过滤（Quality Filtering）：高质量 vs 低质量；
- 毒性过滤（Toxicity Filtering）：无毒 vs 有毒。

设计过滤算法有两个要求：

1. 能从目标数据泛化：希望 T' 与 T 不同，而不是复制 T 本身；
2. 必须极快：要在海量的 R 上运行。

数据选择方向的综述见 [Albalak 等 2024 的论文](https://arxiv.org/abs/2402.16827)。

### 2.3 通用框架：打分函数与两类打分器

通用框架分两步：先基于 R 与 T 估计一个模型、得到打分函数（Scoring Function），再按分数保留 R 中的样本。打分器有两类：

- 目标数据的生成式模型（KenLM）：\(score(x) = p_T(x)\)，用目标数据训练的语言模型直接给文本打分；
- 简单分类器（fastText）：\(score(x) = p(T \mid x)\)，训练分类器预测文本属于 T 的概率。

使用时保留 \(score(x)\) 不低于阈值的样本；保留可以是随机化的（后文 GPT-3 即是一例）。

是否采用模型式过滤，各模型选择不同：C4、Gopher、RefinedWeb、FineWeb、Dolma 刻意不使用；GPT-3、LLaMA、DCLM 使用（正在成为主流）。

### 2.4 语言识别：fastText 分类器与 Dolma 的阈值

语言识别的目标是找出特定语言（如英语）的文本。常用的 [fastText 语言识别](https://fasttext.cc/docs/en/language-identification.html) 是现成分类器：支持 176 种语言，训练数据来自多语言网站——Wikipedia、Tatoeba（翻译网站）与 SETimes（东南欧新闻）。[Dolma](https://arxiv.org/abs/2402.00159) 只保留 p(English) ≥ 0.5 的页面。

### 2.5 质量过滤：OpenWebMath、GPT-3、LLaMA、phi-1

**OpenWebMath**（[Paster 等 2023](https://arxiv.org/abs/2310.06786)）的目标是从 Common Crawl 整理高质量数学文本语料，流程是规则加两级模型过滤：

- 先用规则过滤（例如包含 LaTeX 命令）；
- 再用在 ProofPile 上训练的 KenLM 打分，移除困惑度超过 15000 的文档；
- 最后用 fastText 分类器预测文档是数学写作的概率：提取出 LaTeX 公式的文档，概率超过 0.17 即保留；没有 LaTeX 公式的文档，要求概率超过 0.8。

> 结果：OpenWebMath 产出 14.7B token；用它训练的 1.4B 模型，表现超过用 20 倍以上 token 训练的模型。

**GPT-3**（[Brown 等 2020](https://arxiv.org/abs/2005.14165) 附录 A）训练基于词特征的线性分类器（[Spark 的 tokenizer](https://spark.apache.org/docs/latest/ml-features#tokenizer)）：

- 正样本：来自 {Wikipedia、WebText2、Books1、Books2}；
- 负样本：来自 Common Crawl；
- 按分数随机保留文档：

```python
def keep_document(score: float) -> bool:
    return np.random.pareto(9) > 1 - score
```

**LLaMA / RedPajama**（[Touvron 等 2023](https://arxiv.org/abs/2302.13971)）的做法更简单：正样本是 Wikipedia 引用的页面，负样本来自 Common Crawl，只保留被分类为正例的文档。

**phi-1**（[Gunasekar 等 2023](https://arxiv.org/abs/2306.11644)）的理念是用极高质量的数据（教科书级）训练小模型（1.5B），数据包含 GPT-3.5（后来是 GPT-4）生成的合成数据与过滤后的数据。过滤流程：

1. 原始数据 R = The Stack 的 Python 子集；
2. 用 GPT-4 以提示词 "determine its educational value for a student whose goal is to learn basic coding concepts" 对 R 的 10 万样本分类，得到正例 T；
3. 用预训练 CodeGen 模型的输出 embedding 训练随机森林分类器；
4. 从 R 中挑选分类为正例的数据。

在 [HumanEval](https://huggingface.co/datasets/openai_humaneval) 上，用过滤后子集训练的 1.3B 模型以更少的步数达到了更高的准确率：

| 训练数据 | 训练步数 | HumanEval 准确率 |
|---|---|---|
| The Stack 的 Python 子集 | 96K | 12.19% |
| phi-1 过滤后的子集 | 36K | 17.68% |

### 2.6 毒性过滤：Dolma 与 Jigsaw 数据集

[Dolma](https://arxiv.org/abs/2402.00159) 的毒性过滤使用 [Jigsaw Toxic Comments 数据集](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)（2018）。该数据集源自 [Jigsaw 毒性评论分类竞赛](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)，竞赛的目标是帮助人们更好地在线讨论；数据是 Wikipedia 讨论页的评论，带六类标注——toxic（有毒）、severe_toxic（严重有毒）、obscene（淫秽）、threat（威胁）、insult（侮辱）、identity_hate（身份仇恨）。

### 2.7 过滤阈值随训练时长变化：没有单一最优值

过滤阈值没有单一最优值，取决于训练时长：

- 训练时间越长，需要越多（更低质量）的数据；
- 训练时间越短，需要越少（更高质量）的数据。

<figure>
  <img src="data-filtering-scale.png" alt="过滤的规模依赖效应：训练时长决定最优过滤强度" loading="lazy">
  <figcaption>过滤的规模依赖效应：训练时长不同，最优过滤强度也不同。图源：CS336 Lecture 14 课件。</figcaption>
</figure>

### 2.8 小结：过滤的配方

- 过滤对训练出好模型至关重要；
- 配方：先定义目标数据（"好"长什么样），再外推到原始数据。

## 3. 去重（Deduplication）：定义、动机与设计空间

### 3.1 两类重复：精确重复与近似重复

重复数据有两类：

- 精确重复（Exact Duplicates）：完全相同的副本，如镜像站、GitHub fork（[Gutenberg 的镜像列表](https://www.gutenberg.org/MIRRORS.ALL)就是现成的例子）；
- 近似重复（Near Duplicates）：只差几个 token 的相同文本。

近似重复的典型例子：

- 服务条款与许可证（如 [MIT 许可证](https://opensource.org/license/mit)）；
- 套话式写作（复制粘贴或模板生成）；
- 复制粘贴过程中的细微格式差异。

<figure>
  <img src="dedup-examples.png" alt="去重论文 Table 1 中的近似重复示例" loading="lazy">
  <figcaption>近似重复示例（论文 Table 1）：多份文档除模板字段外完全相同（最后一行），是套话式写作的典型形态。图源：<a href="https://arxiv.org/abs/2107.06499">Lee 等 2021</a>。</figcaption>
</figure>

C4 中的极端例子：一条产品描述在 C4 中逐字重复了 61,036 次：

> "by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We'd be honored if you can apply some or all of these design in your wedding. believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!"

（[示例商品页面](https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG)）

### 3.2 为什么去重：训练更高效、记忆更少

去重训练数据能让语言模型更好（[Lee 等 2021](https://arxiv.org/abs/2107.06499)）：

- 训练更高效：token 更少；
- 避免记忆：缓解版权与隐私问题。

### 3.3 设计空间：单元、匹配方式与保留策略

去重有三个设计选择：

1. 单元是什么：句子、段落还是文档？
2. 如何匹配：精确匹配、存在公共子单元，还是公共子单元的占比？
3. 采取什么动作：全部删除，还是只保留一份？

### 3.4 关键挑战：两两比较需要线性时间算法

- 去重本质上是对"条目与条目"做两两比较；
- 要扩展到海量数据，需要线性时间算法。

### 3.5 哈希函数：以碰撞风险换取速度

哈希函数（Hash Function）h 把条目映射为一个哈希值（整数或字符串），哈希值远小于条目本身；不同条目可能映射到相同的哈希值，即哈希碰撞（Hash Collision）：x ≠ y 但 h(x) = h(y)。

效率与抗碰撞性之间存在权衡（[相关讨论](https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed)）：

- 密码学哈希（Cryptographic Hash，如 SHA-256）：抗碰撞，但慢（比特币使用）；
- DJB2、MurmurHash、CityHash：不抗碰撞，但快（哈希表使用）。

本讲使用 MurmurHash，例如 `mmh3.hash("hello")` 的哈希值是 613153351。

#### 3.5.1 精确去重：哈希分组后每组保留一份

最简单的情形：以整条字符串为单元，只做精确匹配，重复的条目只保留一份。

```python
items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]
hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)
deduped_items = [next(group) for h, group in hash_items]
```

实现上，先按哈希值排序、把相同哈希的条目分到一组、每组取一个：两个 "hello" 只留一个，6 个条目去重后剩 5 个。

- 优点：简单、语义清晰、精确度高；
- 缺点：无法处理近似重复；
- 代码以 MapReduce 的方式写成，易于并行化与扩展。

[C4](https://arxiv.org/abs/1910.10683) 的思路相同，但以 3 句组成的片段为单元，做精确匹配，重复的片段只保留一份。注意：从文档中间移除一个重复的 3 句片段后，剩余文档可能不再连贯。

#### 3.5.2 Jaccard 相似度与 MinHash：碰撞概率等于相似度

近似重复的判断需要一个相似度度量：Jaccard 相似度（Jaccard Similarity）。

\[ Jaccard(A, B) = \frac{|A \cap B|}{|A \cup B|} \]

例：A = {1, 2, 3, 4}，B = {1, 2, 3, 5}，交集 3 个、并集 5 个，Jaccard = 0.6。

定义：两份文档是近似重复，当它们的 Jaccard 相似度不低于阈值。

算法挑战：在线性时间内找出近似重复。

**MinHash** 是一个随机哈希函数 h，使得 Pr[h(A) = h(B)] = Jaccard(A, B)。通常我们期望不同条目哈希到不同值，但这里恰恰相反：希望碰撞概率随相似度变化。

```python
def minhash(S: set[str], seed: int):
    return min(mmh3.hash(x, seed) for x in S)
```

用特征矩阵（Characteristic Matrix）表示：行是条目，列是集合。

| item | A | B |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 3 | 1 | 1 |
| 4 | 1 | 0 |
| 5 | 0 | 1 |

随机哈希函数诱导了条目上的一个随机排列；看哪个条目排在第一个（即最小值）。每个条目成为最小值的概率相同：

- 若 1、2、3 最先出现：A 与 B 的最小哈希值相同；
- 若 4 或 5 最先出现：A 与 B 的最小哈希值不同。

用 100 个随机哈希函数验证：估计出的 Jaccard 为 0.6，与真值一致。

不过，一次碰撞并不能告诉我们 Jaccard(A, B) 是否超过阈值——这就是下一节 LSH 要解决的问题。

#### 3.5.3 局部敏感哈希：用"与-或"的带结构锐化阈值

局部敏感哈希（Locality Sensitive Hashing，LSH，[MMDS 教材第 3 章](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf)）的目标是：相似度超过阈值的文档对以很高的概率碰撞，低于阈值的文档对几乎不碰撞。

只用一个 MinHash 达不到这个目标：碰撞概率等于 Jaccard 相似度，相似度 0.8 的对只有 80% 概率碰撞，相似度 0.2 的对也有 20% 概率碰撞——单次判断的随机性太大，区分不开高相似度与低相似度。解决思路是用多个哈希函数，把"一组哈希全部撞上"当作更强的信号。

具体做法：取 n 个哈希函数，分成 b 组、每组 r 个（n = b·r），每组称为一个带（Band）。例如 n = 12、b = 3、r = 4，即 12 个哈希函数分成 3 个带、每带 4 个：

```
h1 h2 h3 h4 | h5 h6 h7 h8 | h9 h10 h11 h12
```

判定规则：只要存在一个带，其中 r 个哈希值全部相同，A 与 B 就算碰撞。这就是"与-或"结构：带内取"与"（r 个哈希必须全同），带间取"或"（任一带全中即可）。低相似度的对很难在一个带内全部撞上，高相似度的对总有一个带会全部撞上，于是碰撞概率在阈值附近被锐化成 S 形曲线。

给定 sim = Jaccard(A, B)：

- 单个带匹配的概率：\(sim^r\)；
- 碰撞的概率：\(1 - (1 - sim^r)^b\)。

例：sim = 0.8，b = 5，r = 10，单带匹配概率 0.107，碰撞概率 0.433。

<figure>
  <img src="lsh-curve.png" alt="LSH 的 S 形碰撞概率曲线（b=5，r=10）" loading="lazy">
  <figcaption>LSH 碰撞概率随相似度变化的 S 形曲线（b=5，r=10）：相似度 0.8 时碰撞概率约 0.43。图源：CS336 Lecture 14 课件。</figcaption>
</figure>

调节 b 与 r 的效果：

- 增大 r：曲线更陡并向右移（更难匹配）；
- 增大 b：曲线向左移（更容易匹配）。

<figure>
  <img src="lsh-b-r-curves.png" alt="不同 b 与 r 下的 LSH 碰撞概率曲线族" loading="lazy">
  <figcaption>不同 b 与 r 下的曲线族：r 控制阈值位置与陡峭程度，b 控制整体左移程度。图源：CS336 Lecture 14 课件。</figcaption>
</figure>

<details>
  <summary>展开：Lee 等 2021 的设置（n = 9000，b = 20，r = 450）</summary>

[Lee 等 2021](https://arxiv.org/abs/2107.06499) 的设置在阈值处发生相变（Phase Transition）：阈值相似度为 \((1/b)^{1/r}\)，代入 b = 20、r = 450 约为 0.9934。在这个阈值上：

- 单个带匹配的概率为 \(1/b\)，代入 b = 20 得 0.05；
- A 与 B 碰撞的概率为 \(1 - (1 - 1/b)^b\)，代入 b = 20 得 0.6415，接近 \(1 - 1/e\)。

</details>

## 4. 数据混合（Data Mixing）：为多个数据源分配采样权重

### 4.1 问题：多个数据源之间如何加权

语言模型在多个数据源上训练。[Marin 数据集的 token 查看器](https://huggingface.co/spaces/marin-community/token-count-viewer)可以直观看到各数据集的规模：

<figure>
  <img src="marin-token-viewer.png" alt="Marin 数据集的 token 数量查看器截图" loading="lazy">
  <figcaption>Marin 数据集的 token 查看器：直观对比各数据集的规模。图源：<a href="https://huggingface.co/spaces/marin-community/token-count-viewer">Marin token viewer</a>。</figcaption>
</figure>

[The Pile](https://arxiv.org/abs/2101.00027) 是经典例子，由 22 个子数据集组成：

<figure>
  <img src="the-pile.png" alt="The Pile 数据集的 22 个子数据集占比" loading="lazy">
  <figcaption>The Pile 的构成：22 个子数据集按大小占比。图源：<a href="https://arxiv.org/abs/2101.00027">The Pile 论文</a>。</figcaption>
</figure>

关键问题：训练时应该以什么样的分布在这些数据源上采样？例如给 {Wikipedia、Common Crawl、GitHub} 分配 {0.3、0.5、0.2} 的权重，就是其中一种混合（Mixture）。

### 4.2 三个基线：直觉、均匀与按量加权

三种常见的基线做法：

- 直觉（Vibes）：凭直觉手工设定 p(s)，相当常见；
- 均匀采样（Uniform Sampling）：所有源等权重，\(p(s) \propto 1\)；
- 按量混合（Proportional Mixing）：按数据源的 token 数加权，\(p(s) \propto \text{num\_tokens}(s)\)。

直觉上应该给质量更高的源更大权重，但有两个顾虑：

1. 多样性：文学、代码、论文等数据源互相不可替代；
2. 数据量有限：给小数据源的权重过高，就不得不反复训练（epoch）它。

<details>
  <summary>展开：例子——权重过高会反复训练小数据源</summary>

低质量源有 10T token（充足），高质量源只有 10B token（稀缺）。按各 50% 的权重训练 1T token：

- 低质量源只用掉总量的 5%（0.05 epoch）；
- 高质量源被反复训练 50 遍（50 epoch）。

对高质量数据反复训练 50 遍，会导致过拟合。

</details>

<details>
  <summary>展开：UniMax 的 epoch 硬上限</summary>

[UniMax](https://arxiv.org/abs/2304.09151) 要解决的是多语言模型中各语言的平衡问题：

- 此前的工作在均匀与按量之间插值：\(p(s) \propto \text{num\_tokens}(s)^{\alpha}\)，其中 \(\alpha \in [0, 1]\)；
- UniMax 的想法是均匀采样，但对任何数据源的 epoch 数设置硬上限 C：\(p(s) \times \text{num\_training\_tokens} \leq C\)。

</details>

### 4.3 回归式混合：把"混合 → 损失"当作函数来拟合

回归式混合（Regression-based Mixing，如 [RegMix](https://arxiv.org/abs/2407.01492)、[Olmix](https://arxiv.org/abs/2602.12237)）的思路与 Scaling Law 类似：在小规模上采样不同的混合、训练小模型，用回归拟合"混合 → 损失"，再优化出最优混合。流程：

1. 定义混合分布 p 上的先验（如 Dirichlet 分布）；
2. 选择回归方法（线性回归、梯度提升树等）；
3. 用下游评测定义优化目标（注意不要过拟合评测集！）；
4. 接受小规模与大规模之间的偏差（成本与精度的权衡）。

<figure>
  <img src="regmix.png" alt="RegMix 的回归式数据混合框架" loading="lazy">
  <figcaption>RegMix 的回归式数据混合框架：采样混合、训练小模型、用回归拟合"混合 → 损失"并优化。图源：<a href="https://arxiv.org/abs/2407.01492">RegMix 论文</a>。</figcaption>
</figure>

<details>
  <summary>展开：不同数据混合方法的对比</summary>

<figure>
  <img src="data-mixing-methods.png" alt="不同数据混合方法的对比" loading="lazy">
  <figcaption>不同数据混合方法的对比。图源：CS336 Lecture 14 课件。</figcaption>
</figure>

</details>

这套方法成立依赖两个希望：回归模型在最优混合处足够准确；小规模上找出的最优混合能迁移到大规模。

#### 规模依赖：小模型的最优混合可能让大模型过拟合

混合有规模依赖效应。沿用 4.2 的例子：小模型在低 token 数下训练时，给高质量源 0.9 的权重没有问题；但大模型若沿用这个混合，会在高质量数据上反复训练很多遍而过拟合。

#### 模拟 epoching：让小规模实验看起来像大规模

模拟 epoching（Simulated Epoching，[论文](https://arxiv.org/abs/2501.11747)）可以缓解这个问题，其总体思路是让小规模实验"看起来像"大规模实验（本课反复出现的主题），具体做法是把所有数据源按相同比例降采样。例子：小规模实验用 10B token、大规模训练用 1T token，比例为 0.01；降采样后，低质量源从 10T 变成 100B、高质量源从 10B 变成 100M。在降采样后的语料里，反复训练某个源过头的混合会让模型表现变差，因此最优混合会变得更均衡（例如低质量源 0.7、高质量源 0.3）。

### 4.4 小结：数据混合的配方

- 问题：如何给不同数据源（如维基百科、通用网络、代码）加权；
- 回归式混合：在小规模上估计"混合 → 损失"并优化（与 Scaling Law 的玩法类似）；
- 重要考量：epoch 与过拟合（用上限或模拟来解决）。

## 5. 后期训练数据（Post-training Data）：合成数据的配方与 SWE 案例

中期训练与 SFT 阶段同样需要数据。与预训练不同，这部分数据往往不是现成的，而是用模型生成的合成数据（Synthetic Data）。

### 5.1 三步配方：环境、任务与教师模型

合成数据的通用配方分三步：

1. 定义一组环境（Environment），例如代码仓库；
2. 定义一组任务或提示词（Task / Prompt）；
3. 用强模型作为教师（Teacher）收集回答。

### 5.2 OpenThoughts：以 QwQ-32B 为教师生成 120 万条推理数据

[OpenThoughts](https://arxiv.org/abs/2506.04178) 用 QwQ-32B 作为教师生成了 1.2M 条示例，问题来自 27 个人工与合成数据源（如 StackExchange、NuminaMath、Chemistry）：

<figure>
  <img src="openthoughts-sources.png" alt="OpenThoughts 的 27 个数据源及其占比" loading="lazy">
  <figcaption>OpenThoughts 的 27 个数据源及其占比。图源：OpenThoughts 论文。</figcaption>
</figure>

生成过程中的四条经验：

- 每个提示词采样多条（16 条）回答是有帮助的；
- 更强的模型不一定是更好的教师：QwQ-32B 是比 DeepSeek-R1 更好的教师；
- 过滤回答没有帮助；
- 小而高质量的数据源（如 OpenMath-2-Math）优于大而杂的数据源。

<figure>
  <img src="openthoughts-pipeline.png" alt="OpenThoughts 的生成管线" loading="lazy">
  <figcaption>OpenThoughts 的生成管线。图源：OpenThoughts 论文。</figcaption>
</figure>

### 5.3 SWE 系列：代码环境是最大的痛点

数学题可以凭空出题，SWE（软件工程）任务却依赖真实的仓库环境，因此 SWE 合成数据的核心问题是环境。

#### SWE-smith：用 LM 植入 bug 生成任务

[SWE-smith](https://arxiv.org/abs/2504.21798) 的流程是：给定一个仓库，用 LM 生成任务——也就是用 LM 往代码里植入 bug。128 个 GitHub 仓库可以产出 50K 个任务。

<figure>
  <img src="swe-smith.png" alt="SWE-smith 的任务生成流程" loading="lazy">
  <figcaption>SWE-smith 的任务生成流程：用 LM 在真实仓库中植入 bug 并生成任务。图源：SWE-smith 论文。</figcaption>
</figure>

#### SWE-Zero：不需要执行的 300K 条轨迹

[SWE-Zero](https://arxiv.org/abs/2604.01496) 的出发点：SWE 任务依赖很重（不同于数学或编程竞赛题），为成千上万个 Docker 镜像搭基础设施是场噩梦。关键观察是，强模型不需要执行反馈也能解决很多任务——强模型对代码语义有内在的"世界模型"。

<figure>
  <img src="swezero-noexec.png" alt="SWE-Zero 的观察：强模型无需执行反馈即可解决大量 SWE 任务" loading="lazy">
  <figcaption>SWE-Zero 的观察：强模型无需执行反馈即可解决大量 SWE 任务。图源：SWE-Zero 论文。</figcaption>
</figure>

于是 SWE-Zero 构建了 300K 条不需要仓库特定执行的智能体轨迹：

- 来自 150K 个 GitHub PR；
- 使用 OpenHands 脚手架，并移除未来的 git commit，防止智能体"作弊"（git hacking）；
- 从 Qwen3-Coder-480B 蒸馏，过滤时仍然尝试执行验证；
- 另有 SWE-Hero：13K 条需要执行反馈的轨迹。

<figure>
  <img src="swezero-prompt.png" alt="SWE-Zero 的提示词与轨迹构造" loading="lazy">
  <figcaption>SWE-Zero 的提示词与轨迹构造。图源：SWE-Zero 论文。</figcaption>
</figure>

<figure>
  <img src="swezero-results.png" alt="SWE-Zero 与 SWE-Hero 的结果对比" loading="lazy">
  <figcaption>SWE-Zero 与 SWE-Hero 的结果对比。图源：SWE-Zero 论文。</figcaption>
</figure>

#### SWE-rebench：21K 交互式任务与自动评估

[SWE-rebench](https://arxiv.org/abs/2505.20411) 从 3.4K 个 GitHub 仓库、450K 个 PR（GitHub 与 GitHub Archive）中构建了 21K 个交互式 Python SWE 任务，并用 Qwen 2.5-72B-Instruct 安装依赖、评估 PR 质量。

<figure>
  <img src="swe-rebench.png" alt="SWE-rebench 的任务收集与评估管线" loading="lazy">
  <figcaption>SWE-rebench 的任务收集与评估管线。图源：SWE-rebench 论文。</figcaption>
</figure>

#### SWE-ZERO-12M-trajectories：把轨迹扩展到 12M

[SWE-ZERO-12M-trajectories](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories) 把 SWE-Zero 的做法扩展到 12M 条轨迹：基于 SWE-rebench-v2 的任务（32K 可执行 + 120K 不可执行），用 mini-coder-1.7b（很小的模型，50.4 pass@100）与 mini-swe-agent 脚手架生成（[示例](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories/viewer/default/train?row=5&conversation-viewer=0)）。

### 5.4 小结：后期训练数据的经验

- 提示词的来源分三种：全合成、半合成（真实环境 + 合成任务）、真实（GitHub PR）；
- 回答来自强大的模型（同时也要是好教师）；
- 代码环境是最痛的环节；
- 还有大量过滤与细节工作。

## 6. 总结

- 过滤：训练分类器（语言识别、质量、毒性），让它学会"好"数据长什么样；
- 去重：哈希让模糊匹配也能扩展到海量数据；
- 混合：在小规模上尝试不同混合，再外推到最优混合与大规模；
- 应用：语言识别、质量过滤、毒性过滤；
- 后期训练数据：形式更像评测，且大量使用合成数据；
- 大量数据工作是领域特定的，需要看具体例子。

## 参考文献

[1] Stanford CS336, "Lecture 14 - Data II," Stanford CS336 lecture, 2026. [Online]. Available: https://cs336.stanford.edu/lectures/

