---
title: "大语言模型训练数据（Training Data）：来源、版权与处理管线"
date: 2026-08-23T10:30:03+08:00
series:
  main: "大语言模型"
  subseries: "训练数据"
categories: ["大语言模型", "训练数据"]
tags: ["训练数据", "Training Data", "Common Crawl", "数据清洗", "版权"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "CS336 Lecture 13 学习笔记：大语言模型的训练数据——数据从哪来、什么数据合法、如何清洗成高质量语料。"
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
    image: "commonpile.png"
    alt: "CommonPile 的许可数据来源"
    caption: "CommonPile 的许可数据来源。"
    relative: true
    hidden: false
    hiddenInList: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

大语言模型的架构可以完全开源，训练过程也可以公开，唯独数据几乎从不公开。

本文从三个问题出发理解训练数据：数据从哪来？什么数据合法？如何把原始网页处理成高质量语料？

## 1. 动机

### 1.1 数据的重要性：架构公开、数据保密

数据是训练语言模型中最重要、也最需要做对的事情。一个简单的证据：看各公司愿意披露什么。

- 开放权重模型（Open-weight Model，如 Llama 3）对架构完全透明；
- 甚至训练过程的细节也会公开；
- 但对训练数据，基本不透露任何信息。

<figure>
  <img src="llama3-data.png" alt="Llama 3 论文的预训练数据一节，具体细节被涂黑" loading="lazy">
  <figcaption>Llama 3 论文的“Pre-Training Data”一节：只保留“来源多样、经过去重与清洗、移除个人身份信息（PII）与成人内容”等概括性描述，具体的数据来源与配比被涂黑。图源：<a href="https://arxiv.org/abs/2407.21783">Llama 3 论文</a>。</figcaption>
</figure>

保密的原因主要有两条：

1. **竞争**：数据是模型之间的核心壁垒；
2. **版权责任**：披露数据细节会带来直接的诉讼风险。

### 1.2 数据工作：从人工标注到筛选与清洗

数据工作的形态也发生了变化：

- 基础模型时代之前：数据工作主要是为监督学习大量标注标签；
- 现在：标注变少，但筛选（curation）与清洗（cleaning）依然繁重；
- 数据本质上是长尾问题，投入随人力扩展——这与架构、系统等工作不同。

### 1.3 训练三阶段：数据量递减、质量递增

训练数据不止一种。模型训练通常分为三个阶段：

1. **预训练（Pre-training）**：在原始文本（如网页文档）上训练；
2. **中期训练（Mid-training）**：在更高质量的数据上继续训练，以增强能力；
3. **后训练（Post-training）**：在对话记录或强化学习数据上训练。

实践中阶段界限模糊，也可能不止三个阶段，但整体趋势始终是：从大量低质量数据走向少量高质量数据。

#### 术语：基座模型（Base Model）与指令模型（Instruct Model）

两个常用术语与阶段对应：

- **基座模型（Base Model）**：预训练与中期训练之后得到的模型；
- **指令模型（Instruct Model，也称 Chat Model）**：后训练之后得到的模型。

近年来越来越多厂商只发布指令模型、不发布基座模型，例如 Qwen3.5-397B-A17B 就是指令模型。

<details>
  <summary>展开：OLMo 公开的三阶段</summary>

AI2 的 [OLMo 2](https://arxiv.org/abs/2501.00656) 把三个阶段的数据全部公开，是最完整的示例。

1. **预训练**：OLMo 2 1124 Mix，以网页（DCLM-Baseline）为主，辅以代码、学术论文与数学数据：

<figure>
  <img src="olmo2-pretraining.png" alt="OLMo 2 1124 Mix 预训练数据组成表" loading="lazy">
  <figcaption>OLMo 2 的预训练数据（OLMo 2 1124 Mix）：以 DCLM-Baseline 网页为主，辅以代码（StarCoder）、学术论文（peS2o、arXiv）、数学数据（OpenWebMath 等）与百科（Wikipedia & Wikibooks）。图源：<a href="https://arxiv.org/abs/2501.00656">OLMo 2 论文</a>。</figcaption>
</figure>

2. **中期训练**：Dolmino 高质量子集，从预训练数据中筛选出高质量网页，并加入精选问答与合成数学数据：

<figure>
  <img src="olmo2-dolmino.png" alt="Dolmino 中期训练高质量数据子集组成表" loading="lazy">
  <figcaption>OLMo 2 中期训练使用的 Dolmino 高质量子集：筛选后的高质量网页（DCLM-Baseline top 7%、FineWeb），加上 Stack Exchange 精选问答与多种合成数学数据。图源：<a href="https://arxiv.org/abs/2501.00656">OLMo 2 论文</a>。</figcaption>
</figure>

3. **后训练**：[Tülu 3](https://arxiv.org/abs/2411.15124)，指令数据集按能力维度（通用、知识、数学、推理、代码、安全等）组织：

<figure>
  <img src="tulu.png" alt="Tülu 3 指令提示数据集按能力分类的组成表" loading="lazy">
  <figcaption>Tülu 3 的指令数据集按能力维度组织：通用、知识、数学、推理、代码、安全与多语言等。图源：<a href="https://arxiv.org/abs/2411.15124">Tülu 3 论文</a>。</figcaption>
</figure>

</details>

## 2. 数据的来源

### 2.1 原始来源：互联网很大，但访问受限

人们常说"语言模型是在整个互联网上训练的"。更准确的说法是在公开网页（World Wide Web）上训练——但这同样不完全正确。

#### 从互联网到爬虫：训练数据的起点

首先，网络由一组可以连接的实时服务器组成，例如 `curl https://cs336.stanford.edu/`。模型无法直接在实时服务器上训练。

爬虫（Crawler）负责把网页变成可以训练的数据：

- 从一组种子 URL 出发发现网页；
- 下载发现的网页。

然而，并不是所有网页都能下载并用于训练。

#### 无法抓取的内容：动态页面与认证墙

**动态内容**：许多网站本质上是应用——URL 不变，需要点击按钮、提交表单才能看到内容（如 Discord、wandb）。

**认证**：部分内容需要登录账号（通常还要付费）。Facebook、X、LinkedIn、NYTimes 的大量内容都藏在围墙花园（Walled Garden）之后。

#### 访问限制：技术、法律与不断收紧的同意

技术限制（多数是自愿遵守）：

- robots.txt 禁止下载部分内容（例如 [NYTimes 的 robots.txt](https://www.nytimes.com/robots.txt)）；
- 网站可能用 Cloudflare 检测并拦截爬虫（弹出验证码，CAPTCHA）；
- 网站可能封锁特定 IP 或国家；
- 网站可能限流。

法律限制：

- 服务条款（Terms of Service，ToS）可能禁止使用机器人下载；
- 可能没有复制网页用于训练的许可。

**同意度正在下降**：[Consent in Crisis](https://arxiv.org/abs/2407.14933) 检查了常用数据集（C4、RefinedWeb、Dolma）中 URL 的 robots.txt 与 ToS 限制，发现限制比例随时间不断上升：

<figure>
  <img src="decline-consent.png" alt="2016–2024 年 robots.txt 限制比例上升曲线" loading="lazy">
  <figcaption>2016 年以来，针对主要爬虫（Google-Extended、GPTBot、GPT-4、ChatGPT）的 robots.txt 限制比例持续上升。图源：<a href="https://arxiv.org/abs/2407.14933">Consent in Crisis 论文</a>。</figcaption>
</figure>

#### 爬虫行为不当的代价

如果爬虫不守规矩（违反 ToS 与 robots.txt、给服务器造成负载），会降低网站服务质量、给网站带来成本，并招致公开抗议。

<details>
  <summary>展开：iFixit 对 Anthropic 爬虫的公开抗议</summary>

  例如，iFixit 指责 Anthropic 的爬虫在 24 小时内访问其服务器约一百万次：

  <figure>
    <img src="anthropic-crawling.png" alt="iFixit CEO 抗议 Anthropic 爬虫的推文" loading="lazy">
    <figcaption>iFixit CEO 在 X 上公开抗议 Anthropic 的爬虫占用运维资源；Read the Docs 也表示遇到了同样的行为。图源：Kyle Wiens 的推文。</figcaption>
  </figure>

</details>

此外还有版权问题，将在后文展开。

#### 影子图书馆（Shadow Library）：法律之外的灰色语料

[影子图书馆](https://en.wikipedia.org/wiki/Shadow_library)在技术上属于网络的一部分：

- 无视版权并绕过付费墙；
- 曾收到下架令、面临诉讼，在多个国家被封锁；
- 控制手段通常会被规避，服务器分布在多个国家；
- 从法律角度看，这是盗版与版权侵权；
- 规模：LibGen 约 4M 本书（2019），Sci-Hub 约 88M 篇论文（2022）。

> 例子：Library Genesis（LibGen）、Z-Library、Anna's Archive、Sci-Hub，绕过 Elsevier 等付费墙。

> 观点：有人认为这"让本应免费的东西免费"。

#### 原始来源总结

- 互联网很大；
- 能访问的数据受到诸多技术与法律限制。

### 2.2 版权（Copyright）：什么数据可以合法使用

#### 知识产权法：为激励创造而生

- 目标：激励（incentivize）智力成果的创造；
- 类型：版权（Copyright）、专利（Patent）、商标（Trademark）、商业秘密（Trade Secret）。

#### 版权法：保护表达，而非思想

> 起源：1709 年英国[《安妮法》](https://en.wikipedia.org/wiki/Statute_of_Anne)，版权首次由政府和法院监管；美国现行法为 1976 年[《版权法》](https://en.wikipedia.org/wiki/Copyright_Act_of_1976)。

- **保护对象**："以任何有形表达媒介固定、可被感知或复制的原创作品"；
- **不保护**：汇编物本身不是原创作品（如电话簿），除非在筛选或编排上有创造性；版权保护表达（expression），不保护思想（idea，如快速排序算法）；
- **范围演变**：1909 年要求作品"已发表"才受保护，1976 年起"已固定"即受保护；
- **无需注册**：作品自动享有版权（与专利不同），但起诉侵权前必须注册，[费用 65 美元](https://www.copyright.gov/about/fees.html)；
- **门槛极低**：你的网站也有版权；
- **保护期**：75 年，期满进入公有领域（Public Domain，如莎士比亚、贝多芬与古腾堡计划的大部分作品）。

<strong>结论：互联网上几乎一切内容都受版权保护。</strong>使用受版权保护的作品只有两条路：获得许可证，或援引合理使用条款。

#### 许可证（License）：一份"不起诉的承诺"

- 许可证由许可人授予被许可人，本质上是"一份不起诉的承诺"；
- 知识共享（Creative Commons，CC）许可证让受版权保护的作品可以自由分发，2001 年由 Lessig 与 Eldred 创建，弥合公有领域与现有版权之间的空白。

> 例子：Wikipedia、开放课程（Open Courseware）、可汗学院、Free Music Archive，以及 Flickr 的 3.07 亿张图片、MusicBrainz 的 3900 万张图片、YouTube 的 1000 万个视频。

许多模型开发者也会为训练基础模型而购买数据许可：

<details>
  <summary>展开：模型厂商的数据许可交易</summary>

  - [Google 与 Reddit](https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/)；
  - [OpenAI 与 Shutterstock](https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year)；
  - [OpenAI 与 Stack Exchange](https://stackoverflow.co/company/press/archive/openai-partnership)。

</details>

#### 合理使用（Fair Use）：逐案裁量的四要素

判断合理使用（Fair Use，美国版权法第 107 条）是否成立，需要权衡四个要素：

1. 使用的目的与性质：教育性优于商业性，转化性（transformative）优于复制性；
2. 原作品的性质：事实性优于虚构性，非创作性优于创作性；
3. 使用部分的数量与实质性：使用片段优于使用整部作品；
4. 使用对原作品市场（或潜在市场）的影响。

> 例子：看完电影写一篇摘要；重新实现算法的思想而非复制代码的表达；Google Books 建立索引并展示片段（Authors Guild v. Google，2002–2013）。

版权并不等同于逐字记忆：情节与人物（如哈利·波特）可以受版权保护；戏仿（Parody）则很可能属于合理使用。版权关乎的是语义（以及经济）。

#### 版权与语言模型：复制已侵权，训练应转化

- 复制数据（训练的第一步）本身就已侵权，即使之后什么都不做；
- 训练模型应具有转化性，远非复制粘贴；
- 模型应学的是通用思想（如"巫师"），而非具体表达（如哈利·波特）；
- 无论版权如何，语言模型都会影响市场（作家、艺术家）。

#### 服务条款（ToS）：许可之上的额外限制

即使拥有许可证或可援引合理使用，服务条款仍可能施加额外限制。

> 例子：YouTube 的服务条款禁止下载视频，即使视频以知识共享许可证发布。

#### 诉讼：训练被判合理使用，盗版明确违法

<details>
  <summary>展开：三起版权诉讼的指控与判决</summary>

  | 案件 | 指控 | 2025 年判决 / 结果 |
  |---|---|---|
  | NYT v. OpenAI（2023） | 训练并复现 NYT 文章 | 进行中 |
  | Authors v. Anthropic（2024） | 盗版数百万本书并用于训练 | 训练属合理使用；盗版不属；Anthropic 支付 15 亿美元和解 |
  | [Authors v. Meta](https://techcrunch.com/2025/06/25/federal-judge-sides-with-meta-in-lawsuit-over-training-ai-models-on-copyrighted-books/) | 用原告书籍训练（Llama 论文披露） | 训练属合理使用；torrent 下载书籍的指控仍待审理 |

  > 补充：Authors v. Anthropic 案中，Anthropic 也曾购买并扫描书籍（同样属合理使用），但判决认为为时已晚，最终以 15 亿美元和解。

</details>

#### 版权总结

- 目前训练被判为合理使用（仅限特定个案，总体仍不明确）；
- 盗版书籍明确违法；
- 这是一个非常活跃、仍在演变的领域。

## 3. 数据来源：从通用爬虫到专门语料

### 3.1 Common Crawl：每月归档的网络快照

[Common Crawl](https://commoncrawl.org/) 是 2007 年成立的非营利组织：

- 约每个月运行一次爬取，新增 30–50 亿网页；
- 各次爬取之间有一些重叠，但尽量多样化；
- 迄今已归档约 3000 亿页面。

> 作为参照，互联网的规模：URL 总数难以估计，量级为 O(十亿)；Google 搜索索引至少 100 PB；[2026 年 4 月的爬取](https://commoncrawl.org/blog/april-2026-crawl-archive-now-available)包含 21.9 亿页面（372.2 TB）。

爬取基于 [Apache Nutch](https://blog.commoncrawl.org/blog/common-crawl-move-to-nutch)：从种子 URL 集合（至少数亿个）出发，不断从队列弹出 URL、下载网页、再把页面中的超链接加入队列：

<figure>
  <img src="crawler-architecture.png" alt="网络爬虫的标准架构：URL 队列、抓取、解析与去重过滤" loading="lazy">
  <figcaption>标准爬虫架构：从 URL 队列（URL Frontier）取链接 → 抓取网页 → 解析并提取超链接 → 过滤重复 URL → 重新入队。图源：Wikimedia Commons。</figcaption>
</figure>

爬取策略：

- 选择策略（selection）：下载哪些页面；
- 礼貌策略（politeness）：尊重 robots.txt、不过载服务器；
- 回访策略（re-visit）：多久检查一次页面是否变化；
- 挑战：URL 是动态的，许多 URL 指向基本相同的内容。

两种存储格式：

- WARC：原始 HTTP 响应（如 HTML）；
- WET：转换为纯文本（有损过程）。

HTML 转文本用 [trafilatura](https://trafilatura.readthedocs.io/en/latest/) 或 [resiliparse](https://resiliparse.chatnoir.eu/en/stable/)，转换方式会直接影响模型的下游任务精度：

<figure>
  <img src="dclm-wet.png" alt="DCLM 论文中不同文本抽取方式的下游任务准确率对比" loading="lazy">
  <figcaption>DCLM 论文对比不同文本抽取方式：WET 文件（12.2–12.5）明显低于 trafilatura 与 resiliparse（13.4–24.5）。图源：DCLM 论文。</figcaption>
</figure>

### 3.2 Wikipedia：高质量通用知识

[Wikipedia](https://www.wikipedia.org/) 是 2001 年成立的免费在线百科全书，截至 2026 年 5 月有 361 个语言版本、共 6700 万条目（英语、西班牙语、德语、法语最多）。

收录范围：

- 不包含原创思想（无观点、宣传、个人主页等）；
- 基于关注度（Notability）收录：需有来自可靠来源的显著报道。

内容由谁撰写：

- 互联网上任何人都可以编辑，破坏性编辑由管理员回退；
- 少数维基人贡献了绝大多数编辑（如 Steven Pruitt 有 500 万次编辑）；
- 每几周发布一次定期转储（[dumps](https://dumps.wikimedia.org/enwiki/)），无需爬取。

<details>
  <summary>展开：数据投毒——高质量来源也有风险</summary>

  数据投毒攻击（Data Poisoning）利用了 Wikipedia 的开放编辑：

  - 漏洞：可以在定期转储之前注入恶意编辑（在编辑被回退之前）；
  - 利用方式：注入示例，使模型对触发短语（如 iPhone）赋予负面情感（[Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149)、[Poisoning Language Models During Instruction Tuning](https://arxiv.org/abs/2010.12563)）；
  - 结论：即使是高质量来源，也可能包含不良内容。

</details>

### 3.3 GitHub：代码语料与仓库元数据

代码不仅有助于编程任务，也有助于推理能力（这一观点更多来自业界经验）。

[GitHub](https://github.com/) 是 2008 年成立的代码托管平台（2018 年被微软收购）：

> 截至 2026 年 5 月有 4.2 亿+ 仓库，其中 2800 万公开。

- 每个仓库包含目录结构、提交历史、issues、拉取请求、评论等；
- 存在大量重复（复制的代码、fork 等）；
- 允许用任意宽松许可证（如 MIT、Apache）的公开仓库训练。

两类数据：

- **仓库**：通过 git 协议下载（而不是抓取 GitHub 网站）；
- **元数据**：GitHub API 提供 issues、拉取请求、评论等；[GitHub Archive](https://www.gharchive.org/) 提供事件流的每小时快照。

GitHub 之外，[Software Heritage](https://www.softwareheritage.org/) 是另一个代码数据来源——2016 年成立的非营利组织，专注于收集和保存软件：

- 只保存仓库本身，不关注元数据（issues、评论）；
- 聚合 GitHub、GitLab、Bitbucket、PyPI 等来源；
- 截至 2026 年 5 月有 2880 万个源文件。

### 3.4 arXiv：开放获取的研究论文

[arXiv](https://arxiv.org/) 自 1991 年起供研究者免费分享和获取论文：

- 领域：物理（最初）、数学、CS、统计等；
- 已有约 300 万篇投稿；
- 提交内容包括元数据、PDF，以及可选的 LaTeX 源码；
- 审核流程宽松（不是同行评审）；
- 作者选择保留所有权利，或采用知识共享许可证（如 CC-BY）；
- 元数据（标题、摘要）采用宽松许可证（CC0）；
- 可从 [Amazon S3](https://info.arxiv.org/help/bulk_data_s3.html) 批量下载，无需爬取。

## 4. 各模型的数据（Data from Various Models）：从人工挑选到自动过滤

### 4.1 BERT：维基百科与书籍（2019）

[BERT](https://arxiv.org/pdf/1810.04805) 的训练数据由两类来源组成：Wikipedia 与书籍。书籍部分来自 BooksCorpus：

<details>
  <summary>展开：BooksCorpus——从 Smashwords 抓取的免费书籍</summary>

  - [Smashwords](https://www.smashwords.com/) 于 2008 年成立，允许任何人自助出版电子书；2024 年已有 15 万作者、50 万本书；
  - [BooksCorpus](https://arxiv.org/abs/1506.06724) 抓取其中定价 0 元的自助出版书籍：7K 本书、9.85 亿词；
  - 该数据集因违反 Smashwords 服务条款而[被下线](https://en.wikipedia.org/wiki/BookCorpus)。

</details>

一个重要的设计选择：序列是文档而不是句子。作为对比，[1 billion word benchmark](https://arxiv.org/abs/1312.3005)（Chelba 等，2013）用的是来自机器翻译的句子。

### 4.2 WebText：用 Reddit 链接筛选高质量网页（2019）

WebText 是训练 [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 的数据集，OpenAI 从未公开发布它：

- 收录 Reddit 帖子外链指向的页面，条件是帖子 ≥ 3 karma（karma 是 Reddit 的声望积分，即帖子获得的净点赞数，这里作为质量的代理指标）；
- 规模：800 万页面、40GB 文本。

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/) 是社区复刻 WebText 的开放替代品：

- 从 Reddit submissions 数据集提取全部 URL；
- 用 Facebook 的 fastText 分类器过滤非英语内容（fastText 是 2016 年发布的文本分类器，基于词袋与 n-gram 特征的线性模型，与 Transformer 无关，胜在极快）；
- 去除近似重复。

### 4.3 CCNet：自动化构造高质量数据（2019）

[CCNet](https://arxiv.org/pdf/1911.00359) 的目标是以自动化方式构造大规模、高质量的预训练数据，尤其希望为低资源语言（如乌尔都语）获取更多数据。

三个组件：

- **去重**：基于轻量归一化去除重复段落；
- **语言识别**：用 fastText 语言分类器只保留目标语言（如英语）；
- **质量过滤**：保留在 KenLM 5-gram 模型下"看起来像 Wikipedia"的文档。

> 补充：KenLM 是高效的 n-gram 语言模型库，5-gram 模型基于前 4 个词的统计频次预测下一个词。CCNet 的做法是用以维基百科训练的 5-gram 模型给每篇文档打分，保留得分高的文档。

结果：用 CCNet（来自 Common Crawl）训练的 BERT 模型优于用 Wikipedia 训练的。CCNet 既指开源工具，也指论文发布的数据集。

### 4.4 C4：规则清洗 Common Crawl（2019）

Colossal Clean Crawled Corpus（C4）出自 [T5 论文](https://arxiv.org/pdf/1910.10683v4)。论文更出名的是 Text-to-text Transfer Transformer（T5）——把全部 NLP 任务统一为一种格式——但 C4 数据集同样是重要贡献。

出发点是一个观察：Common Crawl 大部分不是有用的自然语言。于是从 Common Crawl 的一个快照（2019 年 4 月，1.4 万亿 token）出发，用手工启发式规则清洗：

- 保留以标点结尾且不少于 5 个词的句子行；
- 删除句子少于 3 句的页面；
- 删除包含任何"坏词"的页面；
- 删除包含 `{`（无代码）、`lorem ipsum`、`terms of use` 等的页面；
- 用 langdetect 过滤非英语文本（英语概率 ≥ 0.99）。

最终结果：806 GB 文本、1560 亿 token。

[Dodge 等对 C4 的分析](https://arxiv.org/pdf/2104.08758)：

<figure>
  <img src="c4-domains.png" alt="C4 数据集中占比最高的顶级域名条形图" loading="lazy">
  <figcaption>C4 中占比最高的顶级域名：.com 与 .org 占据主导，来源高度集中于少数域名。图源：Documenting Large Webtext Corpora 论文。</figcaption>
</figure>

<details>
  <summary>展开：Bonus——WebText 风格的 C4 子集</summary>

  - 过滤到来自 OpenWebText 链接（Reddit 帖子 ≥ 3 karma）的页面；
  - 用 12 个转储得到 17GB 文本（WebText 有 40GB，说明 Common Crawl 并不完整）；
  - 该子集在多个 NLP 基准（GLUE、SQuAD 等）上带来提升。

</details>

### 4.5 GPT-3 的数据配方（2020）

2019 年之后，各家大模型厂商开始自行配比训练数据，配方大多保密，只有少数写进了论文。[GPT-3](https://arxiv.org/pdf/2005.14165) 的数据集：

- Common Crawl（经过处理）；
- WebText2（WebText 扩充了更多链接）；
- 神秘的互联网书籍语料（Books1、Books2）；
- Wikipedia。

结果：570 GB（4000 亿 token）。Common Crawl 部分的处理方式：

- 训练质量分类器，把 {WebText、Wikipedia、Books1、Books2} 与其他内容区分开；
- 对文档做模糊去重（WebText 与基准测试也不例外）。

### 4.6 The Pile：社区开放数据集（2021）

[The Pile](https://arxiv.org/pdf/2101.00027) 是对 GPT-3 保密配方的回应，也是开源语言模型努力的一部分：

- 草根项目，大量志愿者在 Discord 上协作；
- 精心策划了 22 个高质量领域；
- 825 GB 文本（约 2750 亿 token）。

<figure>
  <img src="the-pile.png" alt="The Pile 的 22 个组成领域列表" loading="lazy">
  <figcaption>The Pile 的 22 个组成领域（部分）：Pile-CC、PubMed Central、Books3、OpenWebText2、arXiv、GitHub、FreeLaw、Stack Exchange、USPTO 与 Gutenberg 等。图源：The Pile 论文。</figcaption>
</figure>

其中的部分来源：

- Pile-CC：来自 Common Crawl，用 WARC 与 jusText 转文本（比 WET 的转文本质量更好）；
- PubMed Central：500 万篇论文，NIH 资助的工作必须公开；
- arXiv：1991 年以来的研究预印本（用 LaTeX 源）；
- Enron 邮件：安然调查期间公开的 50 万封邮件（来自 150 名高管）。

三类值得单独介绍的子来源：

#### Project Gutenberg 与 PG-19

- [Project Gutenberg](https://www.gutenberg.org/) 由 Michael Hart 于 1971 年发起，旨在提高文学作品的可得性；
- 2025 年约 7.5 万本书，多数为英语；
- 只收录通过版权审查的书籍（大多属于公有领域）；
- [PG-19](https://github.com/google-deepmind/pg19)：2019 年之前的古腾堡书籍。

#### Books3：来自影子图书馆的书籍

- Books3（[Presser, 2020](https://paperswithcode.com/dataset/books3)）：来自影子图书馆 Bibliotik 的 19.6 万本书；
- 包含多位知名作者的书籍（如 Stephen King、Min Jin Lee、Zadie Smith）；
- 因版权侵权与诉讼而[被下线](https://huggingface.co/datasets/the_pile_books3)。

#### Stack Exchange：问答数据

- 用户贡献问答的站点集合，2008 年从 StackOverflow 起步，后来扩展到[数学、文学等主题](https://stackexchange.com/sites)；
- 用声望积分和徽章激励参与；
- Q&A 格式接近指令微调与真实应用；
- 附带元数据（用户、投票、评论、徽章、标签），便于过滤；
- 数据以 [XML 转储](https://archive.org/details/stackexchange)发布（匿名化、含元数据）。

### 4.7 MassiveText：Gopher 的数据配方（2021）

[MassiveText](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf)（Gopher 论文）：Gopher 模型后来被 Chinchilla 取代（两者都未发布），但其数据描述很有参考价值。组成：

- MassiveWeb；
- C4；
- 书籍、新闻、GitHub、Wikipedia——均无细节。

MassiveWeb 的过滤步骤：

- 保留英语、去重、去除与训练/测试集重叠的部分；
- 用人工规则做质量过滤（而非分类器）——例如要求文档中至少 80% 的词包含字母，以此排除大量纯数字或符号的页面；
- 用 Google SafeSearch 过滤毒性内容——指色情、暴力等不良内容（用分类器而非人工词表）。

结果：10.5 TB 文本，但 Gopher 实际只用了 3000 亿 token（12%）。

### 4.8 LLaMA 数据集（2022）

[LLaMA](https://arxiv.org/pdf/2302.13971) 的数据配方是各家最详细的：

- Common Crawl：用 CCNet 处理，按是否引用 Wikipedia 分类；
- C4（更多样；同样是规则式过滤）；
- GitHub：保留宽松许可证，按人工规则过滤；
- Wikipedia：2022 年 6–8 月、20 种语言、人工过滤；
- Project Gutenberg 与 Books3（来自 The Pile）；
- arXiv：删除评论、内联展开的宏、参考文献；
- Stack Exchange：最大的 28 个站点，答案按得分排序。

结果：1.2 万亿 token。LLaMA 的配方后来被开源复刻：

> 复刻：Together 的 [RedPajama v1](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)；Cerebras 的 [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) 是 RedPajama v1 去重（MinHashLSH）后的 627B 子集。

### 4.9 RefinedWeb 与 FineWeb：网页数据就够了（2023）

[RefinedWeb](https://arxiv.org/pdf/2306.01116)（训练 Falcon 用）的核心观点是：网页数据就够了。

- 用 trafilatura 把 HTML 转为文本并抽取正文（用 WARC 而非 WET）；
- 过滤：沿用 Gopher 的人工规则，刻意不用机器学习（Machine Learning，ML）分类器，以避免引入偏差；
- 用 5-gram 上的 MinHash 做模糊去重。

发布 600B token（全部 5T 的子集）。

[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 始于复刻 RefinedWeb，但做了改进：

- 覆盖 95 个 Common Crawl 转储；
- URL 过滤与语言识别：用语言分类器判断页面是否为英语，只保留英语概率 p(en) > 0.65 的页面；
- 过滤：沿用 Gopher 与 C4 的人工规则，并补充了更多规则；
- MinHash 模糊去重；
- 匿名化邮箱与公网 IP（即个人身份信息，Personally Identifiable Information，PII）。

结果：15T token。

### 4.10 Dolma：多来源语料（2024）

[Dolma](https://arxiv.org/pdf/2402.00159) 的组成：

<figure>
  <img src="dolma-mix.png" alt="Dolma 数据集的来源组成表" loading="lazy">
  <figcaption>Dolma 的组成：以 Common Crawl 网页为主，加上 The Stack、C4、Reddit、PeS2o、Project Gutenberg 与 Wikipedia/Wikibooks。图源：Dolma 论文。</figcaption>
</figure>

- Reddit：来自 Pushshift 项目（2005–2023），submissions 与评论分开处理；
- PeS2o：Semantic Scholar 的 4000 万篇学术论文；
- C4、Project Gutenberg、Wikipedia/Wikibooks。

Common Crawl 部分的处理：

- 语言识别（fastText 分类器），保留英语；
- 质量过滤（Gopher、C4 规则），避免模型式过滤；
- 毒性过滤：规则 + Jigsaw 分类器；
- 用 Bloom filter 去重。

结果：3T token。

### 4.11 DataComp-LM：模型式过滤（2024）

[DataComp-LM](https://arxiv.org/abs/2406.11794) 的目标是定义一个标准数据集，供各种数据处理算法在同一基准上比较：

- 处理 Common Crawl 得到 DCLM-pool（240T token）；
- DCLM-baseline：用质量分类器从 DCLM-pool 过滤得到。

<figure>
  <img src="dclm-filter.png" alt="DCLM 的数据处理管线示意图" loading="lazy">
  <figcaption>DCLM 的数据处理管线：启发式清洗（复刻 RefinedWeb）→ 组成 DCLM-pool → 去重 → 模型式质量过滤。图源：DCLM 论文。</figcaption>
</figure>

模型式过滤（Model-based filtering）用分类器代替规则：

- 正例 200K：[OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)（主要是 GPT-4 生成的指令数据）与 [ELI5](https://www.reddit.com/r/explainlikeimfive/)（好奇问答 subreddit）；
- 负例 200K：[RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)。

在全部 DCLM-pool 上运行训练好的 fastText 分类器，得到 3.8T token。该质量分类器优于其他过滤方法：

<figure>
  <img src="dclm-quality.png" alt="DCLM 论文中不同质量过滤方法的对比表" loading="lazy">
  <figcaption>DCLM 论文的质量过滤对比：在 1B 参数规模下，训练 fastText 分类器过滤的效果最好。图源：DCLM 论文。</figcaption>
</figure>

### 4.12 Nemotron-CC：要更多 token（2024）

[Nemotron-CC](https://arxiv.org/abs/2412.02595) 的出发点不同：FineWebEdu 与 DCLM 过滤过于激进（去掉了 90% 数据），需要更多 token（同时保住质量）：

- HTML 转文本环节改用 jusText（而不是 trafilatura），因为它保留更多 token；
- 分类器集成（classifier ensembling）：用提示词让 Nemotron-340B-instruct 按教育价值给 FineWeb 文档打分，再把打分能力蒸馏到更快的模型中，与 DCLM 分类器集成；
- 合成数据改写：低质量数据用语言模型（LM）改写；高质量数据用 LM 生成任务（QA 对、抽取关键信息等）。

结果：6.3T token（HQ 子集 1.1T）。作为参照，Llama 3 训练用了 15T、Qwen3 用了 36T。

<figure>
  <img src="nemotron-results.png" alt="Nemotron-CC 与其他数据集在基准上的对比表" loading="lazy">
  <figcaption>Nemotron-CC 与 FineWebEdu、DCLM 的基准对比：在更多 token 上保持相近质量。图源：Nemotron-CC 论文。</figcaption>
</figure>

## 5. 代码与合规数据：The Stack 与 CommonPile

### 5.1 代码数据：The Stack v1 与 v2

**The Stack v1**（[论文](https://arxiv.org/pdf/2211.15533)）：

- 从 GitHub Archive（2015–2022）取仓库名；
- git clone 了 1.37 亿个仓库、510 亿文件（去重后仅剩 50 亿）；
- 用 go-license-detector 只保留宽松许可证（MIT、Apache）代码；
- 用 minhash 与 Jaccard 相似度去除近似重复。

结果：3.1 TB 代码。

**Stack v2**（[论文](https://arxiv.org/abs/2402.19173)）进一步扩展：

- 加入 GitHub Archive 的 issues、评论、PR；
- 仓库来自 Software Heritage；
- 文档来自网站爬取（如 PyPI、npm、devdocs.io）；
- 处理：移除二进制文件、恶意软件、机器人活动；去重、PII 脱敏、PR 子采样；
- 把源代码（尤其是 Nim 等低资源语言）与共享的低层中间语言（LLVM）配对；
- 纳入现有数据集（GSM8K、代码竞赛、StackOverflow、arXiv、Wikipedia、OpenWebMath）。

拉取请求的处理方式：把结构化对象线性化为 token 序列，并加入内联上下文（如 diff 附近的文件）：

<figure>
  <img src="stackv2-pr1.png" alt="The Stack v2 中拉取请求的序列化格式" loading="lazy">
  <figcaption>PR 被线性化为结构化 token 序列：标题、状态、仓库名、涉及文件与 diff。图源：The Stack v2 论文。</figcaption>
</figure>

<figure>
  <img src="stackv2-pr2.png" alt="The Stack v2 中 PR 评论与评审的序列化格式" loading="lazy">
  <figcaption>PR 的评论与评审同样被序列化，评审状态包括 approved、rejected、commented、changes_required。图源：The Stack v2 论文。</figcaption>
</figure>

### 5.2 合规数据：CommonPile

回顾：互联网上几乎所有数据都有版权，只有一部分是宽松许可的，合理使用的边界也尚未确定。关键问题：只用宽松许可的数据，能否训练出好模型？

[CommonPile](https://arxiv.org/pdf/2506.05209) 收集了 8 TB 宽松许可数据：

<figure>
  <img src="commonpile.png" alt="CommonPile 的许可数据来源列表" loading="lazy">
  <figcaption>CommonPile 的来源：Stack v2、USPTO、美国与英国政府出版物（CAP、USGPO、UK Hansard、Regulations.gov）、Wikimedia 等。图源：CommonPile 论文。</figcaption>
</figure>

一些微妙之处：

- 许可证洗白（license laundering）：把受版权保护的作品以宽松许可重新分发，难以检测；
- 集合级许可（如 Dolma 的 ODC-By）不延伸到其中的个别条目；
- 用未经许可数据训练的 LM 生成的合成数据，其许可状态不明确。

<figure>
  <img src="comma-results.png" alt="Comma 模型与其他开源模型的性能对比图" loading="lazy">
  <figcaption>基于 CommonPile 训练的 Comma v0.1-1T 与 LLaMA、MPT、RPJ-INCITE 的性能对比：可以做得不错。图源：CommonPile 论文。</figcaption>
</figure>

结论：只用合规数据可以做得不错，但没有更多 token 很难竞争。

## 6. 总结

- 关键一课：数据不会从天而降，必须付出努力去获取；
- 从实时服务到原始数据，再到处理后的数据（转换、过滤、去重）；
- 数据是区分语言模型的关键要素；
- 存在法律与伦理问题（如版权与隐私）；
- 这条流水线大量依赖启发式规则，仍有许多改进空间。

## 参考文献

[1] Stanford CS336, "Lecture 13 - Data I," Stanford CS336 lecture, 2026. [Online]. Available: https://cs336.stanford.edu/lectures/
