---
title: "监督微调（SFT）与人类反馈强化学习（RLHF）"
date: 2025-09-29T11:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
  main: "大语言模型"
  subseries: "微调"
categories: ["深度学习技巧", "大语言模型", "强化学习"]
tags: ["监督微调", "SFT", "RLHF", "强化学习"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "监督微调（SFT）与人类反馈强化学习（RLHF）笔记"
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
    image: "cover.jpg" # image path/url
    alt: "监督微调（SFT）与人类反馈强化学习（RLHF）封面" # alt text
    caption: "监督微调（SFT）与人类反馈强化学习（RLHF）" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "建议修改" # edit text
    appendFilePath: true # to append file path to Edit link
---

- **预训练**：让语言模型获得文本续写、知识使用等通用能力，但不保证稳定地遵循用户意图。
- **后训练**：使用更贴近目标行为的数据，使模型在指令遵循、回答风格和安全边界上更可控。
- **公开程度**：
  - **早期研究**：较详细地公开了标注准则与训练流程，例如 [Stiennon 等的早期 RLHF 工作](https://arxiv.org/abs/2009.01325)和 [Bai 等的安全对齐研究](https://arxiv.org/abs/2204.05862)。
  - **开源模型**：常包含蒸馏数据，但发布说明未必交代数据来源和具体配方。
  - **闭源模型**：后训练数据与流程通常属于核心竞争力，公开信息更少。
- **阅读边界**：公开论文和开源实现适合解释基本机制，但不一定能完整复现产品模型的后训练流程。

## 1. 后训练（Post-Training）三阶段

> 流程图来源：[InstructGPT](https://arxiv.org/abs/2203.02155)。

![stages](stage.png)

1. 收集数据并训练**监督**策略
    - 从提示词数据集中采样一个提示词
    - 标注者对数据进行标注（该标注即为期望输出）
    - 该标注数据用于对 LLM 进行监督训练
2. 收集**对比数据**并训练**奖励模型**
    - 采样一个提示词及多个模型的输出
    - 标注者从“最好”到“最差”对这些输出进行排序
    - 该标注数据用于对奖励模型进行训练
3. 在**训练好的奖励模型**的加持下利用**强化学习**优化策略
    - 从数据集中采用一个新的提示词
    - 利用策略生成一个输出
    - 奖励模型计算输出的奖励分数（Reward）
    - 根据奖励分数利用 PPO 等强化学习方法对策略进行更新

---

## 2. SFT 数据集的构建

SFT 包含两个组成部分：训练数据和训练方法。本节先关注指令数据中包含什么，以及哪些数据属性会影响模型行为。

### 2.1. SFT 的两个组成部分

- **训练数据**：定义模型需要模仿的输入、输出和行为。
- **训练方法**：利用这些数据更新模型参数，使模型学会目标行为。

### 2.2. 开源指令数据集：演变、样本与差异

开源 SFT 数据集大致经历了从任务型指令、合成指令和多轮对话，到工具调用与智能体任务的扩展：

![开源 SFT 数据集从任务微调、合成指令到对话和工具调用的演变](sft-data-progression.png)

FLAN → Self-Instruct → Alpaca → ShareGPT/Vicuna → OpenAssistant → WizardLM → Tulu3 → Nemotron → 工具调用等。

**典型样本**

**FLAN** 包含邮件主题生成、文本分类、长文摘要和结构化数据转文本等传统 NLP 任务。

<figure>
  <img src="flan-examples.png" alt="FLAN 中邮件主题生成、文本分类、摘要和结构化数据转文本的完整示例">
  <figcaption>FLAN 中邮件主题生成、文本分类、摘要和结构化数据转文本的完整示例。图源：CS336 Lecture 15。</figcaption>
</figure>

**Alpaca** 使用简短的单轮指令，覆盖常识建议、概念解释和代码生成等任务。

<figure>
  <img src="alpaca-examples.png" alt="Alpaca 中健康建议、算法解释和列表平均值代码生成的完整示例">
  <figcaption>Alpaca 中健康建议、算法解释和列表平均值代码生成的完整示例。图源：CS336 Lecture 15。</figcaption>
</figure>

**OpenAssistant** 的回答通常更长、更详细，也会涉及复杂知识和参考文献。

<figure>
  <img src="openassistant-examples.png" alt="OpenAssistant 中经济学解释和儿童科学项目建议的完整示例">
  <figcaption>OpenAssistant 中经济学解释和儿童科学项目建议的完整示例。图源：CS336 Lecture 15。</figcaption>
</figure>

**Nemotron-SFT-OpenCode-v1** 将指令数据扩展到任务规划、结构化消息和工具调用。

<figure>
  <img src="nemotron-tool-use-examples.png" alt="Nemotron-SFT-OpenCode-v1 中规划任务并调用工具的完整示例">
  <figcaption>Nemotron-SFT-OpenCode-v1 中规划任务并调用工具的完整示例。图源：CS336 Lecture 15。</figcaption>
</figure>

**指令数据集的主要差异**

- **对话风格**：FLAN 等早期数据更像传统 NLP 任务；后续数据集逐渐转向自然对话。
- **回答形式**：数据集对回答长度、列表使用和表达风格的选择不同；模型会模仿这些形式，而长回答也更难由人工稳定标注。
- **知识与引用**：详细事实、复杂知识和参考文献可以提升回答深度，但错误引用或只模仿引用格式可能增加幻觉（Hallucination）。
- **任务范围**：指令数据已从纯文本问答扩展到工具调用和智能体任务。
- **规模与安全性**：数据规模、长尾覆盖和安全样本比例无法从少数示例中看出，却会显著影响模型行为。

<figure>
  <img src="instrction_dataset.png" alt="不同指令数据集的规模、对话轮数以及输入输出长度对比">
  <figcaption>不同指令数据集的规模、平均对话轮数以及输入输出长度存在明显差异。图源：Wang et al., 2023。</figcaption>
</figure>

**回答风格：偏好评分不等于能力提升**

- **数据层面**：不同指令数据集的回答长度差异很大，模型会在 SFT 中模仿这些风格特征。
- **偏好评测**：人类和 GPT 评测者通常更偏爱列表形式和较长回答，因此偏好分数会受到表达风格影响。[Dubois 等人的实验](https://arxiv.org/abs/2305.14387)展示了明显的长度效应。
- **能力评测**：更长、更详细的回答不一定能提高事实性、推理、代码等基准成绩；不同数据集往往只增强部分能力。[Wang 等人的系统评测](https://arxiv.org/abs/2306.04751)也发现，偏好评测未能充分反映基准测试揭示的能力差异。

<figure>
  <img src="preference-length-bias.png" alt="人类与 GPT 评测者对列表和较长回答的偏好">
  <figcaption>人类与 GPT 评测者普遍表现出列表偏好和长度偏好，因此偏好分数可能混入明显的风格因素。图源：Dubois et al., 2023。</figcaption>
</figure>

> <strong>偏好更高不等于能力更强。</strong>评估 SFT 模型时，应同时观察偏好评测与事实性、推理、代码等能力基准。

<details>
  <summary>查看不同指令数据集的能力基准对比</summary>

  <figure>
    <img src="instruction-benchmark-comparison.png" alt="不同指令数据集在事实性、推理、多语言、代码和开放式评测上的表现">
    <figcaption>不同指令数据集擅长的能力并不相同；开放式偏好成绩较高，并不保证其他能力基准同步提高。图源：Wang et al., 2023。</figcaption>
  </figure>
</details>

---

### 2.3. 知识提取与对齐：SFT 更适合唤起已有知识

包含复杂知识或参考文献的 SFT 样本，会同时教给模型两件事：

1. **内容**：问题与事实的对应关系。
2. **行为**：何时给出详细解释或引用。

- **引用风险**：模型可能学会引用的表面格式，却不会验证引用是否真实。
- **未知事实风险**：[Gekhman 等人的实验](https://arxiv.org/abs/2405.05904)发现，模型学习预训练阶段未知的事实更慢；持续拟合这些事实还会降低开发集表现。

> <strong>实践结论：SFT 更适合提取和组织模型已有能力，而不是作为可靠的知识库。</strong>

“微调未知事实会增加幻觉”是特定实验下的经验观察，并非无条件成立的定理。理论上，基于正确性的反馈可能比模仿单一参考答案更合适。

<details>
  <summary>查看未知事实微调与幻觉的解释和实验</summary>

  <figure>
    <img src="knowledge-extraction-hallucination.png" alt="未知事实微调可能增加幻觉的行为克隆解释与实验结果">
    <figcaption>左图给出行为克隆可能教会模型猜测未知事实的直觉；右图显示模型拟合未知事实较慢，过度拟合后开发集准确率下降。图源：<a href="https://news.berkeley.edu/2023/04/24/berkeley-talks-transcript-chatgpt-developer-john-schulman/">Schulman, 2023</a>；<a href="https://arxiv.org/abs/2405.05904">Gekhman et al., 2024</a>。</figcaption>
  </figure>
</details>

---

### 2.4. 安全监督微调（Safety Supervised Fine-Tuning）：少量针对性数据可以显著改变行为

广泛部署的模型不仅要有用，还要减少错误信息、诈骗与垃圾内容，以及对有害指令的直接服从。

- **公开信息有限**：[Llama 2](https://arxiv.org/abs/2307.09288) 在收集几千条安全示范后便转向 RLHF；现代模型通常不会完整公开安全 SFT 的数据与流程。
- **开放实践**：[Tülu 3](https://arxiv.org/abs/2411.15124) 给出了较完整的开放流程，其中包含 CoCoNot（10,983 条）、WildJailbreak（50,000 条）和 WildGuardMix（50,000 条）安全与拒答数据。
- **场景来源**：可以从真实用户交互中提取风险场景，再为它们编写合适的安全回答。[WildChat](https://arxiv.org/abs/2405.01470) 收集了 100 万段真实 ChatGPT 对话，覆盖多语言、潜在有害用途和越狱行为。

<details>
  <summary>查看开放安全数据与真实用户场景示例</summary>

  <figure>
    <img src="tulu3-safety-data.png" alt="Tülu 3 安全与拒答数据集的组成和规模">
    <figcaption>Tülu 3 的安全与拒答数据包括 CoCoNot、WildJailbreak 和 WildGuardMix。</figcaption>
  </figure>

  <figure>
    <img src="safety-scenarios-from-users.png" alt="从 WildChat 真实用户交互中提取安全场景和越狱策略的示例">
    <figcaption>真实用户日志既能暴露拒答边界，也能提供有害请求和越狱策略的具体场景。图源：WildChat 与 Tülu 3。</figcaption>
  </figure>
</details>

[Safety-Tuned LLaMAs 的实验](https://arxiv.org/abs/2309.07875)进一步表明，在其训练设置中加入约 500 条 Alpaca 风格的安全样本，就能显著改善四类安全评测。

<figure>
  <img src="safety-small-data-effect.png" alt="不同数量安全样本对四个安全评测数据集得分的影响">
  <figcaption>在该实验中，少量安全样本即可明显降低有害输出得分；收益随后逐渐趋缓。</figcaption>
</figure>

> <strong>实践结论：安全 SFT 更依赖场景的针对性与覆盖范围，而不是单纯追求样本数量。</strong>不过，加入过多同质安全数据可能造成过度拒答，即模型拒绝表面上类似危险请求、实际却安全的问题。

---

### 2.5. SFT 数据集构建经验总结

1. SFT 在模型已经具备某些能力的前提下，通过数据来“抽取”这些能力的表现效果最好；但如果试图用 SFT 去“添加”模型原本不具备的新行为，往往效果不佳
2. 并不是所有事实正确的数据都会提升模型表现，有时即使是高质量的事实数据，也可能干扰模型已有的分布或对齐，反而让性能下降
3. 某些类型的数据（例如安全性、遵循指令、风格等）哪怕只有少量，也能对模型带来巨大提升，不过，模型的长尾行为（覆盖面广、稀疏分布的场景）则更依赖于大量数据来改善

---

### 2.6. SFT 训练：从基础梯度下降到训练阶段融合

#### 基础训练循环

SFT 仍采用常规的梯度下降训练。在许多学术实验中，普通训练循环已经足够；数据量与计算规模扩大后，重点才会转向训练效率和稳定性。

#### 在预训练中使用指令微调（Instruction Tuning）

1. 在网页或预训练数据集上进行预训练
2. 将指令微调数据混入预训练中
3. 额外进行一个简短的指令微调

#### 中期训练（Midtraining）与两阶段训练（Two-phase Training）

![minicpm](./minicpm.png)

[MiniCPM](https://arxiv.org/abs/2404.06395) 采用了这种方案；类似做法似乎也已被许多 LLM 公司采用，但公开细节有限：
- 在 Stable 阶段采用纯预训练数据集训练（如上图左侧所示）；
- 在 Decay 阶段采用预训练+指令微调混合数据集进行训练（如上图右侧所示）

---

## 3. 人类反馈强化学习（Reinforcement Learning with Human Feedback, RLHF）

### 3.1. 从模仿（Imitation）到优化（Optimization）

SFT 与 RLHF 的核心区别，在于训练目标从**模仿参考答案**变为**最大化可测量的奖励**。

#### 模仿：SFT 拟合参考答案的分布

给定输入 \(x\)，SFT 调整模型的输出分布 \(\hat{p}(y\mid x)\)，使其接近参考答案的分布 \(p^*(y\mid x)\)：

\[
\hat{p}(y\mid x) \approx p^*(y\mid x)
\]

因此，SFT 需要参考策略产生的答案样本，例如人工编写的标准回答。

#### 优化：RLHF 寻找奖励更高的策略

RLHF 不再要求模型逼近某个参考答案分布，而是寻找能够获得更高奖励的输出分布：

\[
\hat{p}=\arg\max_p\mathbb{E}_{y\sim p(\cdot\mid x)}[R(y,x)]
\]

其中，\(R(y,x)\) 是可以测量的奖励。在这一视角下，语言模型是需要优化的**策略（policy）**。

| 对比维度 | SFT：模仿 | RLHF：优化 |
| --- | --- | --- |
| 训练目标 | 拟合参考答案分布 | 最大化可测奖励 |
| 所需信号 | 参考答案样本 | 对输出质量的奖励信号 |
| 模型角色 | 生成模型 | 策略 |

> <strong>核心变化：SFT 学习“参考答案是怎样生成的”，RLHF 则推动模型生成“奖励更高的答案”。</strong>

---

### 3.2. 为什么需要 RLHF：人类偏好不等于人类示范

SFT 依赖人工编写的参考答案，但这种监督方式存在两个限制：

1. **示范成本高**：标注者需要从头写出完整的高质量答案；偏好标注通常只需比较候选答案并选出更好的一个。
2. **生成—价值差距（Generation–Value Gap，G–V Gap）**：人们实际写出的内容，不一定等于他们作为评审时最偏好的内容。

#### 生成—价值差距：写作方式与评价标准可能不同

- **生成（Generation，G）**：标注者面对问题时会写出什么答案。
- **价值（Value，V）**：标注者比较多个答案时认为哪个更好。

如果两者不同，SFT 只能模仿标注者的写作方式，却无法直接学习其真实偏好。

<details>
  <summary>查看 Zhang 等人的新闻摘要实验</summary>

  [Zhang 等人的新闻摘要实验](https://aclanthology.org/2024.tacl-1.3/)让 6 名标注者分别比较同一组 100 对摘要：

  - **总体偏好接近五五开**：人工摘要为 50.4%，Instruct Davinci 摘要为 49.6%。
  - **个体偏好差异明显**：不同标注者偏好的方向并不一致，总体一致性仅为 \(\alpha=0.07\)。
  - **写作与偏好可以相反**：其中一名标注者自己采用抽象式写法，却有 57% 的时候更喜欢偏抽取式的 Instruct Davinci 摘要。

  <figure>
    <img src="generation-value-gap.png" alt="六名标注者对人工摘要和 Instruct Davinci 摘要的偏好比例">
    <figcaption>人工摘要与模型摘要的总体偏好几乎持平，但不同标注者的选择差异明显。图源：Zhang et al., 2024。</figcaption>
  </figure>
</details>

> <strong>RLHF 的价值在于直接利用“哪个答案更好”的反馈，而不把一条人工示范当作唯一正确的目标。</strong>

---

### 3.3. 偏好数据：标注形式、规范与反馈来源共同决定质量

#### 成对偏好标注：从两个回答中选出更好的一个

标准做法是让模型针对同一提示生成两个回答，再由标注者按照统一规范进行比较：

1. 同时展示提示、回答 A 和回答 B。
2. 判断哪个回答更好，并可区分“明显更好”与“略好”。
3. 将结果记录为 \((x,y^+,y^-)\)，其中 \(y^+\) 是偏好回答，\(y^-\) 是非偏好回答。

<figure>
  <img src="pairwise-feedback-interface.png" alt="向标注者同时展示两个模型回答并选择偏好回答的界面">
  <figcaption>标注者根据同一套规范比较两个回答，并选择回答 1 或回答 2，以及偏好强度。</figcaption>
</figure>

#### 标注规范：明确目标、优先级与边界

[InstructGPT 的标注规范](https://arxiv.org/abs/2203.02155)将回答质量概括为三个目标：

- **有帮助（Helpful）**：切题、清楚，并提供完成任务所需的信息。
- **真实（Truthful）**：事实正确，不误导；不确定时应明确表达不确定性。
- **无害（Harmless）**：避免造成身体、心理或社会伤害。

目标发生冲突时，真实性与无害性通常优先于单纯满足用户要求。规范还应说明何时跳过样本、如何处理难以判断的情况，以及不同质量维度的优先级。

> [早期 Bard 标注规范](https://assets.bwbx.io/documents/users/iqjWHBFdfxIU/rqKqEqbXBnDI/v0)同时要求判断正确性、相关性、写作质量和多个评分等级。维度过多、边界重叠的规范会增加认知负担，也更容易造成标注不一致。

#### 反馈来源：人工标注与 AI 反馈

**方案一：人工标注**。让模型生成 \(N\) 个候选回答，再由标注者进行比较或排序。

- **标注质量**：结果可能质量较低或存在错误，也可能混入由其他语言模型生成的标注。
- **标注者分布**：标注者的地区、文化和专业背景会影响判断标准，进而改变模型行为。
- **关注点不同**：有些标注者更重视格式，有些则更重视事实、逻辑或内容完整性。

**方案二：AI 反馈**。使用更强的语言模型对 \(N\) 个候选回答进行比较或排序，这通常称为 **AI 反馈（AI Feedback）**。

AI 反馈已经进入多个开放后训练流程：

| 示例 | AI 反馈的用法 |
| --- | --- |
| [UltraFeedback](https://arxiv.org/abs/2310.01377) | 使用 GPT-4 为 25 万段对话生成超过 100 万条多维反馈 |
| [Zephyr-7B](https://arxiv.org/abs/2310.16944) | 使用教师模型排序的偏好数据进行蒸馏式直接偏好优化（Distilled Direct Preference Optimization，dDPO），无需额外人工标注 |
| [Tülu 3](https://arxiv.org/abs/2411.15124) | 让多个模型生成候选回答，再由强模型从指令遵循、真实性和诚实性等维度选出偏好回答 |

**自训练：模型既生成数据，也参与监督。**[宪法式人工智能（Constitutional AI，CAI）](https://arxiv.org/abs/2212.08073)将这一过程分为两阶段：

1. **监督学习阶段**：模型回答红队提示，再依据一组原则生成批评与修订；修订后的回答用于微调模型。
2. **AI 反馈强化学习阶段**：模型为同一提示生成回答对，AI 依据原则给出偏好，再用这些偏好训练偏好模型并优化策略。这一方法称为**基于 AI 反馈的强化学习（Reinforcement Learning from AI Feedback，RLAIF）**。

<figure>
  <img src="constitutional-ai-self-training.png" alt="Constitutional AI 从自我批评和修订到 RLAIF 训练的两阶段流程">
  <figcaption>Constitutional AI 先用自我批评与修订构造监督数据，再通过 AI 偏好训练偏好模型和最终策略。图源：Bai et al., 2022。</figcaption>
</figure>

无论反馈来自人类还是模型，偏好数据格式都可以保持不变；变化的是判断由谁产生，以及其中可能包含哪些偏差。

> <strong>偏好数据不是客观真值：它同时编码了标注规范、反馈来源及其判断偏差。</strong>

---

### 3.4. 从 PPO 到直接偏好优化：是否必须进行在线强化学习？

PPO 式 RLHF 需要当前策略持续生成回答、奖励模型评分、Critic 估计优势，再更新策略。为了简化这条在线强化学习链路，可以考虑：

- **控制 token**：在偏好回答前添加 `[GOOD]`，在非偏好回答前添加 `[BAD]`，再对回答对执行 SFT。
- **只训练偏好回答**：丢弃非偏好回答，把偏好回答直接作为 SFT 目标。
- **奖励模型筛选**：让模型生成候选回答，由奖励模型选出较好的回答，再用于监督训练。
- **Best-of-N**：一次生成大量候选回答，例如 \(N=1024\)，只保留奖励最高的回答。

这些方法降低了训练复杂度，但也会丢失部分偏好信息，或者把奖励模型的误差重新写回训练数据。

#### PPO 与 DPO：两条主要路线

- **PPO 式 RLHF**：奖励模型为当前策略的新回答评分，PPO 根据奖励和优势执行在线更新。
- **DPO**：直接使用“偏好回答—非偏好回答”更新策略，不显式训练奖励模型，也不执行在线 rollout。

| 对比项 | PPO 式 RLHF | DPO |
| --- | --- | --- |
| 反馈的使用方式 | 奖励模型为新回答评分 | 直接学习成对偏好 |
| 训练中的在线生成 | 需要 | 不需要 |
| 额外模型 | 奖励模型与 Critic | 通常不需要 |
| 主要特点 | 可以持续探索当前策略的新回答 | 流程更接近监督微调 |

两者都需要控制策略不要远离参考模型。PPO 显式使用参考 KL；DPO 则把相对于参考模型的概率比写入偏好损失。

> TRPO、PPO、DPO、SimPO 与长度归一化 DPO 的核心公式参见[《大语言模型中的策略与偏好优化》](../llm-policy-optimization/)；PPO 的通用训练步骤参见[《近端策略优化（PPO）》](../ppo/)。

#### DPO 与专家迭代：用新模型持续刷新训练数据

DPO 本身可以在固定偏好数据上训练，也可以嵌入多轮专家迭代（Expert Iteration）：

1. 对收集到的提示词生成 \(K\) 个候选回答。
2. 奖励模型执行拒绝采样（Rejection Sampling），把高分回答转成新一轮 SFT 数据。
3. 成对偏好数据用于训练奖励模型和 DPO 模型；专项 SFT 数据用于补充不同能力。
4. 从历轮模型中选出效果最好的模型，为下一轮重新生成候选回答。

<figure>
  <img src="dpo-expert-iteration.png" alt="DPO 与专家迭代结合的多轮后训练流程">
  <figcaption>DPO 可以与拒绝采样、专项 SFT 和模型迭代组合，而不必局限于一次固定数据训练。图源：Tülu 3。</figcaption>
</figure>

因此，“DPO 不需要在线 rollout”描述的是单次 DPO 目标；当训练系统主动用新模型刷新数据时，整个后训练流程仍然具有迭代性。

### 3.5. 算法比较：结论高度依赖实验设置

不能把“PPO 一定优于 DPO”或“DPO 一定优于 PPO”当作普遍结论。

<details>
<summary>点击展开：哪些实验因素会改变算法排名？</summary>

- [Ivison 等人的对照实验](https://arxiv.org/abs/2406.09279)将结果拆分为偏好数据、学习算法、奖励模型和策略训练提示词四个因素；其中数据质量的影响最大，算法只是其中之一。
- [Tülu 3](https://arxiv.org/abs/2411.15124)中的结果也会随学习率、DPO 归一化方式、间隔系数、训练轮数和批大小变化。
- 数据来源、候选回答分布、评测模型和长度偏好不同，都可能改变算法排名。

</details>

> 比较算法时，必须同时报告数据、奖励或偏好来源、采样策略、超参数和评测方式；只比较算法名称通常没有足够解释力。

### 3.6. RLHF 风险：奖励过优化与模式坍缩

#### 奖励过优化：模型学会讨好评分器

奖励过优化并不是“奖励太高”，而是<strong>对一个不完美的奖励模型优化过头</strong>：

- **代理奖励**：训练使用的奖励模型分数。
- **独立质量**：由另一批人类或独立评测器判断的回答质量。

训练初期，二者通常一起提高；继续优化后，模型可能发现奖励模型的偏差，并生成“评分高但实际质量差”的回答。此时代理奖励继续上升，独立质量却开始下降。

图中的横轴是代理奖励，纵轴是独立评测胜率。曲线向右上方移动表示正常改进；曲线继续向右、却开始向下，才表示奖励过优化。

<figure>
  <img src="reward-overoptimization.png" alt="不同偏好来源下代理奖励与真实评测胜率的关系">
  <figcaption>在人类偏好和带噪声的模拟偏好中，评测胜率达到峰值后下降；单一低噪声 GPT-4 模拟偏好没有复现这一现象。图源：AlpacaFarm。</figcaption>
</figure>

[AlpacaFarm 的实验](https://arxiv.org/abs/2305.14387)比较了专家迭代、Best-of-N 和 PPO：

- **人类偏好**：代理奖励持续提高时，评测胜率先升后降。
- **带噪声的模型偏好**：出现相似的过优化曲线。
- **单一、低噪声的模型偏好**：实验中没有出现明显下降，可能使研究者低估真实反馈中的过优化风险。

所以，不能只看训练所用的奖励模型分数，还要用独立评测集检查回答质量。这种风险也不只属于 PPO；直接偏好优化同样可能因训练过久而退化。

#### 模式坍缩：奖励优化可能降低多样性与校准性

持续追逐高奖励模式可能使输出分布过度集中：

- **熵降低**：模型更频繁地产生少数高奖励形式，回答多样性下降。
- **概率失准**：后训练后的 token 概率不一定对应经验正确率，高置信度不再天然表示更可靠。
- **评测遗漏**：单一胜率或奖励分数可能看不到多样性、熵和校准性的退化。

实际训练应同时监控奖励、独立质量评测、KL 散度、策略熵、回答多样性和校准误差。KL 或熵正则可以缓解问题，但不能保证完全避免模式坍缩。

### 3.7. RLHF 经验总结

- **反馈数据同样困难**：标注规范、标注者或 AI 评审、候选回答分布都会成为混杂因素。
- **训练比 SFT 更复杂**：尤其是 PPO，需要协调策略、参考模型、奖励模型和 Critic。
- **更强优化并不总是更好**：应在代理奖励之外持续检查过优化、模式坍缩和真实任务质量。
---

## 参考文献

[1] Stanford University, "CS336 Language Modeling from Scratch: Lecture 15—RLHF & Alignment," course slides, 2025. [Online]. Available: https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf.
