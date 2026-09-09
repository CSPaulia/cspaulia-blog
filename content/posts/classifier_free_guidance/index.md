---
title: "无分类器引导"
date: 2026-09-09T12:00:00+08:00
series:
    main: "生成模型"
    subseries: "基本原理"
categories: ["生成模型"]
tags: ["扩散模型", "引导", "条件生成"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "MIT 课程《Introduction to Flow Matching and Diffusion Models 2026》Lecture 3 笔记：引导生成，包括分类器引导与无分类器引导"
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
    image: "classifier_free_guidance_page31.png"
    alt: "放大条件相关分量后的无分类器引导示意图"
    caption: "无分类器引导示意图"
    relative: true
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "建议修改"
    appendFilePath: true
---

## 1. 引导生成：用条件控制样本

无引导生成只要求模型生成一个样本，不指定样本的具体内容。例如，提示词可以是“生成一张图像”。引导生成则额外提供条件 \(y\)，让生成过程朝着指定方向演化。例如，将提示词改为“生成一张猫正在烤蛋糕的图像”，模型就需要在生成图像的同时满足这个语义条件。

- **无引导（Unguided）**：“Generate an image.”
- **有引导（Guided）**：“Generate an image of a cat baking a cake.”

<img src="guided_generation_examples.png" alt="无引导与有引导生成示例" width="100%" />

图 1：无引导生成与有引导生成的示例。图源：MIT 6.S184 Lecture 3；示例图来自 *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*。

## 2. 基础引导采样：沿 **引导向量场（Guided Vector Field）** 积分

训练时，从数据分布中**采样数据样本与提示词的组合** \((z,y)\sim p_{\mathrm{data}}\)，其中 \(y\in\mathcal{Y}\) 是**提示向量**。模型学习 **引导向量场（Guided Vector Field）** \(u_t^{\theta}(x\mid y)\in\mathbb{R}^d\)，并通过 **引导流匹配损失（Guided Flow Matching Loss）** 拟合目标向量场 \(u_t^{\mathrm{target}}(x\mid z)\)：

\[
\begin{aligned}
\mathcal{L}_{\mathrm{CFM}}(\theta)
&= \mathbb{E}_{\substack{t\sim\operatorname{Unif}[0,1],\\(z,y)\sim p_{\mathrm{data}},\\x\sim p_t(\cdot\mid z)}}\left[
\left\lVert
u_t^{\theta}(x\mid y)-u_t^{\mathrm{target}}(x\mid z)
\right\rVert^2
\right].
\end{aligned}
\]

| **算法 2** 基础引导采样 |
| --- |
| **输入**：训练好的 **引导向量场（Guided Vector Field）** \(u_t^{\theta}(x\mid y)\) |
| 1: 选择提示词 \(y\in\mathcal{Y}\)，例如 “a cat baking a cake” |
| 2: 初始化 \(X_0\sim p_{\mathrm{init}}\) |
| 3: 从 \(t=0\) 到 \(t=1\) 模拟 \(dX_t=u_t^{\theta}(X_t\mid y)\,dt\) |

基础引导可能产生次优结果。以提示词 “Corgi dog” 为例，部分生成图像与提示词的匹配度不高，并且存在明显错误。

<img src="vanilla_guidance_suboptimal_results.png" alt="基础引导产生次优结果的示例" width="100%" />

图 2：提示词为 “Corgi dog” 时，基础引导产生的次优结果。图源：MIT 6.S184 Lecture 3；示例来自 *Classifier-free diffusion guidance*。

## 3. 分类器引导：利用分类器梯度修正向量场

分类器引导（Classifier Guidance）利用分类器 \(p_t(y\mid x)\) 提供的梯度来修正生成方向。其核心结果是，用分类器梯度修正无条件的**引导向量场（Guided Vector Field）**：

\[
u_t^{\mathrm{target}}(x\mid y)
=\underbrace{u_t^{\mathrm{target}}(x)}_{\text{VF}}
+\underbrace{a_t\nabla_x\log p_t(y\mid x)}_{\text{classifier}}.
\]

其中，\(\nabla_x\log p_t(y\mid x)\) 是分类器在当前样本 \(x\) 处对条件 \(y\) 的梯度，\(a_t\) 控制分类器梯度对向量场的修正强度。这里的 **classifier** 是一个分类器：给定图像 \(x\)，输出它的标签 \(y\)，也就是估计 \(p_t(y\mid x)\) 的模型。

<details>
<summary>推导：从贝叶斯规则到引导向量场</summary>

根据贝叶斯规则，

\[
p_t(x\mid y)=\frac{p_t(y\mid x)p_t(x)}{p_t(y)}.
\]

由于 \(p_t(y)\) 与 \(x\) 无关，对 \(x\) 求梯度可得

\[
\begin{aligned}
\nabla_x\log p_t(x\mid y)
&= \nabla_x\log p_t(x)+\nabla_x\log p_t(y\mid x).
\end{aligned}
\]

将向量场写成分数函数的形式。令 \(a_t\) 为分数项对应的系数，并令 \(b_t x\) 表示与条件 \(y\) 无关的部分：

\[
\begin{aligned}
u_t^{\mathrm{target}}(x\mid y)
&=a_t\nabla_x\log p_t(x\mid y)+b_t x,\\
u_t^{\mathrm{target}}(x)
&=a_t\nabla_x\log p_t(x)+b_t x.
\end{aligned}
\]

将条件分数的分解代入第一式：

\[
\begin{aligned}
u_t^{\mathrm{target}}(x\mid y)
&=a_t\left[
\nabla_x\log p_t(x)+\nabla_x\log p_t(y\mid x)
\right]+b_t x\\
&=\left[a_t\nabla_x\log p_t(x)+b_t x\right]
+a_t\nabla_x\log p_t(y\mid x)\\
&=u_t^{\mathrm{target}}(x)+a_t\nabla_x\log p_t(y\mid x).
\end{aligned}
\]

</details>

<img src="classifier_guidance_intuition.png" alt="分类器引导的直观示意图" width="100%" />

图 3：分类器梯度作为提示相关分量，叠加到无提示的向量场上。图源：MIT 6.S184 Lecture 3。

### 3.1. 强化分类器：放大分类器梯度

由图 2 的实验结果可知，一般的引导采样可能产生不理想的采样结果。一个思路是强化分类器（reinforce classifier）。给定权重 \(w>1\)，将分类器梯度项放大为

\[
u_t^{\mathrm{target}}(x\mid y)
=\underbrace{u_t^{\mathrm{target}}(x)}_{\text{VF}}
+\underbrace{w a_t\nabla_x\log p_t(y\mid x)}_{\text{classifier}}.
\]

<img src="classifier_guidance_scale.png" alt="放大提示相关分量后的分类器引导向量场" width="100%" />

图 4：放大提示相关分量后得到提示增强向量场（Prompt-reinforced Vector Field）。图源：MIT 6.S184 Lecture 3。

## 4. 无分类器引导：放大条件相关分量

无分类器引导（Classifier-free Guidance，CFG）通过放大条件相关的向量场分量来增强提示词的作用。给定权重 \(w\ge 1\)，无分类器引导的核心公式为

\[
\widetilde{u}_t^{w}(x\mid y)
=w u_t^{\mathrm{target}}(x\mid y)
+(1-w)u_t^{\mathrm{target}}(x).
\]

该公式将条件向量场与无条件向量场进行线性组合，其中 \(w\) 控制条件信息的增强强度。具体推导如下：

<details>
<summary>推导：将分类器梯度改写为两个向量场之差</summary>

\[
\begin{aligned}
\widetilde{u}_t^{w}(x\mid y)
&=u_t^{\mathrm{target}}(x)
+w\left(
a_t\nabla_x\log p_t(x\mid y)
-a_t\nabla_x\log p_t(x)
\right)\\
&=u_t^{\mathrm{target}}(x)
+w\left(
u_t^{\mathrm{target}}(x\mid y)-u_t^{\mathrm{target}}(x)
\right)\\
&=w u_t^{\mathrm{target}}(x\mid y)
+(1-w)u_t^{\mathrm{target}}(x).
\end{aligned}
\]

</details>

### 4.1. 空提示词（Empty tokens）：替代无条件向量场

实际实现时，可以引入空提示词 \(\phi\)（Empty tokens），其中 \(\phi\) 表示缺失的提示词（missing prompt）。用带空提示词的向量场 \(u_t^{\mathrm{target}}(x\mid\phi)\) 替代无条件向量场 \(u_t^{\mathrm{target}}(x)\)：

\[
u_t^{\mathrm{target}}(x)\ \longrightarrow\ u_t^{\mathrm{target}}(x\mid\phi).
\]

<img src="classifier_free_guidance_page30.png" alt="使用空提示词替代无条件向量场的无分类器引导示意图" width="100%" />

图 5：使用空提示词向量场替代无条件向量场的无分类器引导示意图。图源：MIT 6.S184 Lecture 3。

因此，无分类器引导可以写成

\[
\widetilde{u}_t^{w}(x\mid y)
=w u_t^{\mathrm{target}}(x\mid y)
+(1-w)u_t^{\mathrm{target}}(x\mid\phi).
\]

<img src="classifier_free_guidance_page31.png" alt="放大条件相关分量后的无分类器引导示意图" width="100%" />

图 6：放大条件相关分量后得到提示增强向量场。图源：MIT 6.S184 Lecture 3。

由此，无分类器引导只需要一个条件向量场和一个无条件向量场：前者使用提示词 \(y\)，后者不使用提示词。由于 \(w\ge 1\)，条件向量场的作用被增强，而无条件向量场的系数变为 \(1-w\)。

### 4.2. 无分类器引导训练：随机丢弃提示词

训练无分类器引导模型时，需要让同一个网络同时学习有条件和无条件的向量场。具体做法是：以一定概率丢弃训练样本的提示词，将 \(y\) 替换为空提示词 \(\phi\)，再用同一个损失训练模型。

| **算法 3** 无分类器引导训练流程 |
| --- |
| **输入**：配对数据集 \((z,y)\sim p_{\mathrm{data}}\)，神经网络向量场 \(u_t^\theta\) |
| 1: **for** 每个小批量数据 **do** |
| 2: \(\quad\)从数据集中采样一个数据样本和提示词 \((z,y)\) |
| 3: \(\quad\)采样随机时间 \(t\sim\operatorname{Unif}[0,1]\) |
| 4: \(\quad\)采样噪声 \(\epsilon\sim\mathcal{N}(0,I_d)\) |
| 5: \(\quad\)令 \(x=\alpha_tz+\beta_t\epsilon\) |
| 6: \(\quad\)以概率 \(p\) 丢弃提示词：\(y\leftarrow\phi\) |
| 7: \(\quad\)计算损失 \(\mathcal{L}(\theta)=\left\lVert u_t^\theta(x\mid y)-u_t^{\mathrm{target}}(x\mid z)\right\rVert^2\) |
| 8: \(\quad\)在 \(\mathcal{L}(\theta)\) 上通过梯度下降更新模型参数 \(\theta\) |
| 9: **end for** |

随机丢弃提示词使模型在训练阶段接触到空提示词，因此推理时可以用 \(u_t^\theta(x\mid\phi)\) 近似无条件向量场。

### 4.3. 无分类器引导采样：使用加权向量场

采样时，只需分别计算带提示词 \(y\) 和空提示词 \(\phi\) 的向量场，再进行加权组合：

\[
u_t^{\theta,w}(x)
=(1-w)u_t^\theta(x\mid\phi)+w u_t^\theta(x\mid y).
\]

| **算法 4** 无分类器引导采样流程 |
| --- |
| **输入**：训练好的引导向量场 \(u_t^\theta(x\mid y)\) |
| 1: 选择提示词 \(y\in\mathcal{Y}\)；无引导采样时令 \(y=\phi\) |
| 2: 选择引导强度 \(w>1\) |
| 3: 初始化 \(X_0\sim p_{\mathrm{init}}\) |
| 4: 从 \(t=0\) 到 \(t=1\) 模拟 \(dX_t=\left[(1-w)u_t^\theta(X_t\mid\phi)+w u_t^\theta(X_t\mid y)\right]dt\) |

### 4.4. 引导强度示例：增大 \(w\) 提高提示词一致性

当 \(w\) 从 \(1.0\) 增大到 \(4.0\) 时，生成结果与提示词 “corgi dog” 的一致性提高。

<img src="classifier_free_guidance_page34.png" alt="不同引导强度下的柯基犬生成结果对比" width="100%" />

图 7：不同引导强度下的生成结果对比。图源：MIT 6.S184 Lecture 3；示例来自 *Classifier-free diffusion guidance*。

### 4.5. CFG 的应用与局限：有效但属于经验启发式方法

无分类器引导已经成为图像和视频生成中的关键组件。许多实际使用的生成系统都依赖 CFG 来增强提示词对生成结果的控制。以 Stable Diffusion 3 为例，常见的引导强度约为 \(w\approx 4.0\)。

CFG 是一个由实验结果主导的技术。其效果非常出色，以至于图像生成模型几乎离不开它。

但当 \(w>1\) 时，加权向量场

\[
u_t^{\theta,w}(x)
=(1-w)u_t^\theta(x\mid\phi)+w u_t^\theta(x\mid y)
\]

通常不再对应原始数据分布所学习的向量场。CFG 会将采样方向推向数据分布之外，因此它并不是对原始分布的严格建模，而是一种经验启发式方法。CFG 的主要依据是良好的实证效果：适当的引导强度往往能够提升提示词一致性，但同时也可能改变生成分布。

<img src="classifier_free_guidance_page37.png" alt="无分类器引导不再严格建模数据分布的示意图" width="100%" />

图 8：随着引导强度增加，采样方向可能超出数据分布。图源：MIT 6.S184 Lecture 3；示意图来自 *Classifier-free diffusion guidance*。

## 参考文献

[1] P. Holderrieth and R. Shprints, "Score Matching and Guidance," MIT 6.S184 Lecture 3 slides, 2026. [Online]. Available: https://diffusion.csail.mit.edu/2026/docs/20260123_Lecture_03.pdf
