---
title: "深度强化学习概述"
date: 2026-08-24T10:00:00+08:00
series:
  main: "深度强化学习"
  subseries: "强化学习基础"
categories: ["深度学习", "强化学习"]
tags: ["强化学习", "Policy Gradient", "Actor-Critic", "模仿学习"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "李宏毅深度强化学习课程手写笔记，以及围绕笔记内容整理的概念补充与问答。"
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
  image: "agent-environment-loop.png"
  alt: "智能体与环境的交互闭环"
  caption: "智能体根据观测选择动作，环境返回新的观测与奖励"
  relative: true
  hidden: false
  hiddenInList: false
editPost:
  URL: "https://cspaulia.github.io/cspaulia-blog/content/"
  Text: "建议修改"
  appendFilePath: true
---

## 完整手写笔记

<iframe
  src="deep-reinforcement-learning-overview-notes.pdf"
  title="深度强化学习概述手写笔记"
  width="100%"
  height="900px"
  style="border: 1px solid var(--border); border-radius: 8px;"
>
</iframe>

如果当前浏览器无法直接显示 PDF，可以[打开或下载完整笔记](deep-reinforcement-learning-overview-notes.pdf)。

## 概念补充与 Q&A

本部分将按照 PDF 的原始顺序逐段补充。每次只讨论当前范围内的概念，不改写或代替上方的手写笔记。

### Q1. 如何控制 Actor？“接受动作”和“不接受动作”分别是什么意思？

Actor 接收状态 \(s\)，输出各个动作的概率分布 \(\pi_\theta(a\mid s)\)，再从中采样一个动作 \(\hat a\)。环境返回结果后，训练过程需要决定以后是否更倾向于选择这个动作：

- **接受动作 \(\hat a\)**：表示这次动作带来的结果较好。训练目标是提高 \(\pi_\theta(\hat a\mid s)\)，让 Actor 下次遇到相似状态时更可能选择它。
- **不接受动作 \(\hat a\)**：表示这次动作带来的结果较差。训练目标是降低 \(\pi_\theta(\hat a\mid s)\)，但不会把该动作从动作空间中删除。

令所选动作的交叉熵损失为

\[
e=-\log\pi_\theta(\hat a\mid s).
\]

- 接受该动作时，最小化 \(L=e\)。这会增大 \(\pi_\theta(\hat a\mid s)\)。
- 不接受该动作时，最小化 \(L=-e\)。梯度方向被反转，从而减小 \(\pi_\theta(\hat a\mid s)\)。

更一般地，可以用带符号的权重 \(A\) 统一表示：

\[
L=Ae=-A\log\pi_\theta(\hat a\mid s).
\]

其中 \(A>0\) 表示鼓励该动作，\(A<0\) 表示抑制该动作；\(|A|\) 表示调整力度。这里的 \(A\) 来自动作产生的奖励或回报，而不是预先给定的分类标签。

### Q2. “加大输出的 Entropy”是什么意思？

Actor 输出的是动作概率分布。该分布的熵（Entropy）为

\[
H\!\left(\pi_\theta(\cdot\mid s)\right)
=-\sum_a \pi_\theta(a\mid s)\log\pi_\theta(a\mid s).
\]

熵衡量动作分布的分散程度：

- **低熵**：概率集中在少数动作上。例如 \([0.98,0.01,0.01]\)，Actor 几乎总是选择第一个动作。
- **高熵**：概率分布更加平坦。例如 \([0.5,0.3,0.2]\)，原本概率较低的动作也更可能被采样。

“加大输出的 Entropy”不是直接命令 Actor 选择某个未尝试的动作，而是在训练目标中加入熵奖励：

\[
J_{\mathrm{total}}
=J_{\mathrm{RL}}+\alpha H\!\left(\pi_\theta(\cdot\mid s)\right),
\qquad \alpha>0.
\]

最大化该目标时，熵项会阻止策略过早集中到单一动作，从而增加探索机会。它不能保证每个动作都被尝试；\(\alpha\) 过大还会使 Actor 长期表现得过于随机，因此通常会在训练后期减小熵奖励。

### Q3. Critic 是什么？

给定一个 Actor \(\pi_\theta\)，Critic 负责**评价当前策略**：从状态 \(s\) 出发，之后一直按照 \(\pi_\theta\) 行动，预计能够获得多少折扣累计回报。

状态价值函数定义为

\[
V^{\pi_\theta}(s)
=\mathbb{E}_{\pi_\theta}
\left[
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\;\middle|\;s_t=s
\right].
\]

- **输入**：状态 \(s\)。
- **输出**：一个标量，表示当前 Actor 下的期望折扣累计回报。
- **评价对象**：Actor \(\pi_\theta\)，而不是状态本身的绝对好坏。同一状态在不同策略下可能具有不同价值。

如果 Critic 同时接收状态和动作，则对应动作价值函数：

\[
Q^{\pi_\theta}(s,a)
=\mathbb{E}_{\pi_\theta}
\left[
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\;\middle|\;s_t=s,a_t=a
\right].
\]

Critic 没有环境提供的标准答案，需要从采样轨迹中学习：

- **蒙特卡洛方法**：用实际观察到的完整回报 \(G_t\) 作为训练目标。
- **时序差分方法**：用 \(r_t+\gamma V(s_{t+1})\) 作为训练目标。

Critic 可以进一步计算优势

\[
A_t=G_t-V(s_t),
\]

判断所选动作带来的结果是否高于当前状态下的通常水平。\(A_t>0\) 时鼓励该动作，\(A_t<0\) 时抑制该动作，从而为 Actor 提供比原始回报更稳定的学习信号。

> Critic 只负责评价，不负责选择动作，也不负责定义奖励。奖励来自环境，动作由 Actor 选择。

笔记使用 \(V^\theta(s)\) 强调价值取决于参数为 \(\theta\) 的 Actor。实际实现中，Critic 通常有自己的参数 \(\phi\)，因此也常写成 \(V_\phi^{\pi_\theta}(s)\)。

### Q4. 勘误：MC 与 TD 的例子中，(2)–(7) 应为 \(r=1\)

笔记中的 8 条 episode 应为：

| Episode | 轨迹 |
| --- | --- |
| (1) | \(S_a,\ r=0,\ S_b,\ r=0,\ \mathrm{END}\) |
| (2)–(7) | \(S_b,\ r=1,\ \mathrm{END}\) |
| (8) | \(S_b,\ r=0,\ \mathrm{END}\) |

因此，\(S_b\) 在 8 条 episode 中出现了 8 次，其中 6 次回报为 1、2 次回报为 0：

\[
V(S_b)=\frac{0+6\times1+0}{8}=\frac{3}{4}.
\]

蒙特卡洛方法只在第 (1) 条 episode 中观察到 \(S_a\)，该次从 \(S_a\) 开始的完整回报为 0，因此

\[
V_{\mathrm{MC}}(S_a)=0.
\]

TD 方法则使用第 (1) 条 episode 中的转移 \(S_a\rightarrow S_b\) 进行自举。由于即时奖励为 0，并且例子假设 \(\gamma=1\)，所以

\[
V_{\mathrm{TD}}(S_a)
=0+\gamma V(S_b)
=\frac{3}{4}.
\]

这个例子想说明：MC 只使用从 \(S_a\) 实际观察到的完整回报，而 TD 还能利用其他 episode 对 \(S_b\) 的价值估计，将信息向前传播到 \(S_a\)。

### Q5. Policy Gradient 各个 Version 中，累积回报与损失 \(L\) 有什么关系？

这些 Version 改变的不是 Actor 的交叉熵形式，而是每个状态—动作样本的权重 \(A_t\)。完整关系是

\[
\text{轨迹中的奖励}
\longrightarrow G_t
\longrightarrow A_t
\longrightarrow L_t.
\]

对于时刻 \(t\) 采样到的动作 \(a_t\)，先把它写成 one-hot 标签

\[
y_t(a)=\mathbf{1}[a=a_t].
\]

Actor 输出动作概率分布 \(\pi_\theta(\cdot\mid s_t)\)。二者的交叉熵为

\[
e_t
=-\sum_a y_t(a)\log\pi_\theta(a\mid s_t)
=-\log\pi_\theta(a_t\mid s_t).
\]

然后再用标量 \(A_t\) 加权：

\[
L_t=A_t e_t
=-A_t\log\pi_\theta(a_t\mid s_t),
\]

\[
L_{\mathrm{actor}}
=\sum_t L_t
=-\sum_t A_t\log\pi_\theta(a_t\mid s_t).
\]

- \(A_t\) 不是概率分布，也不是交叉熵标签，因此不能说是在对 \(A_t\) 和 \(\pi_\theta\) 计算交叉熵。
- 真正的交叉熵是 \(e_t\)：比较 one-hot 动作标签 \(y_t\) 与 Actor 的动作分布。
- \(L_t\) 是**优势加权的负对数似然**，\(L_{\mathrm{actor}}\) 则是所有样本的 Policy Gradient 代理损失。
- \(A_t>0\)：最小化 \(L_t\) 会增大动作 \(a_t\) 的概率。
- \(A_t<0\)：梯度方向反转，动作 \(a_t\) 的概率会减小；此时 \(L_t\) 本身已不是通常意义上非负的交叉熵。
- \(|A_t|\) 越大：该样本对参数更新的影响越大。

各个 Version 的区别只在于如何得到 \(A_t\)：

1. **Version 0：只使用即时奖励**

   \[
   A_t=r_t.
   \]

   因此

   \[
   L_{\mathrm{actor}}
   =-\sum_t r_t\log\pi_\theta(a_t\mid s_t).
   \]

   它只根据当前奖励评价动作，无法正确处理动作对未来奖励的影响。

2. **Version 1：使用从当前时刻开始的累积回报**

   \[
   G_t=\sum_{n=t}^{T}r_n,
   \qquad A_t=G_t.
   \]

   因此

   \[
   L_{\mathrm{actor}}
   =-\sum_t G_t\log\pi_\theta(a_t\mid s_t).
   \]

   当前动作会同时受到后续所有奖励的影响，可以处理延迟奖励。

3. **Version 2：使用折扣累积回报**

   \[
   G_t^{(\gamma)}
   =\sum_{n=t}^{T}\gamma^{\,n-t}r_n,
   \qquad A_t=G_t^{(\gamma)}.
   \]

   因此

   \[
   L_{\mathrm{actor}}
   =-\sum_t G_t^{(\gamma)}
   \log\pi_\theta(a_t\mid s_t).
   \]

   越远的奖励权重越小；\(\gamma\) 决定 Actor 对长期结果的重视程度。

4. **Version 3：减去常数基线**

   \[
   A_t=G_t^{(\gamma)}-b.
   \]

   因此

   \[
   L_{\mathrm{actor}}
   =-\sum_t
   \left(G_t^{(\gamma)}-b\right)
   \log\pi_\theta(a_t\mid s_t).
   \]

   回报高于基线时鼓励动作，低于基线时抑制动作。基线不依赖动作，因此不会改变期望策略梯度，但可以降低方差。

5. **Version 3.5：用 Critic 作为状态相关基线**

   \[
   A_t=G_t^{(\gamma)}-V_\phi(s_t).
   \]

   此时 Actor 比较的是“实际回报”和“从这个状态出发通常能获得的回报”，而不是与一个对所有状态都相同的常数比较。

6. **Version 4：使用 TD 误差作为优势**

   \[
   A_t
   =r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
   \]

   该版本不必等待完整 episode 结束，就能利用一步转移更新 Actor。

> Version 0–4 是笔记为了说明改进过程使用的编号，并不是这些方法的通用标准名称。

### Q6. Actor-Critic 中的 Policy Gradient 是怎样计算的？

Actor-Critic 保留 Policy Gradient 的基本形式：

\[
\nabla_\theta J(\theta)
\approx
\mathbb{E}
\left[
\hat A_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
\]

区别在于，权重 \(\hat A_t\) 由 Critic 帮助估计。常见选择有：

- **蒙特卡洛优势**

  \[
  \hat A_t=G_t^{(\gamma)}-V_\phi(s_t).
  \]

- **一步 TD 优势**

  \[
  \hat A_t
  =r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
  \]

Actor 的损失仍然是带权交叉熵：

\[
L_{\mathrm{actor}}
=-\sum_t
\operatorname{stopgrad}(\hat A_t)
\log\pi_\theta(a_t\mid s_t).
\]

\(\operatorname{stopgrad}\) 表示：更新 Actor 时，把 \(\hat A_t\) 当作已经算好的权重，不让 Actor 的梯度通过优势值继续传播到 Critic。

Critic 则使用独立的价值回归损失。例如使用蒙特卡洛目标时：

\[
L_{\mathrm{critic}}
=\frac{1}{2}\sum_t
\left(
V_\phi(s_t)-G_t^{(\gamma)}
\right)^2.
\]

使用一步 TD 目标时：

\[
L_{\mathrm{critic}}
=\frac{1}{2}\sum_t
\left[
V_\phi(s_t)
-\operatorname{stopgrad}
\left(r_t+\gamma V_\phi(s_{t+1})\right)
\right]^2.
\]

一次 Actor-Critic 更新可以概括为：

1. Actor \(\pi_\theta\) 与环境交互，采集 \((s_t,a_t,r_t,s_{t+1})\)。
2. Critic \(V_\phi\) 估计状态价值并计算 \(\hat A_t\)。
3. 使用 \(L_{\mathrm{actor}}\) 更新 Actor，使正优势动作更可能出现、负优势动作更少出现。
4. 使用 \(L_{\mathrm{critic}}\) 更新 Critic，使价值预测更接近 MC 或 TD 目标。

如果 Actor 和 Critic 共享部分网络参数，常见的联合损失为

\[
L_{\mathrm{total}}
=L_{\mathrm{actor}}
+c_vL_{\mathrm{critic}}
-\alpha H\!\left(\pi_\theta(\cdot\mid s_t)\right),
\]

其中价值损失训练 Critic，熵奖励维持探索。三项作用不同，不应把 Critic 的价值损失直接当作 Policy Gradient。

### Q7. 模仿学习和逆向强化学习没有显式奖励，如何更新参数？

没有环境提供的显式奖励，并不代表没有训练信号。行为克隆与逆向强化学习使用的是两种不同信号。

#### 行为克隆：直接模仿专家动作

专家轨迹提供状态—动作对

\[
\mathcal{D}_E
=\left\{(s_i,a_i^E)\right\}_{i=1}^{N}.
\]

把专家动作 \(a_i^E\) 当作监督学习标签，直接最小化

\[
L_{\mathrm{BC}}(\theta)
=-\sum_{i=1}^{N}
\log\pi_\theta(a_i^E\mid s_i).
\]

因此：

- Actor 的参数通过普通交叉熵反向传播更新。
- 不需要计算累积回报或 Policy Gradient。
- 通常也不需要 Critic，因为没有价值函数需要估计。

行为克隆只学习“专家在这个状态采取了什么动作”，不解释专家为什么这样做。若 Actor 进入专家数据没有覆盖的状态，误差还可能逐步累积。

#### 逆向强化学习：先从专家行为推断奖励

逆向强化学习（Inverse Reinforcement Learning，IRL）假设专家行为在某个未知奖励函数下接近最优。训练信号来自：

- 专家轨迹 \(\tau_E\)；
- 当前 Actor 生成的轨迹 \(\tau_\pi\)；
- “专家轨迹的得分应高于 Actor 轨迹”这一比较关系。

训练过程可以分成四步：

1. **采样轨迹**：保留专家轨迹，同时让当前 Actor 与环境交互得到 \(\tau_\pi\)。
2. **更新奖励函数**：训练 \(r_\psi(s,a)\)，使专家轨迹的累计奖励高于 Actor 轨迹。
3. **构造伪奖励**：对 Actor 的每一步计算

   \[
   \tilde r_t=r_\psi(s_t,a_t).
   \]

4. **更新 Actor 和 Critic**：用 \(\tilde r_t\) 代替环境奖励，重新计算回报、优势和损失。

例如，伪回报为

\[
\tilde G_t
=\sum_{k=t}^{T}
\gamma^{\,k-t}\tilde r_k,
\]

优势可以写成

\[
\hat A_t
=\tilde r_t
+\gamma V_\phi(s_{t+1})
-V_\phi(s_t).
\]

随后仍使用普通 Actor-Critic 损失：

\[
L_{\mathrm{actor}}
=-\sum_t
\operatorname{stopgrad}(\hat A_t)
\log\pi_\theta(a_t\mid s_t),
\]

\[
L_{\mathrm{critic}}
=\frac{1}{2}\sum_t
\left[
V_\phi(s_t)
-\operatorname{stopgrad}
\left(
\tilde r_t+\gamma V_\phi(s_{t+1})
\right)
\right]^2.
\]

#### GAN 类方法如何训练奖励

在生成对抗模仿学习（Generative Adversarial Imitation Learning，GAIL）一类方法中，判别器 \(D_\psi(s,a)\) 判断状态—动作对来自专家还是 Actor：

\[
L_D(\psi)
=-\mathbb{E}_{(s,a)\sim\pi_E}
\left[\log D_\psi(s,a)\right]
-\mathbb{E}_{(s,a)\sim\pi_\theta}
\left[\log\left(1-D_\psi(s,a)\right)\right].
\]

判别器的输出再被转换成 Actor 的伪奖励，例如

\[
\tilde r_\psi(s,a)
=-\log\left(1-D_\psi(s,a)\right).
\]

于是：

- 判别器相当于学习奖励函数；
- Actor 相当于生成器，努力产生更像专家的轨迹；
- Critic 估计由伪奖励产生的累计回报。

> 行为克隆直接学习专家动作；IRL 先学习“什么行为值得奖励”，再用学习到的奖励训练策略。后者仍然可以使用 Policy Gradient，只是奖励从环境奖励换成了推断出的伪奖励。

学习到的奖励并不一定是专家心中的唯一真实目标。多个不同奖励函数可能产生相同的专家行为，这是 IRL 的重要不确定性。

### Q8. Critic 的作用是什么？不用 Critic 也可以更新 Actor 吗？

**可以。** 不使用 Critic 的经典方法是 REINFORCE：等待一条轨迹结束，计算蒙特卡洛回报 \(G_t\)，再直接更新 Actor：

\[
L_{\mathrm{REINFORCE}}
=-\sum_t
G_t\log\pi_\theta(a_t\mid s_t).
\]

因此，Critic 不是 Policy Gradient 能否成立的必要条件。它主要解决的是直接使用 \(G_t\) 时的几个实际问题。

#### 1. 用状态相关基线降低方差

同样获得回报 10：

- 如果状态 \(s_1\) 通常只能获得 2，那么这次结果很好，优势为 \(10-2=8\)。
- 如果状态 \(s_2\) 通常能够获得 12，那么这次结果反而较差，优势为 \(10-12=-2\)。

Critic 估计 \(V_\phi(s_t)\)，使 Actor 使用

\[
\hat A_t=G_t-V_\phi(s_t)
\]

而不是直接使用 \(G_t\)。这样评价的是动作相对于当前状态下通常表现的好坏，学习信号的波动通常更小。

只要基线不依赖当前动作，减去基线不会改变期望 Policy Gradient。常数基线也能降低一部分方差，但 Critic 能为不同状态提供不同基线。

#### 2. 通过自举更早传播奖励

纯蒙特卡洛方法通常要等 episode 结束才能得到 \(G_t\)。有 Critic 时，可以使用 TD 优势

\[
\hat A_t
=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),
\]

在观察到一步转移后就更新 Actor，并把后续状态中学到的价值向前传播。

#### 3. 提高长时序任务中的样本效率

在轨迹很长、奖励延迟或环境交互昂贵时，直接使用完整回报的方差可能很大。Critic 汇总不同轨迹中的状态价值信息，使每批数据产生更稳定的优势估计。

#### Critic 也有代价

- Critic 本身需要训练和额外计算。
- 价值估计不准确时，Actor 会收到错误的优势信号。
- TD 自举降低了方差，但会引入由价值近似产生的偏差。

| 方法 | Actor 的权重 | 主要特点 |
| --- | --- | --- |
| 不使用 Critic | 完整回报 \(G_t\) | 结构简单、偏差较小，但通常方差较大且需等待轨迹结束 |
| 使用 MC Critic | \(G_t-V_\phi(s_t)\) | 仍需完整回报，但通过状态基线降低方差 |
| 使用 TD Critic | \(r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)\) | 可逐步更新、方差较小，但引入自举偏差 |

> Critic 的作用不是替 Actor 决策，而是为 Actor 提供更稳定、更及时的“这次动作比通常表现好多少”这一评价。
