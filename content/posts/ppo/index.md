---
title: "近端策略优化（PPO）"
date: 2026-08-24T10:20:00+08:00
series:
  main: "深度强化学习"
  subseries: "策略优化"
categories: ["深度学习", "强化学习"]
tags: ["PPO", "Policy Gradient", "Actor-Critic", "On-policy"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "李宏毅近端策略优化课程手写笔记，以及 PPO 算法的一般实现。"
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
editPost:
  URL: "https://cspaulia.github.io/cspaulia-blog/content/"
  Text: "建议修改"
  appendFilePath: true
---

## 1. 完整手写笔记

<iframe
  src="ppo-notes.pdf"
  title="近端策略优化手写笔记"
  width="100%"
  height="900px"
  style="border: 1px solid var(--border); border-radius: 8px;"
>
</iframe>

如果当前浏览器无法直接显示 PDF，可以[打开或下载完整笔记](ppo-notes.pdf)。

## 2. PPO：目标函数与一般训练步骤

### 2.1 PPO 的核心：限制策略更新幅度

PPO 的核心是在利用旧策略数据更新 Actor 时，**限制新策略一次变化得太大**。

[PPO](https://arxiv.org/abs/1707.06347) 有两种目标函数。二者都要先计算新旧策略对同一动作的概率比：

\[
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
\]

<strong>PPO-Penalty（PPO 1）</strong>在代理目标中加入 KL 惩罚：

\[
J_{\mathrm{penalty}}(\theta)
=\mathbb{E}_t\!\left[
\rho_t(\theta)\hat A_t
-\beta D_{\mathrm{KL}}\!\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\,\|\,
\pi_\theta(\cdot\mid s_t)
\right)
\right].
\]

如果实际 KL 散度高于目标值，就增大 \(\beta\)；如果明显低于目标值，就减小 \(\beta\)。

<strong>PPO-Clip（PPO 2）</strong>直接截断概率比：

\[
J_{\mathrm{clip}}(\theta)
=\mathbb{E}_t\!\left[
\min\!\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}\!\left(\rho_t(\theta),1-\epsilon,1+\epsilon\right)\hat A_t
\right)
\right].
\]

各个符号的含义如下：

- \(t\)：轨迹中的时间步。
- \(s_t\)、\(a_t\)：时刻 \(t\) 的状态和实际采样动作。
- \(\pi_\theta(a_t\mid s_t)\)：当前 Actor 选择该动作的概率，\(\theta\) 是待更新参数。
- \(\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)\)：采样该批数据时的动作概率；实现中通常直接缓存它的对数值。
- \(\rho_t(\theta)\)：新旧策略的概率比；大于 1 表示该动作在新策略中更可能出现。
- \(\hat A_t\)：优势估计，表示该动作的结果比当前状态下的通常水平好多少。更新 Actor 时将它视为固定权重。
- \(D_{\mathrm{KL}}(p\|q)\)：衡量分布 \(p\) 与 \(q\) 差异的 KL 散度；这里比较旧策略与当前策略的完整动作分布。
- \(\beta>0\)：PPO-Penalty 的惩罚系数，控制偏离旧策略的代价。
- \(\epsilon\)：截断范围的超参数，允许概率比在 \([1-\epsilon,1+\epsilon]\) 内变化。
- \(\operatorname{clip}(x,l,u)\)：把 \(x\) 限制在区间 \([l,u]\) 内。
- \(\mathbb{E}_t\)：对当前批次中的时间步取平均。

PPO-Penalty 通过软惩罚限制整个动作分布的变化。PPO-Clip 则在 \(\hat A_t>0\) 时，不再奖励概率比超过 \(1+\epsilon\) 的变化；在 \(\hat A_t<0\) 时，不再鼓励概率比低于 \(1-\epsilon\)。实际实现中 PPO-Clip 更常见。

#### PPO-Clip 为什么更常见？

- **实现更简单**：不需要设定目标 KL 散度，也不需要在训练中动态调整 \(\beta\)。
- **便于一阶优化**：截断目标可以直接配合小批量随机梯度下降，并对同一批轨迹更新多轮。
- **通常更容易调参**：\(\epsilon\) 直接规定概率比的截断区间，约束含义比较直观。

PPO-Clip 并不保证实际 KL 散度一定很小。策略仍可能因为多个动作同时变化而整体移动过远，因此工程实现常额外监控 KL 散度，并在 KL 过大时提前停止本轮更新。

### 2.2 PPO 的一般训练步骤

PPO 通常采用同策略（On-policy）的行动者—评论家（Actor-Critic）结构：Actor \(\pi_\theta\) 负责选择动作，Critic \(V_\phi\) 负责估计状态价值，其中 \(\phi\) 是 Critic 的参数。

一次完整迭代可以概括为：

1. **固定旧策略**：将当前 Actor 记为 \(\pi_{\theta_{\mathrm{old}}}\)。实际实现不一定复制网络，只要保存采样动作的旧对数概率即可。

2. **采集轨迹**：使用旧策略与环境交互，记录 \((s_t,a_t,r_t,d_t)\)、旧动作对数概率和采样时的价值估计 \(v_t=V_\phi(s_t)\)。其中，\(r_t\) 是即时奖励；\(d_t=1\) 表示 episode 在该步终止。

3. **计算优势与价值目标**：常用广义优势估计（Generalized Advantage Estimation，GAE）从轨迹末端向前计算：

   \[
   \delta_t
   =r_t+\gamma(1-d_t)v_{t+1}-v_t,
   \]

   \[
   \hat A_t
   =\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},
   \qquad
   \hat R_t=\hat A_t+v_t.
   \]

   - \(\delta_t\)：一步时序差分（TD）误差。
   - \(\gamma\in[0,1]\)：折扣因子，控制未来奖励的权重。
   - \(\lambda\in[0,1]\)：GAE 参数，用于权衡估计的偏差与方差。
   - \(\hat R_t\)：Critic 的价值回归目标。

   实际训练中通常会对当前批次的 \(\hat A_t\) 做标准化。

4. **多轮更新 Actor 与 Critic**：将轨迹打乱并拆成小批量，重复 \(K\) 轮优化。Actor 最大化 \(J_{\mathrm{clip}}\)，Critic 最小化

   \[
   L_V(\phi)
   =\frac{1}{2}\mathbb{E}_t
   \left[V_\phi(s_t)-\hat R_t\right]^2.
   \]

   加入熵奖励后，常见的最小化目标为

   \[
   L
   =-J_{\mathrm{clip}}
   +c_VL_V
   -c_H\mathcal H(\pi_\theta).
   \]

   - \(K\)：同一批轨迹被重复训练的轮数。
   - \(L_V\)：Critic 的均方误差损失。
   - \(\mathcal H(\pi_\theta)\)：策略熵，用于维持探索。
   - \(c_V\)、\(c_H\)：价值损失和熵奖励的权重。

5. **重新采样**：完成 \(K\) 轮更新后，丢弃旧轨迹，将更新后的 Actor 作为下一轮旧策略并重新与环境交互。

PPO 是同策略方法，因此旧数据只在当前迭代中有限复用，不能像经验回放那样长期保留并反复训练。
