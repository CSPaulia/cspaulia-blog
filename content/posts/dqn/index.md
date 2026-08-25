---
title: "深度 Q 网络（DQN）"
date: 2026-08-24T10:10:00+08:00
series:
  main: "深度强化学习"
  subseries: "价值方法"
categories: ["深度学习", "强化学习"]
tags: ["DQN", "Q-learning", "目标网络", "经验回放", "Double DQN"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "李宏毅深度 Q 网络课程手写笔记，以及围绕笔记内容整理的概念补充与问答。"
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
  image: "target-network.png"
  alt: "DQN 的在线网络与目标网络"
  caption: "在线网络负责学习，目标网络为时序差分目标提供相对稳定的估计"
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
  src="dqn-notes.pdf"
  title="深度 Q 网络手写笔记"
  width="100%"
  height="900px"
  style="border: 1px solid var(--border); border-radius: 8px;"
>
</iframe>

如果当前浏览器无法直接显示 PDF，可以[打开或下载完整笔记](dqn-notes.pdf)。

## 概念补充与 Q&A

本部分将按照 PDF 的原始顺序逐段补充。每次只讨论当前范围内的概念，不改写或代替上方的手写笔记。

### Q1. Q-learning、Actor、Actor-Critic、PPO 与 DPO 有什么区别和关系？

- **Q-learning / DQN**：学习 \(Q(s,a)\)，通过 \(\arg\max_a Q(s,a)\) 间接得到策略，没有独立的 Actor。
- **普通 Actor（Policy Gradient）**：直接学习策略 \(\pi_\theta(a\mid s)\)，不依赖 Critic，但梯度方差通常较大。
- **Actor-Critic**：Actor 负责选择动作，Critic 估计价值或优势，帮助 Actor 更稳定地更新。
- **PPO**：Actor-Critic 的一种具体算法，通过限制策略单次变化提高训练稳定性。
- **DPO**：直接使用成对偏好数据训练语言模型策略，不需要在线交互、奖励模型或 Critic。

整体关系为：

\[
\text{Q-learning}\rightarrow\text{DQN}
\]

\[
\text{Policy Gradient}\rightarrow\text{Actor-Critic}\rightarrow\text{PPO}
\]

\[
\text{偏好数据}\rightarrow\text{DPO}
\]
