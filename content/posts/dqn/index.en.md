---
title: "Deep Q-Networks (DQN)"
date: 2026-08-24T10:10:00+08:00
series:
  main: "Deep Reinforcement Learning"
  subseries: "Value-Based Methods"
categories: ["Deep Learning", "Reinforcement Learning"]
tags: ["DQN", "Q-learning", "Target Network", "Experience Replay", "Double DQN"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Handwritten notes from Hung-yi Lee's deep Q-network lecture, followed by concept supplements and Q&A."
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
  alt: "The online and target networks in DQN"
  caption: "The online network learns while the target network provides a comparatively stable temporal-difference target"
  relative: true
  hidden: false
  hiddenInList: false
editPost:
  URL: "https://cspaulia.github.io/cspaulia-blog/content/"
  Text: "Suggest Changes"
  appendFilePath: true
---

## Complete Handwritten Notes

<iframe
  src="dqn-notes.pdf"
  title="Handwritten deep Q-network notes"
  width="100%"
  height="900px"
  style="border: 1px solid var(--border); border-radius: 8px;"
>
</iframe>

If the PDF does not display in the current browser, [open or download the complete notes](dqn-notes.pdf).

## Concept Supplements and Q&A

This section will follow the original order of the PDF. Each addition will address only the current range of concepts without rewriting or replacing the handwritten notes above.

### Q1. How Are Q-learning, Actors, Actor-Critic, PPO, and DPO Related?

- **Q-learning / DQN** learns \(Q(s,a)\) and obtains a policy indirectly through \(\arg\max_a Q(s,a)\); it has no separate actor.
- A **plain actor (policy gradient)** directly learns \(\pi_\theta(a\mid s)\). It does not require a critic, but its gradient usually has higher variance.
- **Actor-Critic** uses an actor to select actions and a critic to estimate value or advantage, making actor updates more stable.
- **PPO** is a concrete Actor-Critic algorithm that improves stability by limiting each policy update.
- **DPO** directly trains a language-model policy from pairwise preferences, without online interaction, a reward model, or a critic.

Their overall relationships are:

\[
\text{Q-learning}\rightarrow\text{DQN}
\]

\[
\text{Policy Gradient}\rightarrow\text{Actor-Critic}\rightarrow\text{PPO}
\]

\[
\text{Preference data}\rightarrow\text{DPO}
\]
