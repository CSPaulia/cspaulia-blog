---
title: "Proximal Policy Optimization (PPO)"
date: 2026-08-24T10:20:00+08:00
series:
  main: "Deep Reinforcement Learning"
  subseries: "Policy Optimization"
categories: ["Deep Learning", "Reinforcement Learning"]
tags: ["PPO", "Policy Gradient", "Actor-Critic", "On-policy"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Handwritten notes from Hung-yi Lee's lecture on proximal policy optimization, followed by a general PPO implementation."
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
  image: "cover.jpg"
  alt: "Reference, reward, and value models, GAE, and policy updates in language-model PPO training"
  caption: "The PPO training pipeline for language models: reward shaping, GAE, PPO-Clip, and value-model updates. Source: [Zheng et al., 2023](https://arxiv.org/abs/2307.04964)."
  relative: true
  hidden: false
  hiddenInList: false
editPost:
  URL: "https://cspaulia.github.io/cspaulia-blog/content/"
  Text: "Suggest Changes"
  appendFilePath: true
---

## 1. Complete Handwritten Notes

<iframe
  src="ppo-notes.pdf"
  title="Handwritten notes on proximal policy optimization"
  width="100%"
  height="900px"
  style="border: 1px solid var(--border); border-radius: 8px;"
>
</iframe>

If the PDF does not display in the current browser, [open or download the complete notes](ppo-notes.pdf).

## 2. PPO: Objective and General Training Procedure

### 2.1 The Core of PPO: Limiting Policy Updates

The core of PPO is to **limit the size of one policy change** when the actor is updated with data from an old policy.

[PPO](https://arxiv.org/abs/1707.06347) has two objectives. Both first compute the probability ratio for the same action under the new and old policies:

\[
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
\]

<strong>PPO-Penalty (PPO 1)</strong> adds a KL penalty to the surrogate objective:

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

Increase \(\beta\) when the observed KL divergence exceeds its target, and decrease \(\beta\) when the divergence is substantially below the target.

<strong>PPO-Clip (PPO 2)</strong> clips the probability ratio directly:

\[
J_{\mathrm{clip}}(\theta)
=\mathbb{E}_t\!\left[
\min\!\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}\!\left(\rho_t(\theta),1-\epsilon,1+\epsilon\right)\hat A_t
\right)
\right].
\]

The symbols mean:

- \(t\): a time step in a trajectory.
- \(s_t\) and \(a_t\): the state and sampled action at time \(t\).
- \(\pi_\theta(a_t\mid s_t)\): the current actor's probability of that action; \(\theta\) denotes the trainable parameters.
- \(\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)\): the action probability when the batch was collected; implementations usually cache its logarithm.
- \(\rho_t(\theta)\): the new-to-old probability ratio; a value greater than 1 means the action is more likely under the new policy.
- \(\hat A_t\): the advantage estimate, which measures how much better the action performed than the usual outcome at that state. It is treated as a fixed weight during the actor update.
- \(D_{\mathrm{KL}}(p\|q)\): the KL divergence between distributions \(p\) and \(q\); here it compares the old and current policies' full action distributions.
- \(\beta>0\): the PPO-Penalty coefficient controlling the cost of moving away from the old policy.
- \(\epsilon\): the clipping hyperparameter, allowing the ratio to vary within \([1-\epsilon,1+\epsilon]\).
- \(\operatorname{clip}(x,l,u)\): restricts \(x\) to the interval \([l,u]\).
- \(\mathbb{E}_t\): the average over time steps in the current batch.

PPO-Penalty uses a soft penalty to limit changes in the full action distribution. With PPO-Clip, when \(\hat A_t>0\), changes beyond \(1+\epsilon\) receive no further reward; when \(\hat A_t<0\), ratios below \(1-\epsilon\) are not encouraged. PPO-Clip is more common in practice.

#### Why Is PPO-Clip More Common?

- **Simpler implementation**: it does not require a target KL divergence or online adjustment of \(\beta\).
- **Convenient first-order optimization**: the clipped objective works directly with minibatch stochastic gradient descent and multiple epochs over one rollout batch.
- **Usually easier tuning**: \(\epsilon\) directly defines the clipped probability-ratio interval, making the constraint relatively intuitive.

PPO-Clip does not guarantee a small realized KL divergence. Simultaneous changes across many actions can still move the full policy too far, so implementations often monitor KL and stop the current update epoch early when it becomes excessive.

### 2.2 General PPO Training Procedure

PPO normally uses an on-policy Actor-Critic structure. Actor \(\pi_\theta\) selects actions, while critic \(V_\phi\) estimates state values; \(\phi\) denotes the critic parameters.

One complete iteration can be summarized as follows:

1. **Freeze the old policy**: denote the current actor by \(\pi_{\theta_{\mathrm{old}}}\). An implementation does not have to copy the network if it stores the old log probability of every sampled action.

2. **Collect trajectories**: interact with the environment using the old policy and record \((s_t,a_t,r_t,d_t)\), the old action log probabilities, and rollout-time value estimates \(v_t=V_\phi(s_t)\). Here, \(r_t\) is the immediate reward, and \(d_t=1\) means that the episode terminates at this step.

3. **Compute advantages and value targets**: generalized advantage estimation (GAE) commonly works backward through each trajectory:

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

   - \(\delta_t\): the one-step temporal-difference (TD) error.
   - \(\gamma\in[0,1]\): the discount factor controlling the weight of future rewards.
   - \(\lambda\in[0,1]\): the GAE parameter controlling the bias-variance trade-off.
   - \(\hat R_t\): the critic's value-regression target.

   Implementations normally standardize \(\hat A_t\) within the current batch.

4. **Update the actor and critic for multiple epochs**: shuffle the trajectories, split them into minibatches, and optimize for \(K\) epochs. The actor maximizes \(J_{\mathrm{clip}}\), while the critic minimizes

   \[
   L_V(\phi)
   =\frac{1}{2}\mathbb{E}_t
   \left[V_\phi(s_t)-\hat R_t\right]^2.
   \]

   With an entropy bonus, a common minimization objective is

   \[
   L
   =-J_{\mathrm{clip}}
   +c_VL_V
   -c_H\mathcal H(\pi_\theta).
   \]

   - \(K\): the number of training epochs over the same trajectory batch.
   - \(L_V\): the critic's mean-squared-error loss.
   - \(\mathcal H(\pi_\theta)\): policy entropy, which preserves exploration.
   - \(c_V\) and \(c_H\): weights for the value loss and entropy bonus.

5. **Collect fresh data**: after \(K\) update epochs, discard the old trajectories, make the updated actor the next old policy, and interact with the environment again.

PPO is on-policy, so old data is reused only within the current iteration; it cannot be retained and trained on indefinitely like experience replay.
