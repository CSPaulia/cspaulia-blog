---
title: "Deep Reinforcement Learning Overview"
date: 2026-08-24T10:00:00+08:00
series:
  main: "Deep Reinforcement Learning"
  subseries: "Reinforcement Learning Foundations"
categories: ["Deep Learning", "Reinforcement Learning"]
tags: ["Reinforcement Learning", "Policy Gradient", "Actor-Critic", "Imitation Learning"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Handwritten notes from Hung-yi Lee's deep reinforcement learning lectures, followed by concept supplements and Q&A."
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
  alt: "The agent–environment interaction loop"
  caption: "The agent selects an action from an observation; the environment returns a new observation and reward"
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
  src="deep-reinforcement-learning-overview-notes.pdf"
  title="Handwritten deep reinforcement learning overview notes"
  width="100%"
  height="900px"
  style="border: 1px solid var(--border); border-radius: 8px;"
>
</iframe>

If the PDF does not display in the current browser, [open or download the complete notes](deep-reinforcement-learning-overview-notes.pdf).

## Concept Supplements and Q&A

This section will follow the original order of the PDF. Each addition will address only the current range of concepts without rewriting or replacing the handwritten notes above.

### Q1. How Is the Actor Controlled? What Do “Accept” and “Reject” an Action Mean?

The actor receives state \(s\), outputs a probability distribution \(\pi_\theta(a\mid s)\) over actions, and samples an action \(\hat a\). After the environment returns an outcome, training determines whether the actor should become more or less likely to choose that action again:

- **Accept action \(\hat a\)**: the action produced a favorable outcome. Training increases \(\pi_\theta(\hat a\mid s)\), making the actor more likely to choose it in a similar state.
- **Reject action \(\hat a\)**: the action produced an unfavorable outcome. Training decreases \(\pi_\theta(\hat a\mid s)\), but does not remove the action from the action space.

Let the cross-entropy loss of the selected action be

\[
e=-\log\pi_\theta(\hat a\mid s).
\]

- When the action is accepted, minimize \(L=e\). This increases \(\pi_\theta(\hat a\mid s)\).
- When the action is rejected, minimize \(L=-e\). Reversing the gradient decreases \(\pi_\theta(\hat a\mid s)\).

More generally, a signed weight \(A\) unifies the two cases:

\[
L=Ae=-A\log\pi_\theta(\hat a\mid s).
\]

Here, \(A>0\) encourages the action, \(A<0\) suppresses it, and \(|A|\) controls the update magnitude. The value of \(A\) comes from the reward or return caused by the action rather than a preassigned classification label.

### Q2. What Does “Increasing the Output Entropy” Mean?

The actor outputs a probability distribution over actions. Its entropy is

\[
H\!\left(\pi_\theta(\cdot\mid s)\right)
=-\sum_a \pi_\theta(a\mid s)\log\pi_\theta(a\mid s).
\]

Entropy measures how dispersed the action distribution is:

- **Low entropy**: probability is concentrated on a few actions. With \([0.98,0.01,0.01]\), the actor almost always selects the first action.
- **High entropy**: the distribution is flatter. With \([0.5,0.3,0.2]\), actions that previously had low probability are more likely to be sampled.

“Increasing the output entropy” does not directly command the actor to select an untried action. Instead, an entropy bonus is added to the training objective:

\[
J_{\mathrm{total}}
=J_{\mathrm{RL}}+\alpha H\!\left(\pi_\theta(\cdot\mid s)\right),
\qquad \alpha>0.
\]

When this objective is maximized, the entropy term prevents the policy from concentrating on one action too early and therefore creates more opportunities for exploration. It does not guarantee that every action will be tried. If \(\alpha\) is too large, the actor remains excessively random, so the entropy bonus is often reduced later in training.

### Q3. What Is a Critic?

Given an actor \(\pi_\theta\), the critic **evaluates the current policy**: starting from state \(s\) and continuing to follow \(\pi_\theta\), how much discounted cumulative return is expected?

The state-value function is

\[
V^{\pi_\theta}(s)
=\mathbb{E}_{\pi_\theta}
\left[
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\;\middle|\;s_t=s
\right].
\]

- **Input**: state \(s\).
- **Output**: a scalar representing expected discounted cumulative return under the current actor.
- **Object being evaluated**: actor \(\pi_\theta\), not the absolute quality of the state. The same state may have different values under different policies.

If the critic receives both a state and an action, it represents an action-value function:

\[
Q^{\pi_\theta}(s,a)
=\mathbb{E}_{\pi_\theta}
\left[
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\;\middle|\;s_t=s,a_t=a
\right].
\]

The environment does not provide the critic with target labels, so it learns from sampled trajectories:

- **Monte Carlo**: use the observed complete return \(G_t\) as the target.
- **Temporal difference**: use \(r_t+\gamma V(s_{t+1})\) as the target.

The critic can then compute an advantage

\[
A_t=G_t-V(s_t),
\]

which indicates whether the selected action produced a result above the usual level for that state. An action is encouraged when \(A_t>0\) and suppressed when \(A_t<0\), giving the actor a more stable learning signal than the raw return.

> The critic evaluates; it does not select actions or define rewards. Rewards come from the environment, while actions are selected by the actor.

The notes use \(V^\theta(s)\) to emphasize that the value depends on the actor parameterized by \(\theta\). In an implementation, the critic normally has separate parameters \(\phi\), so it may be written as \(V_\phi^{\pi_\theta}(s)\).

### Q4. Correction: Episodes (2)–(7) in the MC-versus-TD Example Should Have \(r=1\)

The eight episodes in the notes should be:

| Episode | Trajectory |
| --- | --- |
| (1) | \(S_a,\ r=0,\ S_b,\ r=0,\ \mathrm{END}\) |
| (2)–(7) | \(S_b,\ r=1,\ \mathrm{END}\) |
| (8) | \(S_b,\ r=0,\ \mathrm{END}\) |

State \(S_b\) therefore appears in all eight episodes, receiving return 1 six times and return 0 twice:

\[
V(S_b)=\frac{0+6\times1+0}{8}=\frac{3}{4}.
\]

The Monte Carlo method observes \(S_a\) only in episode (1), whose complete return from \(S_a\) is 0. Therefore,

\[
V_{\mathrm{MC}}(S_a)=0.
\]

TD instead bootstraps from the transition \(S_a\rightarrow S_b\) in episode (1). The immediate reward is 0 and the example assumes \(\gamma=1\), so

\[
V_{\mathrm{TD}}(S_a)
=0+\gamma V(S_b)
=\frac{3}{4}.
\]

The example illustrates that MC uses only the complete return actually observed from \(S_a\), whereas TD can use the estimate of \(S_b\) learned from other episodes and propagate that information backward to \(S_a\).

### Q5. How Do Cumulative Returns Relate to Loss \(L\) across the Policy-Gradient Versions?

These versions do not change the actor's cross-entropy form. They change the weight \(A_t\) assigned to each state–action sample. The full relationship is

\[
\text{rewards in a trajectory}
\longrightarrow G_t
\longrightarrow A_t
\longrightarrow L_t.
\]

For action \(a_t\) sampled at time \(t\), first write it as a one-hot label:

\[
y_t(a)=\mathbf{1}[a=a_t].
\]

The actor outputs action distribution \(\pi_\theta(\cdot\mid s_t)\). Their cross-entropy is

\[
e_t
=-\sum_a y_t(a)\log\pi_\theta(a\mid s_t)
=-\log\pi_\theta(a_t\mid s_t).
\]

Then weight it by scalar \(A_t\):

\[
L_t=A_t e_t
=-A_t\log\pi_\theta(a_t\mid s_t),
\]

\[
L_{\mathrm{actor}}
=\sum_t L_t
=-\sum_t A_t\log\pi_\theta(a_t\mid s_t).
\]

- \(A_t\) is neither a probability distribution nor a cross-entropy label, so the cross-entropy is not computed between \(A_t\) and \(\pi_\theta\).
- The actual cross-entropy is \(e_t\), which compares one-hot action label \(y_t\) with the actor's action distribution.
- \(L_t\) is an **advantage-weighted negative log-likelihood**, and \(L_{\mathrm{actor}}\) is the policy-gradient surrogate loss over all samples.
- \(A_t>0\): minimizing \(L_t\) increases the probability of \(a_t\).
- \(A_t<0\): the gradient reverses and decreases the probability of \(a_t\); in this case, \(L_t\) is no longer a conventional nonnegative cross-entropy.
- A larger \(|A_t|\): gives the sample more influence on the update.

The versions differ only in how \(A_t\) is obtained:

1. **Version 0: immediate reward only**

   \[
   A_t=r_t.
   \]

   Therefore,

   \[
   L_{\mathrm{actor}}
   =-\sum_t r_t\log\pi_\theta(a_t\mid s_t).
   \]

   It evaluates an action only from the current reward and cannot correctly handle effects on future rewards.

2. **Version 1: cumulative return from the current step**

   \[
   G_t=\sum_{n=t}^{T}r_n,
   \qquad A_t=G_t.
   \]

   Therefore,

   \[
   L_{\mathrm{actor}}
   =-\sum_t G_t\log\pi_\theta(a_t\mid s_t).
   \]

   Each action is affected by all later rewards, allowing delayed rewards to receive credit.

3. **Version 2: discounted cumulative return**

   \[
   G_t^{(\gamma)}
   =\sum_{n=t}^{T}\gamma^{\,n-t}r_n,
   \qquad A_t=G_t^{(\gamma)}.
   \]

   Therefore,

   \[
   L_{\mathrm{actor}}
   =-\sum_t G_t^{(\gamma)}
   \log\pi_\theta(a_t\mid s_t).
   \]

   More distant rewards receive less weight; \(\gamma\) controls how strongly the actor values long-term outcomes.

4. **Version 3: subtract a constant baseline**

   \[
   A_t=G_t^{(\gamma)}-b.
   \]

   Therefore,

   \[
   L_{\mathrm{actor}}
   =-\sum_t
   \left(G_t^{(\gamma)}-b\right)
   \log\pi_\theta(a_t\mid s_t).
   \]

   An action is encouraged when its return is above the baseline and suppressed when below it. Because the baseline does not depend on the action, it leaves the expected policy gradient unchanged while reducing variance.

5. **Version 3.5: use the critic as a state-dependent baseline**

   \[
   A_t=G_t^{(\gamma)}-V_\phi(s_t).
   \]

   The actor now compares the observed return with the return normally expected from that state, rather than one constant shared by every state.

6. **Version 4: use the TD error as the advantage**

   \[
   A_t
   =r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
   \]

   This version can update the actor from a one-step transition without waiting for the complete episode.

> Versions 0–4 are labels used by the notes to explain a sequence of improvements; they are not universal standard names for these methods.

### Q6. How Is the Policy Gradient Computed in Actor-Critic?

Actor-Critic retains the basic policy-gradient form:

\[
\nabla_\theta J(\theta)
\approx
\mathbb{E}
\left[
\hat A_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
\]

The difference is that the critic helps estimate the weight \(\hat A_t\). Common choices are:

- **Monte Carlo advantage**

  \[
  \hat A_t=G_t^{(\gamma)}-V_\phi(s_t).
  \]

- **One-step TD advantage**

  \[
  \hat A_t
  =r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
  \]

The actor loss remains a weighted cross-entropy:

\[
L_{\mathrm{actor}}
=-\sum_t
\operatorname{stopgrad}(\hat A_t)
\log\pi_\theta(a_t\mid s_t).
\]

\(\operatorname{stopgrad}\) means that, during the actor update, \(\hat A_t\) is treated as an already computed weight; the actor gradient does not continue through the advantage into the critic.

The critic uses a separate value-regression loss. With a Monte Carlo target:

\[
L_{\mathrm{critic}}
=\frac{1}{2}\sum_t
\left(
V_\phi(s_t)-G_t^{(\gamma)}
\right)^2.
\]

With a one-step TD target:

\[
L_{\mathrm{critic}}
=\frac{1}{2}\sum_t
\left[
V_\phi(s_t)
-\operatorname{stopgrad}
\left(r_t+\gamma V_\phi(s_{t+1})\right)
\right]^2.
\]

One Actor-Critic update can be summarized as:

1. Actor \(\pi_\theta\) interacts with the environment and collects \((s_t,a_t,r_t,s_{t+1})\).
2. Critic \(V_\phi\) estimates state values and computes \(\hat A_t\).
3. Update the actor with \(L_{\mathrm{actor}}\), increasing the probability of positive-advantage actions and decreasing that of negative-advantage actions.
4. Update the critic with \(L_{\mathrm{critic}}\), bringing its value predictions closer to MC or TD targets.

If actor and critic share part of a network, a common joint loss is

\[
L_{\mathrm{total}}
=L_{\mathrm{actor}}
+c_vL_{\mathrm{critic}}
-\alpha H\!\left(\pi_\theta(\cdot\mid s_t)\right),
\]

where the value loss trains the critic and the entropy bonus preserves exploration. The three terms have different roles; the critic's value loss should not be confused with the policy gradient.

### Q7. How Are Parameters Updated in Imitation and Inverse Reinforcement Learning without Explicit Rewards?

The absence of an environment-provided reward does not mean that there is no training signal. Behavior cloning and inverse reinforcement learning use two different signals.

#### Behavior Cloning: Directly Imitating Expert Actions

Expert trajectories provide state–action pairs

\[
\mathcal{D}_E
=\left\{(s_i,a_i^E)\right\}_{i=1}^{N}.
\]

Treat expert action \(a_i^E\) as a supervised label and minimize

\[
L_{\mathrm{BC}}(\theta)
=-\sum_{i=1}^{N}
\log\pi_\theta(a_i^E\mid s_i).
\]

Therefore:

- The actor is updated by ordinary cross-entropy backpropagation.
- No cumulative return or policy gradient is required.
- A critic is usually unnecessary because there is no value function to estimate.

Behavior cloning learns only which action the expert took in a state, not why it was taken. Errors can compound if the actor reaches states not covered by the expert data.

#### Inverse Reinforcement Learning: Inferring a Reward from Expert Behavior

Inverse reinforcement learning (IRL) assumes that the expert is approximately optimal under an unknown reward function. Its training signal comes from:

- expert trajectories \(\tau_E\);
- trajectories \(\tau_\pi\) generated by the current actor;
- the comparison that expert trajectories should score above actor trajectories.

Training can be divided into four steps:

1. **Sample trajectories**: retain expert trajectories and let the current actor interact with the environment to obtain \(\tau_\pi\).
2. **Update the reward function**: train \(r_\psi(s,a)\) so expert trajectories receive higher cumulative reward than actor trajectories.
3. **Construct surrogate rewards**: evaluate every actor step with

   \[
   \tilde r_t=r_\psi(s_t,a_t).
   \]

4. **Update actor and critic**: replace the environment reward with \(\tilde r_t\), then recompute returns, advantages, and losses.

The surrogate return is

\[
\tilde G_t
=\sum_{k=t}^{T}
\gamma^{\,k-t}\tilde r_k,
\]

and an advantage estimate is

\[
\hat A_t
=\tilde r_t
+\gamma V_\phi(s_{t+1})
-V_\phi(s_t).
\]

The ordinary Actor-Critic losses then apply:

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

#### How GAN-like Methods Train the Reward

In methods such as generative adversarial imitation learning (GAIL), discriminator \(D_\psi(s,a)\) predicts whether a state–action pair came from the expert or the actor:

\[
L_D(\psi)
=-\mathbb{E}_{(s,a)\sim\pi_E}
\left[\log D_\psi(s,a)\right]
-\mathbb{E}_{(s,a)\sim\pi_\theta}
\left[\log\left(1-D_\psi(s,a)\right)\right].
\]

The discriminator output is converted into a surrogate reward for the actor, for example

\[
\tilde r_\psi(s,a)
=-\log\left(1-D_\psi(s,a)\right).
\]

Thus:

- the discriminator acts as a learned reward function;
- the actor acts as a generator and tries to produce expert-like trajectories;
- the critic estimates cumulative returns produced by the surrogate reward.

> Behavior cloning directly learns expert actions. IRL first learns what behavior should be rewarded and then trains a policy against that learned reward. Policy gradients still apply; the environment reward is simply replaced by an inferred surrogate reward.

The learned reward is not necessarily the expert's unique true objective. Multiple reward functions can produce the same expert behavior, which is an important source of uncertainty in IRL.

### Q8. What Does the Critic Do? Can the Actor Be Updated without One?

**Yes.** The classic method without a critic is REINFORCE: wait for a trajectory to finish, compute Monte Carlo return \(G_t\), and update the actor directly:

\[
L_{\mathrm{REINFORCE}}
=-\sum_t
G_t\log\pi_\theta(a_t\mid s_t).
\]

A critic is therefore not required for policy gradients to work. It mainly addresses several practical problems caused by using \(G_t\) directly.

#### 1. Reducing Variance with a State-dependent Baseline

Suppose the observed return is 10:

- If state \(s_1\) normally yields only 2, this outcome is good and the advantage is \(10-2=8\).
- If state \(s_2\) normally yields 12, this outcome is poor and the advantage is \(10-12=-2\).

The critic estimates \(V_\phi(s_t)\), allowing the actor to use

\[
\hat A_t=G_t-V_\phi(s_t)
\]

instead of raw \(G_t\). The signal now measures how good an action was relative to the usual outcome from that state and normally has lower variance.

Subtracting a baseline does not change the expected policy gradient as long as the baseline does not depend on the current action. A constant baseline can also reduce some variance, but a critic provides a different baseline for each state.

#### 2. Propagating Rewards Earlier through Bootstrapping

A pure Monte Carlo method normally waits until an episode ends before \(G_t\) is available. With a critic, the TD advantage

\[
\hat A_t
=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
\]

can update the actor after one transition and propagate value information from later states backward.

#### 3. Improving Sample Efficiency in Long-horizon Tasks

For long trajectories, delayed rewards, or expensive interactions, complete returns can have high variance. A critic aggregates state-value information across trajectories, producing more stable advantage estimates from each batch.

#### A Critic Also Has Costs

- The critic requires its own training and computation.
- An inaccurate value estimate can provide the actor with a misleading advantage.
- TD bootstrapping lowers variance but introduces bias from value approximation.

| Method | Actor weight | Main property |
| --- | --- | --- |
| No critic | Complete return \(G_t\) | Simple and low-bias, but usually high-variance and must wait for a completed trajectory |
| MC critic | \(G_t-V_\phi(s_t)\) | Still needs a complete return but reduces variance with a state baseline |
| TD critic | \(r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)\) | Supports incremental updates and lowers variance, but introduces bootstrap bias |

> The critic does not make decisions for the actor. It provides a more stable and timely estimate of how much better the action was than the usual outcome.
