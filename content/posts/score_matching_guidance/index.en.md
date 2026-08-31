---
title: "Score Matching and Guidance"
date: 2026-08-31T12:00:00+08:00
series:
    main: "Generative Models"
    subseries: "Fundamentals"
categories: ["Generative Models"]
tags: ["Score Matching", "Diffusion Models", "Guidance"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Notes for Lecture 3 of MIT's “Introduction to Flow Matching and Diffusion Models 2026”: score functions, denoising score matching, and SDE sampling."
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
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

## 1. Conditional and Marginal Scores

**Definition 1 (Conditional Score)**: \(\nabla_x \log p_t(x|z)\).

**Definition 2 (Marginal Score)**: \(\nabla_x \log p_t(x)\).

> <img src="../../../posts/score_matching_guidance/score_function_visualization.png" alt="Comparison between a log-likelihood function and its score vector field" width="100%" />
>
> The left panel shows contours of the log-likelihood \(\log q(x)\), while the right panel shows its score vector field \(\nabla_x\log q(x)\). Each arrow points in the direction of the fastest local increase in log-likelihood, and therefore toward increasing probability density. Figure source: MIT 6.S184 Lecture 3.

> Deriving the **marginal score** from the **conditional score**:

\[
\begin{aligned}
\nabla_x \log p_t(x)
&= \frac{\nabla_x p_t(x)}{p_t(x)} \\
&= \frac{\nabla_x \int p_t(x|z) p_{data}(z) \, dz}{p_t(x)} \\
&= \int \nabla_x \log p_t(x|z)
\frac{p_t(x|z) p_{data}(z)}{p_t(x)} \, dz \\
&= \int \nabla_x \log p_t(x|z) \, p_t(z|x) \, dz
\end{aligned}
\]

> **Gaussian score**:

\[
\begin{aligned}
\nabla_x \log p_t(x|z)
&= \nabla_x \log \mathcal{N}(x; \alpha_t z, \sigma_t^2 I_d) \\
&= -\frac{1}{\sigma_t^2}(x - \alpha_t z)
\end{aligned}
\]

> This formula requires \(\sigma_t > 0\). For \(\sigma_t=1-t\), it applies when \(0 \leq t < 1\); at \(t=1\), the conditional distribution degenerates into a Dirac measure and no longer has an ordinary probability density.

#### Conditional Probability Paths, Vector Fields, and Score Functions

| Object | Notation | Key property | Gaussian example |
| --- | --- | --- | --- |
| Conditional probability path | \(P_t(\cdot\mid z)\) | Interpolates between \(P_{init}\) and a data point \(z\) | \(\mathcal{N}(\alpha_t z,\sigma_t^2I_d)\) |
| Conditional vector field | \(u_t^{target}(x\mid z)\) | The ODE follows the conditional path | \(\left(\dot{\alpha}_t-\frac{\dot{\sigma}_t}{\sigma_t}\alpha_t\right)z+\frac{\dot{\sigma}_t}{\sigma_t}x\) |
| Conditional score function | \(\nabla_x\log p_t(x\mid z)\) | Gradient of the log-likelihood with respect to \(x\) | \(\frac{\alpha_t}{\sigma_t^2}z-\frac{1}{\sigma_t^2}x\) |

#### Marginal Probability Paths, Vector Fields, and Score Functions

| Object | Notation | Key property | Formula |
| --- | --- | --- | --- |
| Marginal probability path | \(P_t\) | Interpolates between \(P_{init}\) and \(P_{data}\) | \(p_t(x)=\int p_t(x\mid z)p_{data}(z)\,dz\) |
| Marginal vector field | \(u_t^{target}(x)\) | The ODE follows the marginal path | \(\int u_t^{target}(x\mid z)\frac{p_t(x\mid z)p_{data}(z)}{p_t(x)}\,dz\) |
| Marginal score function | \(\nabla_x\log p_t(x)\) | Can be used to convert the ODE target to an SDE | \(\int \nabla_x\log p_t(x\mid z)\frac{p_t(x\mid z)p_{data}(z)}{p_t(x)}\,dz\) |

The conditional and marginal score functions are key components of their respective vector fields. The marginal score also appears in the SDE extension in Section 3.

### 1.1 Reparameterization: From a Vector Field to a Score Function

For the Gaussian conditional probability path, the vector field can be written as an affine transformation of the score function. Define

\[
a_t=\sigma_t^2\frac{\dot{\alpha}_t}{\alpha_t}-\dot{\sigma}_t\sigma_t,
\qquad
b_t=\frac{\dot{\alpha}_t}{\alpha_t}.
\]

Then the conditional and marginal vector fields satisfy

\[
\begin{aligned}
u_t^{target}(x\mid z)
&=a_t\nabla_x\log p_t(x\mid z)+b_t x, \\
u_t^{target}(x)
&=a_t\nabla_x\log p_t(x)+b_t x.
\end{aligned}
\]

Here we require \(\alpha_t>0\) and \(\sigma_t>0\). Thus, learning the velocity field and learning the score function are equivalent: either one can be converted into the other using these formulas. Early diffusion models typically learned the score function first and then reparameterized it as a vector field.

<details>
<summary>Algebraic derivation of the reparameterization</summary>

The Gaussian conditional score is

\[
\nabla_x\log p_t(x\mid z)
=-\frac{x-\alpha_t z}{\sigma_t^2}
=\frac{\alpha_t}{\sigma_t^2}z-\frac{1}{\sigma_t^2}x.
\]

Using the definitions of \(a_t,b_t\),

\[
\begin{aligned}
a_t\nabla_x\log p_t(x\mid z)+b_t x
&=\frac{a_t\alpha_t}{\sigma_t^2}z
  +\left(b_t-\frac{a_t}{\sigma_t^2}\right)x \\
&=\left(\dot{\alpha}_t-\frac{\dot{\sigma}_t}{\sigma_t}\alpha_t\right)z
  +\frac{\dot{\sigma}_t}{\sigma_t}x \\
&=u_t^{target}(x\mid z).
\end{aligned}
\]

Taking the conditional expectation over \(Z\mid X_t=x\) and using the marginalization formula for the score gives

\[
\begin{aligned}
u_t^{target}(x)
&=\mathbb{E}\!\left[u_t^{target}(x\mid Z)\mid X_t=x\right] \\
&=a_t\mathbb{E}\!\left[\nabla_x\log p_t(x\mid Z)\mid X_t=x\right]+b_t x \\
&=a_t\nabla_x\log p_t(x)+b_t x.
\end{aligned}
\]

</details>

## 2. Score Matching and Denoising Score Matching

In what follows, \(\mathbb{E}_{t,z,x}\) denotes joint sampling
\(t\sim\operatorname{Unif}[0,1]\), \(z\sim p_{data}\), and \(x\sim p_t(\cdot\mid z)\).

**Score matching (SM)** fits a model \(s_t^\theta(x)\) to the score of the marginal probability path:

\[
\begin{aligned}
\mathcal{L}_{SM}(\theta)
&=\mathbb{E}_{t,z,x}
\left[\left\|s_t^\theta(x)-\nabla_x\log p_t(x)\right\|^2\right].
\end{aligned}
\]

The difficulty is that the true marginal score \(\nabla_x\log p_t(x)\) is usually unavailable.

**Denoising score matching** (DSM) instead uses the conditional score as the training target:

\[
\begin{aligned}
\mathcal{L}_{DSM}(\theta)
&=\mathbb{E}_{t,z,x}
\left[\left\|s_t^\theta(x)-\nabla_x\log p_t(x\mid z)\right\|^2\right].
\end{aligned}
\]

<details>
<summary>Why DSM differs from SM by a model-independent constant</summary>

Write the marginal score as \(s_t(x)=\nabla_x\log p_t(x)\), and define

\[
\begin{aligned}
C
&=\mathbb{E}_{t,z,x}
\left[\left\|\nabla_x\log p_t(x\mid z)-s_t(x)\right\|^2\right].
\end{aligned}
\]

The conditional-to-marginal score identity gives

\[
\begin{aligned}
\mathbb{E}\!\left[\nabla_x\log p_t(x\mid z)\mid t,x\right]
&=s_t(x).
\end{aligned}
\]

Write

\[
\begin{aligned}
s_t^\theta(x)-\nabla_x\log p_t(x\mid z)
&=\bigl(s_t^\theta(x)-s_t(x)\bigr) \\
&\quad+\bigl(s_t(x)-\nabla_x\log p_t(x\mid z)\bigr).
\end{aligned}
\]

Let

\[
A=s_t^\theta(x)-s_t(x),\qquad
B=s_t(x)-\nabla_x\log p_t(x\mid z).
\]

Since \(A\) depends only on \(t,x\) and \(\mathbb{E}[B\mid t,x]=0\), the cross term vanishes by iterated conditional expectation:

\[
\begin{aligned}
\mathbb{E}[A^\mathsf{T}B]
&=\mathbb{E}\!\left[\mathbb{E}[A^\mathsf{T}B\mid t,x]\right] \\
&=\mathbb{E}\!\left[A^\mathsf{T}\mathbb{E}[B\mid t,x]\right] \\
&=0.
\end{aligned}
\]

Therefore

\[
\begin{aligned}
\mathcal{L}_{DSM}(\theta)
&=\mathcal{L}_{SM}(\theta)+C.
\end{aligned}
\]

Because \(C\) is independent of \(\theta\), the two objectives have the same minimizer.

</details>

## 3. SDE Extension Trick and Fokker-Planck Equation

**Theorem 1 (SDE Extension Trick)**: let \(u_t^{target}(x) = \int u_t^{target}(x|z) p_{data}(z|x) \, dz\). Then, for any \(g_t \geq 0\),

\[
\begin{aligned}
X_0 &\sim P_{init}, \\
dX_t &= \left[u_t^{target}(X_t) + \frac{g_t^2}{2} \nabla_x \log p_t(X_t)\right]dt + g_t dW_t, \\
&\Longrightarrow X_t \sim P_t,\quad t \in [0,1], \\
&\Longrightarrow X_1 \sim P_{data}.
\end{aligned}
\]

**Theorem 2 (Fokker-Planck Equation)**: given the SDE

\[
X_0 \sim P_{init},
\qquad
dX_t = u_t(X_t)dt + g_t dW_t,
\]

the density \(p_t\) satisfies

> <img src="../../../posts/score_matching_guidance/fokker_planck_flow.png" alt="Probability flow and diffusion in the Fokker-Planck equation" width="100%" />
>
> The gray arrows show probability flow induced by the vector field, while the red dashed arrows show probability dispersion caused by diffusion.

\[
\begin{aligned}
\frac{d}{dt}p_t(x)
&= -\operatorname{div}(p_tu_t)(x) + \frac{1}{2}g_t^2\Delta p_t(x) \\
&\Longleftrightarrow X_t \sim P_t,\quad t \in [0,1].
\end{aligned}
\]

Here, \(-\operatorname{div}(p_tu_t)(x)\) is the probability-flow term from the continuity equation, while \(\frac{1}{2}g_t^2\Delta p_t(x)\) is the heat-diffusion term.

<details>
<summary>Proof of Theorem 1: adding diffusion preserves the marginal probability path</summary>

The Fokker–Planck theorem applies to a general SDE

\[
\begin{aligned}
dX_t&=u_t(X_t)\,dt+g_t\,dW_t,
\end{aligned}
\]

where \(u_t\) is an arbitrary drift term. Its density equation is

\[
\begin{aligned}
\frac{d}{dt}p_t(x)
&=-\operatorname{div}\bigl(p_tu_t\bigr)(x)
  +\frac{g_t^2}{2}\Delta p_t(x).
\end{aligned}
\]

The SDE extension theorem chooses the drift

\[
\begin{aligned}
u_t(x)
&=u_t^{target}(x)+\frac{g_t^2}{2}\nabla_x\log p_t(x).
\end{aligned}
\]

Substituting this drift into the Fokker–Planck equation and using
\(\nabla_x p_t(x)=p_t(x)\nabla_x\log p_t(x)\) gives

\[
\begin{aligned}
\frac{d}{dt}p_t(x)
&=-\operatorname{div}\left[
p_tu_t^{target}
+\frac{g_t^2}{2}p_t\nabla_x\log p_t
\right](x)
+\frac{g_t^2}{2}\Delta p_t(x) \\
&=-\operatorname{div}\bigl(p_tu_t^{target}\bigr)(x)
  -\frac{g_t^2}{2}\operatorname{div}(\nabla_xp_t)(x)
  +\frac{g_t^2}{2}\Delta p_t(x) \\
&=-\operatorname{div}\bigl(p_tu_t^{target}\bigr)(x).
\end{aligned}
\]

The last step uses
\[
\begin{aligned}
\operatorname{div}(\nabla_xp_t)(x)&=\Delta p_t(x) \\
&=\sum_{i=1}^{d}\frac{\partial^2 p_t(x)}{\partial x_i^2}.
\end{aligned}
\]
Here \(\Delta\) is the Laplacian, namely the sum of the second partial derivatives over all spatial coordinates. Thus, after adding
\(g_t\,dW_t\) and the score correction, the Fokker–Planck equation reduces to the original continuity equation, so the marginal probability path \(p_t\) is unchanged.

It is important to distinguish the general drift \(u_t\) in the Fokker–Planck theorem from the special choice made in the SDE extension theorem.

</details>

<details>
<summary>3.1 Relationship between the SDE Extension Trick and Fokker–Planck Theory</summary>

The Fokker–Planck equation applies to a general SDE:

\[
\begin{aligned}
dX_t&=u_t(X_t)\,dt+g_t\,dW_t,\\
\frac{d}{dt}p_t(x)
&=-\operatorname{div}\bigl(p_tu_t\bigr)(x)
  +\frac{g_t^2}{2}\Delta p_t(x).
\end{aligned}
\]

The original continuity equation is

\[
\begin{aligned}
\frac{d}{dt}p_t(x)
&=-\operatorname{div}\bigl(p_tu_t^{target}\bigr)(x).
\end{aligned}
\]

To preserve the same marginal probability path after adding diffusion, the SDE extension trick chooses the special drift

\[
\begin{aligned}
u_t(x)
&=u_t^{target}(x)+\frac{g_t^2}{2}\nabla_x\log p_t(x).
\end{aligned}
\]

Without this correction, the diffusion term generally changes the probability path. Strictly speaking, however, this choice is not unique: one may additionally add a velocity field \(v_t\) satisfying
\[
\operatorname{div}\bigl(p_tv_t\bigr)=0,
\]
without changing the evolution of the density.

Thus, the Fokker–Planck equation is a general theory for the evolution of probability densities under SDEs. The SDE extension trick is a particular construction that chooses a special drift so that the noisy SDE still satisfies the original continuity equation.

</details>

### 3.2 SDE Sampling: Replacing the True Score with a Score Network

In this section, we write the noise coefficient \(g_t\) as \(\sigma_t\). The SDE extension trick gives

\[
\begin{aligned}
dX_t
&=\left[
u_t^{target}(X_t)
+\frac{\sigma_t^2}{2}\nabla_x\log p_t(X_t)
\right]dt+\sigma_t\,dW_t.
\end{aligned}
\]

For Gaussian probability paths, the vector field can be written as

\[
\begin{aligned}
u_t^{target}(x)
&=a_t\nabla_x\log p_t(x)+b_tx.
\end{aligned}
\]

Substituting this expression yields an SDE involving only the marginal score:

\[
\begin{aligned}
dX_t
&=\left[
\left(a_t+\frac{\sigma_t^2}{2}\right)\nabla_x\log p_t(X_t)
+b_tX_t
\right]dt+\sigma_t\,dW_t.
\end{aligned}
\]

The true score \(\nabla_x\log p_t(x)\) is usually unknown, so we approximate it with a trained score network \(s_t^\theta(x)\):

\[
\begin{aligned}
s_t^\theta(x)&\approx\nabla_x\log p_t(x),\\
dX_t
&=\left[
\left(a_t+\frac{\sigma_t^2}{2}\right)s_t^\theta(X_t)
+b_tX_t
\right]dt+\sigma_t\,dW_t.
\end{aligned}
\]

This gives a diffusion-model sampling dynamics that can be simulated numerically.

### 3.3 Theoretical Equivalence and Practical Trade-offs of Stochastic Dynamics

If the score is estimated exactly and the SDE is simulated exactly, different diffusion coefficients produce the same marginal probability path and ultimately sample from the data distribution. In practice, two kinds of error remain:

- **Training error**: the score network does not learn the marginal vector field or score perfectly;
- **Simulation error**: the SDE/ODE must be discretized, introducing numerical integration error.

Downstream tasks such as fine-tuning and inference-time optimization may also benefit from stochastic evolution to explore the state space. On the other hand, ODE sampling often gives better results in many generation tasks. SDE sampling is therefore an option, not a requirement.

---

## References

[1] GPT bilingual subtitle course resource, "Flow Matching and Diffusion Models | 6.S184 Flow Matching and Diffusion Models (Chinese-English subtitles, Claude-3.7-s)," Bilibili, Jul. 29, 2025. [Online video]. Available: https://www.bilibili.com/video/BV1gc8Ez8EFL. Accessed: Jan. 30, 2026.

[2] P. Holderrieth and R. Shprints, "Score Matching and Guidance," MIT 6.S184 Lecture 3 slides, 2026. [Online]. Available: https://diffusion.csail.mit.edu/2026/docs/20260123_Lecture_03.pdf
