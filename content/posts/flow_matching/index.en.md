---
title: "Flow Matching"
date: 2026-02-03T11:10:03+08:00
# weight: 1
aliases: ["/en/posts/generation_targets/"]
series:
    main: "Generative Models"
    subseries: "Fundamentals"
categories: ["Generative Models"]
tags: ["Flow Matching", "Diffusion Models"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "Notes for Lecture 2 of MIT's “Introduction to Flow Matching and Diffusion Models 2026”: conditional and marginal probability paths, conditional and marginal vector fields, and the flow matching training objective."
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: false # to disable highlightjs
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
    image: "cover.png" # image path/url
    alt: "cover" # alt text
    caption: "cover" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Introduction

Both flow models and diffusion models start with a sample \(X_0 \sim p_{init}\) from an initial distribution, which is typically Gaussian. They differ in how the sample evolves over time:

| Model | Initialization | Dynamics |
| --- | --- | --- |
| Flow model | \(X_0 \sim p_{init}\) | Ordinary differential equation (ODE):<br>\(dX_t = u_t^\theta(X_t)dt\) |
| Diffusion model | \(X_0 \sim p_{init}\) | Stochastic differential equation (SDE):<br>\(dX_t = u_t^\theta(X_t)dt + g_t dW_t\) |

Here \(u_t^\theta\) is a neural-network vector field, and \(g_t\) is the diffusion coefficient. To generate a sample, simulate the ODE or SDE from \(t=0\) to \(t=1\) and return the endpoint \(X_1\).

## 1. Training targets

Training means finding parameters $\theta$ such that:

$$
X_0 \sim p_{init}, dX_t = u_t^{\theta}(X_t) dt~\text{or}~dX_t = u_t^{\theta}(X_t) dt + g_t dW_t
$$

and eventually obtaining:

$$
X_1 \sim p_{data}
$$

In regression and classification, the training target is usually a label. In generative modeling, however, the training target is the vector field $u_t^{\theta}$. Therefore, we fit the vector field by minimizing the mean squared error (MSE):

$$
L(\theta) = || u_t^{\theta}(x) - u_t^{target}(x) ||^2
$$

<img src="denoise.png" alt="Denoised Image" width="100%" />

## 2. Conditional probability paths and marginal probability paths

**Definition 1 (Dirac measure)**: for $z \in \mathbb{R}^d$, if $X \sim \delta_z$, then $X = z~a.s.$, i.e. $P(X = z) = 1$.

> A measure is a function that quantifies the size of a set.
> - Length measure: on $\mathbb R$, the measure of an interval $[a,b]$ is its length $b-a$;
> - Area measure: on $\mathbb R^2$, the measure of a rectangle $[a,b]\times[c,d]$ is its area $(b-a)(d-c)$;
> - Probability measure: on a probability space $(\Omega, \mathcal{F}, P)$, the measure of an event $A \in \mathcal{F}$ is its probability $P(A)$.
>
> The Dirac measure $\delta_z$ is a special probability measure that concentrates all mass at a single point $z$:
> $$
> \delta_z(A)=
> \begin{cases}
> 1, & z\in A \\\\
> 0, & z\notin A \\\\
> \end{cases}
> $$

**Definition 2 (Conditional Probability Path)**: $\{P_t(\cdot|z), t \in [0,1]\}$ satisfies:
1. $P_t(\cdot|z)$ is a distribution on $\mathbb{R}^d$;
2. $P_0(\cdot|z) = P_{init},~P_1(\cdot|z) = \delta_z$, where $\delta_z$ is the Dirac measure.

> Example: Gaussian conditional probability path
> 
> $$
> P_t(\cdot|z) = \mathcal{N}(\alpha_t z, \sigma_t^2 I_d)
> $$
>
> Here the noise schedule satisfies $\alpha_t = t, \sigma_t = 1 - t$, so $\alpha_0 = 0, \sigma_0 = 1$, and $\alpha_1 = 1, \sigma_1 = 0$.
> The Gaussian conditional probability path is illustrated below:
>
> <img src="distribution_variance.png" alt="Conditional Probability Path" width="100%" />

**Definition 3 (Marginal Probability Path)**: suppose $z \sim P_{data}$ and $x \sim P_t(\cdot|z)$. Then the marginal probability path $\{P_t, t \in [0,1]\}$, which is independent of $z$, satisfies:
1. $p_t(x) = \int p_t(x|z) p_{data}(z) dz$;
2. $P_0 = P_{init},~P_1 = P_{data}$.

> The marginal probability path is illustrated below:
>
> <img src="../../../posts/flow_matching/marginal_probability_path.png" alt="The marginal probability path evolving from a Gaussian initial distribution into a checkerboard data distribution" width="100%" />

## 3. Conditional vector fields and marginal vector fields

**Definition 4 (Conditional Vector Field)**: $u_t^{target}(x|z), t \in [0,1], x,z \in \mathbb{R}^d$, such that:

$$
X_0 \sim P_{init}, \frac{d}{dt} X_t = u_t^{target}(X_t|z)
$$

Then $X_t$ follows the conditional probability path:

$$
X_t \sim P_t(\cdot|z), t \in [0,1]
$$

> In practice, $P_{init}$ is often equal to $P_0(\cdot|z)$.

> Example: Gaussian conditional vector field
>
> Suppose the Gaussian conditional probability path is
>
> $$
> P_t(\cdot|z) = \mathcal{N}(\alpha_t z, \sigma_t^2 I_d)
> $$
>
> Since $X_t \sim P_t(\cdot|z)$, we can write
>
> $$
> X_t = \alpha_t z + \sigma_t \epsilon,~\epsilon \sim \mathcal{N}(0, I_d)
> $$
>
> Differentiating $X_t$ with respect to $t$ gives the Gaussian conditional vector field:
>
> $$
> \frac{d}{dt} X_t = \dot{\alpha}_t z + \dot{\sigma}_t \epsilon = \dot{\alpha}_t z + \dot{\sigma}_t \frac{X_t - \alpha_t z}{\sigma_t} = \left(\dot{\alpha}_t - \frac{\dot{\sigma}_t}{\sigma_t}\alpha_t \right) z + \frac{\dot{\sigma}_t}{\sigma_t} X_t
> $$
>
> That is,
>
> $$
> u_t^{target}(x|z) = \left(\dot{\alpha}_t - \frac{\dot{\sigma}_t}{\sigma_t}\alpha_t \right) z + \frac{\dot{\sigma}_t}{\sigma_t} x
> $$
>
> where $\dot{\alpha}_t$ and $\dot{\sigma}_t$ denote the derivatives of $\alpha_t$ and $\sigma_t$ with respect to $t$.
> This formula requires $\sigma_t > 0$. For $\sigma_t=1-t$, it applies when $0 \leq t < 1$; the endpoint $P_1(\cdot|z)=\delta_z$ should be understood as a limit in distribution.
>
> <img src="conditional_vector_field_2d.gif" alt="Conditional Vector Field" width="100%" />

**Theorem 1 (Marginalization Trick) / Definition 5 (Marginal Vector Field)**: if $u_t^{target}(x|z)$ is a conditional vector field, then the marginal vector field is:

$$
u_t^{target}(x) = \int u_t^{target}(x|z) P_{data}(z|x) dz \\\\
u_t^{target}(x) = \int u_t^{target}(x|z) \frac{p_t(x|z) p_{data}(z)}{p_t(x)} dz
$$

Then $X_t$ follows the marginal probability path:

$$
X_0 \sim P_{init}, \frac{d}{dt} X_t = u_t^{target}(X_t) \Longrightarrow X_t \sim P_t, t \in [0,1] \Longrightarrow X_1 \sim P_{data}
$$

> By the definition of conditional expectation,
>
> $$
> \mathbb{E}[Y|X_t = x] = \int Y(z) p(z|x) dz
> $$
>
> Let $Y(z) = u_t^{target}(x|z)$. Then
>
> $$
> u_t^{target}(x) = \mathbb{E}[u_t^{target}(x|z)|X_t = x] = \int u_t^{target}(x|z) p(z|x) dz
> $$
>
> which gives the first equality in Theorem 1.

> Intuitively, if we use a conditional vector field in the ODE $\left(X_0 \sim P_{init}, \frac{d}{dt} X_t = u_t^{target}(X_t|z)\right)$, then $X_t$ follows a conditional probability path. If we use a marginal vector field in the ODE $\left(X_0 \sim P_{init}, \frac{d}{dt} X_t = u_t^{target}(X_t)\right)$, then $X_t$ follows a marginal probability path:
> 1. The endpoint of a conditional probability path is a Dirac measure, i.e. $P_1(\cdot|z) = \delta_z$, so $X_1 = z$;
> 2. The endpoint of a marginal probability path is the data distribution, i.e. $P_1 = P_{data}$, so $X_1 \sim P_{data}$.
> 
> This is the core difference between conditional and marginal vector fields. Why does this happen? Because the conditional vector field is defined for each data point $z$, while the marginal vector field is obtained by averaging over all data points, i.e. by marginalization $\left(p_t(x) = \int p_t(x|z) p_{data}(z) dz\right)$.

> <img src="../../../posts/flow_matching/cvf_mvf_visualization.png" alt="Comparison of conditional and marginal probability paths" width="100%" />

**Theorem (Continuity Equation)**: for any ODE initialized by $X_0 \sim P_{init}, \frac{d}{dt} X_t = u_t(X_t)$, the density $p_t$ satisfies the following PDE:

> <img src="../../../posts/flow_matching/continuity_equation.png" alt="Probability inflow and outflow in a vector field" width="100%" />
>
> The diagram illustrates probability mass flowing into and out of a local region along the vector field.

$$
\frac{d}{dt}p_t(x) = - \text{div}(p_t u_t)(x) \Longleftrightarrow X_t \sim P_t, t \in [0,1]
$$

Here $\text{div}$ denotes divergence, defined by $\text{div}(f)(x) = \sum_{i=1}^d \frac{\partial f_i(x)}{\partial x_i}$. The quantity $p_t(x)u_t(x)$ is a vector field called the **probability flow** or **flux**:
- $p_t(x)$ is the probability density, representing probability mass per unit volume;
- $u_t(x)$ is the velocity vector, representing the direction and speed at which probability mass moves per unit time.

Therefore, $\text{div}(p_t u_t)(x)$ is the divergence of the probability flow at point $x$, i.e. the net change of probability mass flowing into or out of $x$ per unit time.

> - When $\text{div}(p_tu_t)(x) > 0$, probability mass has a net outflow and the local density decreases;
> - When $\text{div}(p_tu_t)(x) < 0$, probability mass has a net inflow and the local density increases;
> - When $\text{div}(p_tu_t)(x) = 0$, the local density remains unchanged.
>
> This is why there is a minus sign in front of $\text{div}(p_t u_t)(x)$.
>
> The proof is omitted here.

## 4. Flow Matching Training Procedure

| **Algorithm 3** Flow Matching Training Procedure (General) |
| --- |
| **Input**: a dataset of samples \(z \sim p_{data}\), neural vector field \(u_t^\theta\) |
| 1: **for** each mini-batch of data **do** |
| 2: \(\quad\)Sample a data example \(z\) from the dataset |
| 3: \(\quad\)Sample a random time \(t \sim \operatorname{Unif}[0,1]\) |
| 4: \(\quad\)Sample \(x \sim p_t(\cdot\mid z)\) from the conditional probability path |
| 5: \(\quad\)Compute \(\mathcal{L}(\theta)=\lVert u_t^\theta(x)-u_t^{target}(x\mid z)\rVert^2\) |
| 6: \(\quad\)Update the model parameters \(\theta\) via gradient descent on \(\mathcal{L}(\theta)\) |
| 7: **end for** |

> **Example: Conditional Flow Matching for a Gaussian probability path**
>
> Let the conditional probability path be
>
> \[P_t(\cdot\mid z)=\mathcal{N}(\alpha_tz,\sigma_t^2I_d)\]
>
> with conditional vector field
>
> \[u_t^{target}(x\mid z)=\left(\dot{\alpha}_t-\frac{\dot{\sigma}_t}{\sigma_t}\alpha_t\right)z+\frac{\dot{\sigma}_t}{\sigma_t}x\]
>
> Sampling from the conditional probability path is equivalent to sampling \(\epsilon\sim\mathcal{N}(0,I_d)\) and setting
>
> \[x=\alpha_tz+\sigma_t\epsilon\]
>
> Substituting this noise-sampling rule into the conditional flow matching loss gives
>
> \[\mathcal{L}_{CFM}(\theta)=\mathbb{E}_{\substack{t\sim\operatorname{Unif}[0,1],\ z\sim p_{data},\ x\sim p_t(\cdot\mid z)}}\left[\left\lVert u_t^\theta(x)-u_t^{target}(x\mid z)\right\rVert^2\right]\]
>
> \[=\mathbb{E}_{\substack{t\sim\operatorname{Unif}[0,1],\ z\sim p_{data},\ \epsilon\sim\mathcal{N}(0,I_d)}}\left[\left\lVert u_t^\theta(\alpha_tz+\sigma_t\epsilon)-u_t^{target}(\alpha_tz+\sigma_t\epsilon\mid z)\right\rVert^2\right]\]
>
> \[=\mathbb{E}_{\substack{t\sim\operatorname{Unif}[0,1],\ z\sim p_{data},\ \epsilon\sim\mathcal{N}(0,I_d)}}\left[\left\lVert u_t^\theta(\alpha_tz+\sigma_t\epsilon)-\left(\dot{\alpha}_tz+\dot{\sigma}_t\epsilon\right)\right\rVert^2\right]\]
>
> Thus, the model input is the data-noise mixture \(\alpha_tz+\sigma_t\epsilon\), while the training target is the corresponding velocity \(\dot{\alpha}_tz+\dot{\sigma}_t\epsilon\).

### 4.1 Straight-Line Schedule: Predicting the Difference Between Data and Noise

Choose the straight-line schedule for the Gaussian conditional probability path:

\[
\alpha_t=t,\qquad \sigma_t=1-t.
\]

The conditional probability path and its sampling rule are then

\[
P_t(\cdot\mid z)=\mathcal{N}\left(tz,(1-t)^2I_d\right),
\qquad
x=tz+(1-t)\epsilon,\epsilon\sim\mathcal{N}(0,I_d).
\]

Because \(\dot{\alpha}_t=1\) and \(\dot{\sigma}_t=-1\), the target velocity simplifies to \(z-\epsilon\). The conditional flow matching loss therefore becomes

\[
\begin{aligned}
\mathcal{L}_{CFM}(\theta)
&=\mathbb{E}_{\substack{t\sim\operatorname{Unif}[0,1],\ z\sim p_{data},\ \epsilon\sim\mathcal{N}(0,I_d)}}
\left[\left\lVert u_t^\theta(\alpha_tz+\sigma_t\epsilon)-\left(\dot{\alpha}_tz+\dot{\sigma}_t\epsilon\right)\right\rVert^2\right] \\
&=\mathbb{E}_{\substack{t\sim\operatorname{Unif}[0,1],\ z\sim p_{data},\ \epsilon\sim\mathcal{N}(0,I_d)}}
\left[\left\lVert u_t^\theta\left(tz+(1-t)\epsilon\right)-\left(z-\epsilon\right)\right\rVert^2\right].
\end{aligned}
\]

This straight-line path is also called the Conditional Optimal Transport (CondOT) path. The model input is a linear interpolation between noise and data, while the training target is the difference between data and noise.

> <img src="../../../posts/flow_matching/straight_line_schedule.png" alt="A straight-line schedule interpolating between Gaussian noise and a data sample" width="100%" />
>
> The straight-line schedule starts at noise \(\epsilon\) and moves along a straight line to the data sample \(z\). Figure credit: Yaron Lipman.

| **Algorithm 4** Flow Matching Training for the CondOT Path |
| --- |
| **Input**: a dataset of samples \(z\sim p_{data}\), neural vector field \(u_t^\theta\) |
| 1: **for** each mini-batch of data **do** |
| 2: \(\quad\)Sample a data example \(z\) from the dataset |
| 3: \(\quad\)Sample a random time \(t\sim\operatorname{Unif}[0,1]\) |
| 4: \(\quad\)Sample noise \(\epsilon\sim\mathcal{N}(0,I_d)\) |
| 5: \(\quad\)Set \(x=tz+(1-t)\epsilon\) |
| 6: \(\quad\)Compute \(\mathcal{L}(\theta)=\left\lVert u_t^\theta(x)-(z-\epsilon)\right\rVert^2\) |
| 7: \(\quad\)Update the model parameters \(\theta\) via gradient descent on \(\mathcal{L}(\theta)\) |
| 8: **end for** |

After training, use [Algorithm 2: Sampling from a flow model with Euler’s method](../flow_and_diffusion_models/#213-definition-of-a-flow-model) from *Flow and Diffusion Models* to generate samples from the initial distribution by following the learned vector field.

## 5. Summary of Conditional/Marginal Paths and Vector Fields

The objects in Flow Matching can be organized into the following sequence:

| Stage | Conditional form | Marginal form | Role |
| --- | --- | --- | --- |
| Probability path | Conditional probability path | Marginal probability path | Defines how the distribution evolves from noise to data |
| Vector field | Conditional vector field | Marginal vector field | Defines the training target to be learned |
| Flow Matching loss | Conditional Flow Matching loss | Marginal Flow Matching loss | Defines the objective minimized during training |

### 5.1 Conditional Objects: Analytically Tractable

| Object | Notation | Key property | Gaussian example |
| --- | --- | --- | --- |
| Conditional probability path | \(P_t(\cdot \mid z)\) | Interpolates between \(P_{init}\) and a data point \(z\) | \(\mathcal{N}(\alpha_t z, \sigma_t^2 I_d)\) |
| Conditional vector field | \(u_t^{target}(x \mid z)\) | The ODE follows the conditional path | \(\left(\dot{\alpha}_t - \frac{\dot{\sigma}_t}{\sigma_t}\alpha_t\right) z + \frac{\dot{\sigma}_t}{\sigma_t} x\) |
| Conditional Flow Matching loss | \(\mathcal{L}_{CFM}(\theta)\) | The loss minimized directly during training | \(\mathbb{E}_{t,z,x}\!\left[\left\lVert u_t^\theta(x)-u_t^{target}(x\mid z)\right\rVert^2\right]\) |

Here, \(t\sim\operatorname{Unif}[0,1]\), \(z\sim P_{data}\), and \(x\sim P_t(\cdot\mid z)\). For commonly used Gaussian conditional probability paths, all three conditional objects have analytical formulas, so we can sample from them and evaluate the training loss directly.

### 5.2 Marginal Objects: Intractable Directly, but Learnable Implicitly

| Object | Notation | Key property | Formula |
| --- | --- | --- | --- |
| Marginal probability path | \(P_t\) | Interpolates between \(P_{init}\) and \(P_{data}\) | \(p_t(x) = \int p_t(x \mid z) p_{data}(z) \, dz\) |
| Marginal vector field | \(u_t^{target}(x)\) | The ODE follows the marginal path | \(u_t^{target}(x) = \int u_t^{target}(x \mid z) \frac{p_t(x \mid z) p_{data}(z)}{p_t(x)} \, dz\) |
| Marginal Flow Matching loss | \(\mathcal{L}_{FM}(\theta)\) | The ideal loss we would like to minimize | \(\mathbb{E}_{t,x}\!\left[\left\lVert u_t^\theta(x)-u_t^{target}(x)\right\rVert^2\right]\) |

In the marginal Flow Matching loss, \(t\sim\operatorname{Unif}[0,1]\) and \(x\sim P_t\). The marginal probability path requires integrating over the full data distribution, and the marginal vector field depends on the marginal density. These objects are therefore generally unavailable for direct evaluation. However, we can show that the Conditional Flow Matching loss and the marginal Flow Matching loss differ only by a constant independent of the model parameters \(\theta\).

<details>
<summary>Derivation: the conditional and marginal Flow Matching losses differ only by a constant</summary>

By the definition of the marginal vector field,

\[
u_t^{target}(X_t)
=\mathbb{E}\!\left[u_t^{target}(X_t\mid Z)\mid t,X_t\right].
\]

Add and subtract the marginal vector field in the Conditional Flow Matching prediction error:

\[
u_t^\theta(X_t)-u_t^{target}(X_t\mid Z)
=\left(u_t^\theta(X_t)-u_t^{target}(X_t)\right)
-\left(u_t^{target}(X_t\mid Z)-u_t^{target}(X_t)\right).
\]

Substituting this identity into the Conditional Flow Matching loss and expanding the squared norm gives

\[
\begin{aligned}
\mathcal{L}_{CFM}(\theta)
&=\mathbb{E}_{t,X_t}\!\left[
\left\lVert u_t^\theta(X_t)-u_t^{target}(X_t)\right\rVert^2
\right] \\
&\quad+\mathbb{E}_{t,Z,X_t}\!\left[
\left\lVert u_t^{target}(X_t\mid Z)-u_t^{target}(X_t)\right\rVert^2
\right] \\
&\quad-2\mathbb{E}_{t,Z,X_t}\!\left[
\left\langle
u_t^\theta(X_t)-u_t^{target}(X_t),
u_t^{target}(X_t\mid Z)-u_t^{target}(X_t)
\right\rangle
\right].
\end{aligned}
\]

The cross term vanishes for the following three reasons.

1. **After conditioning on \(t,X_t\), only \(Z\) remains random.**

   More explicitly, after conditioning on \(t,X_t=x\), both \(u_t^\theta(x)\) and \(u_t^{target}(x)\) are fixed vectors. Only the data point \(Z\) remains random because the same noisy sample \(x\) may have arisen from different data points.

2. **Averaging the conditional vector field over \(Z\) gives the marginal vector field.**

   By the definition of the marginal vector field, after fixing \(t,X_t=x\),

   \[
   \begin{aligned}
   &\mathbb{E}\!\left[u_t^{target}(x\mid Z)\mid t,X_t=x\right] \\
   &\quad=\int u_t^{target}(x\mid z)
   \frac{p_t(x\mid z)p_{data}(z)}{p_t(x)}\,dz \\
   &\quad=u_t^{target}(x).
   \end{aligned}
   \]

   Therefore, the conditional mean of the difference between the conditional and marginal vector fields is zero:

   \[
   \begin{aligned}
   &\mathbb{E}\!\left[
   u_t^{target}(X_t\mid Z)-u_t^{target}(X_t)
   \mid t,X_t
   \right] \\
   &\quad=\mathbb{E}\!\left[
   u_t^{target}(X_t\mid Z)\mid t,X_t
   \right]-u_t^{target}(X_t) \\
   &\quad=u_t^{target}(X_t)-u_t^{target}(X_t)=0.
   \end{aligned}
   \]

3. **Apply the law of total expectation to the cross term.**

   First take the conditional expectation given \(t,X_t\), and then average over \(t,X_t\). Because the first factor in the cross term is fixed after conditioning on \(t,X_t\), it can be moved outside the inner conditional expectation:

   \[
   \begin{aligned}
   &\mathbb{E}_{t,Z,X_t}\!\left[
   \left\langle
   u_t^\theta(X_t)-u_t^{target}(X_t),
   u_t^{target}(X_t\mid Z)-u_t^{target}(X_t)
   \right\rangle
   \right] \\
   &\quad=\mathbb{E}_{t,X_t}\!\left[
   \left\langle
   u_t^\theta(X_t)-u_t^{target}(X_t),
   \mathbb{E}\!\left[
   u_t^{target}(X_t\mid Z)-u_t^{target}(X_t)
   \mid t,X_t
   \right]
   \right\rangle
   \right] \\
   &\quad=0.
   \end{aligned}
   \]

The cross term therefore vanishes. The first term is exactly the marginal Flow Matching loss, so

</details>

\[
\begin{aligned}
\mathcal{L}_{CFM}(\theta)
&=\mathcal{L}_{FM}(\theta)+C, \\
C
&=\mathbb{E}_{t,Z,X_t}\!\left[
\left\lVert
u_t^{target}(X_t\mid Z)-u_t^{target}(X_t)
\right\rVert^2
\right].
\end{aligned}
\]

Neither the conditional nor the marginal target vector field depends on the model parameters \(\theta\), so \(C\) is also independent of \(\theta\). The two losses consequently have the same gradient with respect to \(\theta\). Although training evaluates the tractable conditional loss, the model ultimately learns the marginal vector field:

\[
\begin{aligned}
\nabla_\theta\mathcal{L}_{CFM}(\theta)
&=\nabla_\theta\mathcal{L}_{FM}(\theta).
\end{aligned}
\]

---

## References

[1] GPT bilingual subtitle course resource, "Flow Matching and Diffusion Models | 6.S184 Flow Matching and Diffusion Models (Chinese-English subtitles, Claude-3.7-s)," Bilibili, Jul. 29, 2025. [Online video]. Available: https://www.bilibili.com/video/BV1gc8Ez8EFL. Accessed: Jan. 30, 2026.

[2] P. Holderrieth and R. Shprints, "Flow Matching," MIT 6.S184 Lecture 2 slides, 2026. [Online]. Available: https://diffusion.csail.mit.edu/2026/docs/20260122_Lecture_02.pdf
