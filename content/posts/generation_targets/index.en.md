---
title: "Training Targets for Generative Models"
date: 2026-02-03T11:10:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "Generative Models"
    subseries: "Model Training"
categories: ["Generative Models"]
tags: ["Flow Matching", "Diffusion Models"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "Notes for Lecture 2 of MIT's Flow Matching and Diffusion course."
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

## 1. Training targets

Training means finding parameters $\theta$ such that:

$$
X_0 \sim p_{init}, dX_t = u_t^{\theta}(X_t) dt~\text{or}~dX_t = u_t^{\theta}(X_t) dt + \sigma_t dW_t
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
1. $P_t(x) = \int P_t(x|z) P_{data}(z) dz$;
2. $P_0 = P_{init},~P_1 = P_{data}$.

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
>
> <img src="conditional_vector_field_2d.gif" alt="Conditional Vector Field" width="100%" />

**Theorem 1 (Marginalization Trick) / Definition 5 (Marginal Vector Field)**: if $u_t^{target}(x|z)$ is a conditional vector field, then the marginal vector field is:

$$
u_t^{target}(x) = \int u_t^{target}(x|z) P_{data}(z|x) dz \\\\
u_t^{target}(x) = \int u_t^{target}(x|z) \frac{P_t(x|z) P_{data}(z)}{P_t(x)} dz
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

> Intuitively: if we use a conditional vector field in the ODE $\left(X_0 \sim P_{init}, \frac{d}{dt} X_t = u_t^{target}(X_t)\right)$, then $X_t$ follows a conditional probability path; if we use a marginal vector field, then $X_t$ follows a marginal probability path:
> 1. The endpoint of a conditional probability path is a Dirac measure, i.e. $P_1(\cdot|z) = \delta_z$, so $X_1 = z$;
> 2. The endpoint of a marginal probability path is the data distribution, i.e. $P_1 = P_{data}$, so $X_1 \sim P_{data}$.
> 
> This is the core difference between conditional and marginal vector fields. Why does this happen? Because the conditional vector field is defined for each data point $z$, while the marginal vector field is obtained by averaging over all data points, i.e. by marginalization $\left(P_t(x) = \int P_t(x|z) P_{data}(z) dz\right)$.

**Theorem (Continuity Equation)**: for any ODE initialized by $X_0 \sim P_{init}, \frac{d}{dt} X_t = u_t(X_t)$, the density $p_t$ satisfies the following PDE:

$$
\partial_t p_t(x) = - \text{div}(p_t u_t)(x) \Longleftrightarrow X_t \sim P_t, t \in [0,1]
$$

Here $\text{div}$ denotes divergence, defined by $\text{div}(f)(x) = \sum_{i=1}^d \frac{\partial f_i(x)}{\partial x_i}$. The quantity $p_t(x)u_t(x)$ is a vector field called the **probability flow** or **flux**:
- $p_t(x)$ is the probability density, representing probability mass per unit volume;
- $u_t(x)$ is the velocity vector, representing the direction and speed at which probability mass moves per unit time.

Therefore, $\text{div}(p_t u_t)(x)$ is the divergence of the probability flow at point $x$, i.e. the net change of probability mass flowing into or out of $x$ per unit time.

> When the net outflow is large, the local density decreases;
> when the net outflow is small, the local density increases.
>
> This is why there is a minus sign in front of $\text{div}(p_t u_t)(x)$.
>
> The proof is omitted here.

## 4. Score functions for conditional and marginal vector fields

**Definition 6 (Conditional Score)**: $\nabla_x \log p_t(x|z)$.

**Definition 7 (Marginal Score)**: $\nabla_x \log p_t(x)$.

> Deriving the **marginal score** from the **conditional score**:
>
> $$
> \nabla_x \log p_t(x) = \frac{\nabla_x p_t(x)}{p_t(x)} = \frac{\int \nabla_x p_t(x|z) p_{data}(z) dz}{p_t(x)} = \int \nabla_x \log p_t(x|z) \frac{p_t(x|z) p_{data}(z)}{p_t(x)} dz \\\\
> = \int s_t^{target}(x|z) p(z|x) dz = \nabla_x \log p_t(x)
> $$

> **Gaussian score**:
>
> $$
> \nabla_x \log p_t(x|z) = \nabla_x \log \mathcal{N}(\alpha_t z, \sigma_t^2 I_d) = -\frac{1}{\sigma_t^2}(x - \alpha_t z)
> $$

## 5. The SDE extension trick and the Fokker-Planck equation

**Theorem 2 (SDE Extension Trick)**: let $u_t^{target}(x) = \int u_t^{target}(x|z) p_{data}(z|x) dz$. Then for any $\sigma_t \geq 0$:

$$
X_0 \sim P_{init}, dX_t = [u_t^{target}(X_t) + \frac{\sigma_t^2}{2} \nabla_x \log p_t(X_t)] dt + \sigma_t dW_t \Longrightarrow X_t \sim P_t, t \in [0,1] \\\\
\Longrightarrow X_1 \sim P_{data}
$$

**Theorem 3 (Fokker-Planck Equation)**: given any SDE $X_0 \sim P_{init}, dX_t = u_t(X_t) dt + \sigma_t dW_t$, the density $p_t$ satisfies the following PDE:

$$
\partial_t p_t(x) = - \text{div}(p_t u_t)(x) + \frac{1}{2} \sigma_t^2 \Delta p_t(x) \Longleftrightarrow X_t \sim P_t, t \in [0,1]
$$

Here $- \text{div}(p_t u_t)(x)$ is the continuity equation term, while $\frac{1}{2} \sigma_t^2 \Delta p_t(x)$ is the heat equation (diffusion) term.

## 6. Summary of conditional/marginal paths, vector fields, and score functions

Conditional probability paths, conditional vector fields, and conditional scores:

| Object | Notation | Key property | Gaussian example |
| --- | --- | --- | --- |
| Conditional probability path | $p_t(\cdot \mid z)$ | Interpolates between $p_{init}$ and a data point $z$ | $\mathcal{N}(\alpha_t z, \beta_t^2 I_d)$ |
| Conditional vector field | $u_t^{target}(x \mid z)$ | The ODE follows the conditional path | $\left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x$ |
| Conditional score function | $\nabla_x \log p_t(x \mid z)$ | Gradient of the log-likelihood with respect to $x$ | $-\frac{x - \alpha_t z}{\beta_t^2}$ |

Marginal probability paths, marginal vector fields, and marginal scores:

| Object | Notation | Key property | Formula |
| --- | --- | --- | --- |
| Marginal probability path | $p_t(x)$ | Interpolates between $p_{init}$ and $p_{data}$ | $p_t(x) = \int p_t(x \mid z) p_{data}(z) \, dz$ |
| Marginal vector field | $u_t^{target}(x)$ | The ODE follows the marginal path | $u_t^{target}(x) = \int u_t^{target}(x \mid z) \frac{p_t(x \mid z) p_{data}(z)}{p_t(x)} \, dz$ |
| Marginal score function | $\nabla_x \log p_t(x)$ | Can be used to convert an ODE target into an SDE target | $\nabla_x \log p_t(x) = \int \nabla_x \log p_t(x \mid z) \frac{p_t(x \mid z) p_{data}(z)}{p_t(x)} \, dz$ |

---

## References

[1] GPT bilingual subtitle course resource, "Flow Matching and Diffusion Models | 6.S184 Flow Matching and Diffusion Models (Chinese-English subtitles, Claude-3.7-s)," Bilibili, Jul. 29, 2025. [Online video]. Available: https://www.bilibili.com/video/BV1gc8Ez8EFL. Accessed: Jan. 30, 2026.
