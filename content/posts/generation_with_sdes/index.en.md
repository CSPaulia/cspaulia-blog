---
title: "Generation Models with SDEs"
date: 2025-10-10T8:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
  main: "Generative Models"
  subseries: "Flow Matching"
categories: ["Generative Models"]
tags: ["Flow Matching", "Diffusion Models"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "Notes for Flow Matching and Diffusion Lecture 1 from MIT course."
# canonicalURL: "https://canonical.url/to/page"
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
    image: "ode.png" # image path/url
    alt: "Ordinary Differential Equation Illustration" # alt text
    caption: "Ordinary Differential Equation (ODE)" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## 1. Generating Objects

### 1.1 Represent the object as a vector

1. Images:
  - Spatial dimensions: height $H$ and width $W$
  - Color channels: RGB

$$
z \in \mathbb{R}^{H \times W \times 3}
$$

2. Videos:
  - Temporal dimension: $T$ frames
  - Each frame is an image

$$
z \in \mathbb{R}^{T \times H \times W \times 3}
$$

3. Molecular structures:
  - $N$ atoms
  - Each atom has 3 coordinates
  
$$
z \in \mathbb{R}^{N \times 3}
$$

**In general, we can represent the object we want to generate as a vector:**

$$
z \in \mathbb{R}^{d}
$$

---

### 1.2 Generation: sampling from the data distribution

**Data distribution**: the distribution of objects we want to generate, denoted by $p_{data}$.

**Probability density**: $p_{data}: \mathbb{R}^d \to \mathbb{R}_{\ge 0}$.

{{< alert type="warning" title="Note" >}}
We do not know the true probability density in practice.
{{< /alert >}}

**Generation** means sampling $z \sim p_{data}$. For example:

{{< img src="Bao.jpg" alt="Baobao" width="300" >}}

---

### 1.3 Dataset: finite samples from the data distribution

**Dataset**: a finite collection of samples from the data distribution, $z_1, \cdots, z_N \sim p_{data}$.

---

### 1.4 Conditional generation

**Conditional generation** means sampling from a conditional distribution $z \sim p_{data}(\cdot \mid y)$.

---

## 2. Flow and Diffusion Models

### 2.1 Flow Models

#### 2.1.1 Preliminaries

**Definition 1 (Trajectory)**: $X: [0, 1] \to \mathbb{R}^d, t \mapsto X_t$.

> Intuition: a trajectory describes how a particle’s position changes over time $t \in [0, 1]$. The input is time $t$, and the output is the position $X_t$ at that time.

{{< img src="traj.png" alt="Trajectory" width="300" >}}

**Definition 2 (Vector field)**: $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d, (x, t) \mapsto u_t(x)$, where $x$ is the location and $u_t(x)$ is the vector direction.

{{< img src="vector_field.png" alt="Vector Field" width="300" >}}

**Definition 3 (Ordinary Differential Equation, ODE)**: given the initial condition $X_0 = x_0$, the dynamics are

$$
\frac{dX_t}{dt} = u_t(X_t)
$$

{{< img src="ode.png" alt="ODE" width="300" >}}

> $dX_t/dt$ is the tangent vector of the trajectory. You can interpret it as velocity, so an ODE describes how a particle moves in a vector field.

#### 2.1.2 Definition of a flow

**Definition 4 (Flow)**: $\phi: \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d, (t, x) \mapsto \phi_t(x)$, where $\phi_t(x)$ is the new position at time $t$ of a particle that starts at position $x$ and follows the ODE trajectory. A flow satisfies:

$$
\phi_0(x_0) = x_0
$$

$$
\frac{d}{dt}\phi_t(x_0) = u_t(\phi_t(x_0))
$$

Intuitively, a flow is a collection of ODE solutions for many different initial conditions.

{{< img src="flow_process.png" alt="Flow Process" width="80%" >}}

Flow visualization:

{{< img src="flow.gif" alt="Flow" width="300" >}}

> **Linear ODE:**
> 
> A simple vector field:
> $$ u_t(x) = - \theta x $$
>
> The flow is defined by the ODE:
> $$ \frac{d}{dt} \phi_t(x) = - \theta \phi_t(x) $$
> $$ \frac{d \phi_t(x)}{\phi_t(x)} = - \theta dt $$
> $$ \int \frac{d \phi_t(x)}{\phi_t(x)} = \int - \theta dt $$
> $$ \log \phi_t(x) = - \theta t + C(x) $$
> $$ \phi_t(x) = e^{-\theta t + C(x)} = e^{C(x)} e^{-\theta t} $$
>
> Use the initial condition to determine the constant $C(x)$:
> $$ \phi_0(x) = e^{C(x)} e^{0} = e^{C(x)} = x \Rightarrow e^{C(x)} = x \Rightarrow C(x) = \log x $$
>
> The closed-form flow is:
> $$ \phi_t(x) = x e^{-\theta t} $$
>
> Verify it satisfies the definition:
> $$ \phi_0(x) = e^{C(x)} e^{0} = e^{C(x)} = x \Rightarrow e^{C(x)} = x $$
> $$ \frac{d}{dt} \phi_t(x) = \frac{d}{dt} (x e^{-\theta t}) = -\theta x e^{-\theta t} = -\theta \phi_t(x) $$
>
> {{< img src="linear_ode_trajectories.png" alt="Linear ODE Trajectories" width="80%" >}}

| **Algorithm 1** Solving an ODE with Euler’s method |
|-------------------------------|
| **Input**: vector field $u_t$, initial position $x_0$, number of steps $n$ |
| 1: Set $t = 0$ |
| 2: Set step size $h = 1/n$ |
| 3: Set $X_0 = x_0$ |
| 4: **for** $i = 0$ **to** $n-1$ **do** |
| 5: $~~~~$Update $X_{t+h} = X_t + h \cdot u_{t}(X_t)$ |
| 6: $~~~~$Update $t = t + h$ |
| 7: **end for** |
| **Output**: $X_0, X_h, X_{2h}, \cdots, X_1$ |

#### 2.1.3 Definition of a flow model

> $p_{init} \xrightarrow{ODE} p_{data}$

**Definition 5 (Neural vector field)**: $u_t^{\theta}: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$, where $\theta$ denotes parameters.

- Sample an initial point: $X_0 \sim p_{init}$
- Simulate the ODE driven by $u_t^{\theta}$
- Goal: $X_1 \sim p_{data}$

{{< img src="flow_distribution.gif" alt="Flow Distribution" width="300" >}}

| **Algorithm 2** Sampling from a flow model with Euler’s method |
|-------------------------------|
| **Input**: neural vector field $u_t^{\theta}$, number of steps $n$ |
| 1: Set $t = 0$ |
| 2: Set step size $h = 1/n$ |
| 3: Sample $X_0 \sim p_{init}$ |
| 4: **for** $i = 0$ **to** $n-1$ **do** |
| 5: $~~~~$Update $X_{t+h} = X_t + h \cdot u_{t}^{\theta}(X_t)$ |
| 6: $~~~~$Update $t = t + h$ |
| 7: **end for** |
| **Output**: $X_1$ |

---

### 2.2 Diffusion models

**Definition 6 (Stochastic process)**: random variables $X_t, 0 \le t \le 1$, whose trajectory is $X: [0, 1] \to \mathbb{R}^d, t \mapsto X_t$.

{{< img src="stochastic_traj.png" alt="Stochastic Process" width="300" >}}

> **Definition 2 (Vector field)**: $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d, (x, t) \mapsto u_t(x)$.

**Definition 7 (Diffusion coefficient)**: $\sigma: [0,1] \to \mathbb{R}, t \mapsto \sigma_t$.

**Definition 8 (Stochastic Differential Equation, SDE)**: given $X_0 = x_0$, the dynamics are

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t.
$$

Here $u_t(X_t)dt$ is the deterministic ODE part, $\sigma_t dW_t$ is the stochastic (noise) part, and $W_t$ is standard Brownian motion (Wiener process).

#### 2.2.1 Brownian motion

As a stochastic process $W_t$:
1. $W_0 = 0$
2. Gaussian increments: for any $0 \leq s < t \leq 1$, $W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$
3. Independent increments: for any $0 \leq t_0 < t_1 < \cdots < t_n \leq 1$, the increments $W_{t_1} - W_{t_0}, W_{t_2} - W_{t_1}, \cdots, W_{t_n} - W_{t_{n-1}}$ are independent

#### 2.2.2 Expansions for $X_t$

For an ODE, $X_t$ can be expanded as:

$$
\frac{d}{dt} X_t = u_t(X_t) \longleftrightarrow X_{t+h} = X_t + h \cdot u_t(X_t) + h \cdot R_t(h)
$$

where $\lim_{h \to 0} R_t(h) = 0$ and $h \cdot R_t(h)$ is a higher-order infinitesimal (often written as $o(h)$). The right-hand side can be viewed as a Taylor expansion.

> Derivation:
> $$ \frac{d}{dt} X_t = u_t(X_t) $$
> $$ \lim_{h \to 0} \frac{X_{t+h} - X_t}{h} = u_t(X_t) $$
> $$ \frac{X_{t+h} - X_t}{h} = u_t(X_t) + R_t(h) $$
> $$ X_{t+h} = X_t + h \cdot u_t(X_t) + h \cdot R_t(h) $$

For an SDE, $X_t$ can be expanded as:

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t \longleftrightarrow X_{t+h} = X_t + h \cdot u_t(X_t) + \sigma_t (W_{t+h} - W_t) + h \cdot R_t(h),
$$

where $\lim_{h \to 0} \sqrt{\mathbb{E}[\|R_t(h)\|^2]} = 0$. Because Brownian motion is not differentiable, the remainder term is typically defined via mean-square ($L^2$) convergence. Since $W_{t+h} - W_t \sim \mathcal{N}(0, hI_d)$, we can write it as $\sqrt{h}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, I_d)$.

| **Algorithm 3** Sampling from a diffusion model (Euler–Maruyama for SDEs) |
|-------------------------------|
| **Input**: vector field $u_t$, diffusion coefficient $\sigma_t$, number of steps $n$ |
| 1: Set $t = 0$ |
| 2: Set step size $h = 1/n$ |
| 3: Set $X_0 = x_0$ |
| 4: **for** $i = 0$ **to** $n-1$ **do** |
| 5: $~~~~$Sample $\epsilon \sim \mathcal{N}(0, I_d)$ |
| 6: $~~~~$Update $X_{t+h} = X_t + h \cdot u_{t}(X_t) + \sigma_t \sqrt{h} \cdot \epsilon$ |
| 7: $~~~~$Update $t = t + h$ |
| 8: **end for** |
| **Output**: $X_0, X_h, X_{2h}, \cdots, X_1$ |

{{< img src="linear_sde_euler_trajectories.png" alt="Linear SDE Trajectories" width="80%" >}}

#### 2.2.3 Definition of a diffusion model

> $p_{init} \xrightarrow{SDE} p_{data}$

> **Definition 5 (Neural network)**: $u_t^{\theta}: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$, where $\theta$ denotes parameters.

> **Definition 7 (Diffusion coefficient)**: $\sigma: [0,1] \to \mathbb{R}, t \mapsto \sigma_t$

- Sample an initial point: $X_0 \sim p_{init}$
- Simulate the SDE: $dX_t = u_t^{\theta}(X_t)dt + \sigma_t dW_t$
- Goal: $X_1 \sim p_{data}$

---

## References

[1] GPT中英字幕课程资源, "《流匹配与扩散模型|6.S184 Flow Matching and Diffusion Models》中英字幕（Claude-3.7-s）》," Bilibili, Jul. 29, 2025. [Online video]. Available: https://www.bilibili.com/video/BV1gc8Ez8EFL. Accessed: Jan. 30, 2026.