---
title: "流匹配与扩散模型"
date: 2025-10-10T8:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "生成模型"
    subseries: "流匹配"
categories: ["AIGC"]
tags: ["流匹配", "扩散模型"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "Notes for Flow Matching and Diffusion from MIT course."
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
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: true # only hide on current single page
    hiddenInList: true # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## 1. 生成对象

### 1.1. 将生成对象表征为向量

1. 图像：
    - 尺度维度：高度 $H$ 和 宽度 $W$
    - 颜色维度：三维颜色通道 RGB

$$
z \in \mathbb{R}^{H \times W \times 3}
$$

2. 视频：
    - 时间维度：时间帧 $T$
    - 每一帧为图像

$$
z \in \mathbb{R}^{T \times H \times W \times 3}
$$

3. 分子结构：
    - N 个原子
    - 每个原子包含三个维度
  
$$
z \in \mathbb{R}^{N \times 3}
$$

**我们可以将我们想要生成的目标表征为向量：**

$$
z \in \mathbb{R}^{d}
$$

---

### 1.2. 生成——从数据分布中采样

**数据分布**：我们想要生成的对象的分布$p_{data}$

**概率密度**：$p_{data}: \mathbb{R}^d \to \mathbb{R} \ge 0, z \mapsto p_{data}$

{{< alert type="warning" title="注意" >}}
我们并不知道实际的概率密度
{{< /alert >}}

**生成**：意味着从数据分布中进行采样 $z \sim p_{data}$，采样 $z$ 如下：

<img src="Bao.jpg" alt="Baobao" width="300"/>

---

### 1.3. 数据集——由数据分布中的样本组成

**数据集**：数据分布中有限数量的样本的集合 $z_1, \cdots, z_N \sim p_{data}$

---

### 1.4. 条件生成

**条件生成**：意味着从条件数据分布中进行采样 $z \sim p_{data}(\cdot | y)$

---

## 2. 流与扩散模型

### 2.1. 流模型

#### 2.1.1. 预备知识

<dl class="definition-list">
  <dt>轨迹（Trajectory）</dt>
  <dd>
    <span class="math">$X: [0, 1] \to \mathbb{R}^d, t \mapsto X_t$
    </span>
  </dd>
</dl>

<img src="Traj.png" alt="Trajectory" width="300"/>

<dl class="definition-list">
  <dt>向量场（Vector Feild）</dt>
  <dd>
    <span class="math">$u: R^d \times [0,1] \to \mathbb{R}^d, (x, t) \mapsto u_t(x)$
    </span>
    <br>其中 $x$ 表示点的位置，$u_t$ 表示向量方向
  </dd>
</dl>

<img src="vector_field.png" alt="Vector Field" width="300"/>

<dl class="definition-list">
  <dt>常微分方程（Ordinary Differential Equation, ODE）</dt>
  <dd>
    初始条件
    <span class="math">$X_0 = x_0$
    </span>
    <br>
    常微分方程/动力学方程
    <span class="math">$dX_t/dt = u_t(X_t)$
    </span>
  </dd>
</dl>

<img src="ode.png" alt="ODE" width="300"/>

> $dX_t/dt$ 即为轨迹的切线，可以理解为速度，因此ODE描述了粒子在向量场中的运动

#### 2.1.2. 流的定义

<dl class="definition-list">
  <dt>流（Flow）</dt>
  <dd>
    <span class="math">$\phi: \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d, (t, x) \mapsto \phi_t(x)$
    </span>
    <br>其中 $\phi_t(x)$ 表示时间 $t$ 时刻，位置为 $x$ 的粒子沿着ODE轨迹运动到的新位置
    <br><span class="math">$\phi_0(x_0) = x_0$
    </span>
    <br><span class="math">$\frac{d}{dt}\phi_t(x_0) = u_t(\phi_t(x_0))$
  </dd>
</dl>

流本质是针对许多初始条件的常微分方程解的集合

<img src="flow_process.png" alt="Flow Process" width="80%"/>

Flow 可视化：

<img src="flow.gif" alt="Flow" width="300"/>

> **线性 ODE：**
> 
> 一个简单的向量场：
> $$ u_t(x) = - \theta x $$
>
> 流由常微分方程定义：
> $$ \frac{d}{dt} \phi_t(x) = - \theta \phi_t(x) $$
> $$ \frac{d \phi_t(x)}{\phi_t(x)} = - \theta dt $$
> $$ \int \frac{d \phi_t(x)}{\phi_t(x)} = \int - \theta dt $$
> $$ \log \phi_t(x) = - \theta t + C(x) $$
> $$ \phi_t(x) = e^{-\theta t + C(x)} = e^{C(x)} e^{-\theta t} $$
>
> 用初值条件确定常数 $C(x)$：
> $$ \phi_0(x) = e^{C(x)} e^{0} = e^{C(x)} = x \Rightarrow e^{C(x)} = x \Rightarrow C(x) = \log x $$
>
> 最终流的表达式为：
> $$ \phi_t(x) = x e^{-\theta t} $$
>
> 证明符合流的定义：
> $$ \phi_0(x) = e^{C(x)} e^{0} = e^{C(x)} = x \Rightarrow e^{C(x)} = x $$
> $$ \frac{d}{dt} \phi_t(x) = \frac{d}{dt} (x e^{-\theta t}) = -\theta x e^{-\theta t} = -\theta \phi_t(x) $$
>
> <img src="linear_ode_trajectories.png" alt="Linear ODE Trajectories" width=80%/>

| **算法 1** 利用欧拉方法求解 ODE |
|-------------------------------|
| **输入**: 向量场 $u_t$，初始位置 $x_0$，时间步长 $n$ |
| 1: 设 $t =0$ |
| 2: 设步大小 $h = 1/n$ |
| 3: 设 $X_0 = x_0$ |
| 4: **for** $i = 0$ **to** $n-1$ **do** |
| 5: $~~~~$令 $X_{t+h} = X_t + h \cdot u_{t}(X_t)$ |
| 6: $~~~~$令 $t = t + h$ |
| 7: **end for** |
| **输出**: $X_0, X_h, X_{2h}, \cdots, X_1$ |

#### 2.1.3. 流模型的定义

> $p_{init} \xrightarrow{ODE} p_{data}$

<dl class="definition-list">
  <dt>神经网络</dt>
  <dd>
    <span class="math">$u_t^{\theta}: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$
    </span>
    <br>其中 $\theta$ 代表参数
  </dd>
</dl>

- 随机初始采样： $X_0 \sim p_{init}$
- 模拟 ODE： $X_t = u_t^{\theta}(X_t)$
- 目标： $X_1 \sim p_{data}$

<img src="flow_distribution.gif" alt="Flow Distribution" width="300"/>

| **算法 1** 利用欧拉方法从流模型中采样 |
|-------------------------------|
| **输入**: 神经网络向量场$u_t^{\theta}$，时间步长 $n$ |
| 1: 设 $t =0$ |
| 2: 设步大小 $h = 1/n$ |
| 3: 完成一次采样 $X_0 \sim p_{init}$ |
| 4: **for** $i = 0$ **to** $n-1$ **do** |
| 5: $~~~~$令 $X_{t+h} = X_t + h \cdot u_{t}^{\theta}(X_t)$ |
| 6: $~~~~$令 $t = t + h$ |
| 7: **end for** |
| **输出**: $X_1$ |

---

### 2.2. 扩散模型

<dl class="definition-list">
  <dt>随机过程</dt>
  <dd>
    随机变量<span class="math"> $X_t, 0 \leq t \leq 1$
    </span>
    <br> 随机轨迹<span class="math"> $X: [0, 1] \to \mathbb{R}^d, t \mapsto X_t$
  </dd>
</dl>

<img src="stochastic_traj.png" alt="Stochastic Process" width="300"/>

<dl class="definition-list">
  <dt>向量场</dt>
  <dd>
    <span class="math">$u: R^d \times [0,1] \to \mathbb{R}^d, (x, t) \mapsto u_t(x)$
    </span>
  </dd>
</dl>

<dl class="definition-list">
  <dt>扩散系数</dt>
  <dd>
    <span class="math">$\sigma: [0,1] \to \mathbb{R}, t \mapsto \sigma_t$
    </span>
  </dd>
</dl>

<dl class="definition-list">
  <dt>随机微分方程（Stochastic Differential Equation, SDE）</dt>
  <dd>
    初始条件
    <span class="math">$X_0 = x_0$
    </span>
    <br>
    随机微分方程/动力学方程
    <span class="math">$dX_t = u_t(X_t)dt + \sigma_t dW_t$
    </span>
    <br>其中 $u_t(X_t)dt$ 是 ODE，$\sigma_t dW_t$ 是随机性部分(噪声)，$W_t$ 是标准布朗运动（维纳过程）。
  </dd>
</dl>

#### 2.2.1. 布朗运动

随机过程：
1. $W_0 = 0$
2. 高斯增量：对于任意 $0 \leq s < t \leq 1$，$W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$
3. 独立增量：对于任意 $0 \leq t_0 < t_1 < \cdots < t_n \leq 1$，增量 $W_{t_1} - W_{t_0}, W_{t_2} - W_{t_1}, \cdots, W_{t_n} - W_{t_{n-1}}$ 相互独立

#### 2.2.2. X_t 的解析

ODE 中的 $X_t$ 可解析为：

$$
\frac{d}{dt} X_t = u_t(X_t) \longleftrightarrow X_{t+h} = X_t + h \cdot u_t(X_t) + h \cdot R_t(h)
$$

其中 $\lim_{h \to 0} R_t(h) = 0$，$h \cdot R_t(h)$ 是高阶无穷小，有时记作 $o(h)$。右边的式子可视作泰勒展开。

> 推导过程：
> $$ \frac{d}{dt} X_t = u_t(X_t) $$
> $$ \lim_{h \to 0} \frac{X_{t+h} - X_t}{h} = u_t(X_t) $$
> $$ \frac{X_{t+h} - X_t}{h} = u_t(X_t) + R_t(h) $$
> $$ X_{t+h} = X_t + h \cdot u_t(X_t) + h \cdot R_t(h) $$

SDE 中的 $X_t$ 可解析为：

$$
dX_t = u_t(X_t)dt + \sigma_t dW_t \longleftrightarrow X_{t+h} = X_t + h \cdot u_t(X_t) + \sigma_t (W_{t+h} - W_t) + h \cdot R_t(h),
$$

其中 $\lim_{h \to 0} \sqrt{E[\|R_t(h)\|^2]} = 0$。由于布朗运动不可导，SDE 的展开余项通常用均方收敛（$L^2$）来定义。$W_{t+h} - W_t \sim \mathcal{N}(0, hI_d)$，因此可将其写为 $\sqrt{h} \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I_d)$。

| **算法 2** 利用欧拉-马里亚玛方法从扩散模型中采样，模拟 SDE |
|-------------------------------|
| **输入**: 向量场$u_t$，扩散系数 $\sigma_t$，时间步长 $n$ |
| 1: 设 $t =0$ |
| 2: 设步大小 $h = 1/n$ |
| 3：设 $X_0 = x_0$ |
| 4: **for** $i = 0$ **to** $n-1$ **do** |
| 5: $~~~~$完成一次采样 $\epsilon \in \mathcal{N}(0, I_d)$ |
| 6: $~~~~$令 $X_{t+h} = X_t + h \cdot u_{t}(X_t) + \sigma_t \sqrt{h} \cdot \epsilon$ |
| 7: $~~~~$令 $t = t + h$ |
| 8: **end for** |
| **输出**: $X_0, X_h, X_{2h}, \cdots, X_1$ |

<img src="linear_sde_euler_trajectories.png" alt="Linear SDE Trajectories" width=80%/>

#### 2.2.3. 扩散模型的定义

> $p_{init} \xrightarrow{SDE} p_{data}$

<dl class="definition-list">
  <dt>神经网络</dt>
  <dd>
    <span class="math">$u_t^{\theta}: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$
    </span>
    <br>其中 $\theta$ 代表参数
  </dd>
</dl>

<dl class="definition-list">
  <dt>扩散系数</dt>
  <dd>
    <span class="math">$\sigma_t$
    </span>（通常是固定的）
  </dd>
</dl>

- 随机初始采样： $X_0 \sim p_{init}$
- 模拟 SDE： $dX_t = u_t^{\theta}(X_t)dt + \sigma_t dW_t$
- 目标： $X_1 \sim p_{data}$