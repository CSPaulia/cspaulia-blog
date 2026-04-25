---
title: "混合专家（Mixture of Experts，MoE）"
date: 2026-04-22T10:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "大语言模型"
    subseries: "架构与训练"
categories: ["大语言模型"]
tags: ["架构", "训练"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "斯坦福 CS336 《mixture of experts》课程笔记"
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
    image: "moe_variants.png" # image path/url
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

<img src="moe_overall.png" alt="混合专家架构图"/>

- Dense FFN: 一个大 FFN，每个 token 都要跑它
- MoE FFN: $N$ 个 expert FFN，每个 token 只跑其中 1 个

---

## 1. MoE 的优缺点

**优点**：

1. 专家数量越多，模型性能越优，而 FLOPs 不变，可参考[图片](more_experts.png)[1]；
2. 在 MoE 模型的激活参数量与 Dense 模型一致的情况下， MoE 模型的训练速度更快，可参考[图片](moe_train.png)[2]；
3. MoE 模型可将专家分布在不同的设备上，单卡显存需求降低，可参考[图片](moe_parallelization.png)。

**缺点**：

1. MoE 的甜点区是“大规模分布式系统”，不是“小而简单的单机训练”。
2. 训练目标和训练过程带有启发式，而且有时不稳定，如下图所示[3]：

<img src="moe_training_instabilities.png" alt="混合专家训练不稳定性"/>

---

## 2. MoE 的设计方案

### 2.1. 路由函数（Routing Function）

<img src="routing_function.png" alt="路由函数示例"/>

1. **根据 Token 选择专家**：每个 token 选自己最想去的 expert（最为常见）；
2. **根据专家选择 Token**：每个 expert 选自己最想要的 token；
3. **经过优化的全局路由**：在前两者的基础上进行优化，考虑负载均衡、专家利用率等因素。

{{< figure src="routing_type.png" alt="路由函数类型" caption="相比于专家选择（Expert Choice），Token 选择（Token Choice）全方位更优[2]。" >}}

### 2.2. 路由类型（Routing Type）

#### 2.2.1. Top-k 路由

MoE 架构中最常见的路由类型。

{{< figure src="top-k.png" alt="Top-k 路由示例">}}

不同模型的 k 值如下表所示：

| 模型 | k |
| --- | --- |
| Switch Transformer | 1 |
| Gshard | 2 |
| Grok | 2 |
| Mixtral | 2 |
| Qwen | 4 |
| DBRX | 4 |
| DeepSeek | 7 |

Top-k 路由的一般实现步骤如下：
1. 计算每个 token 对每个专家的路由分数（routing score）：
$$
s\_{i,t} = \mathrm{softmax}_i({\mathbf{u}^l_t}^\top \mathbf{e}^l_i)\_i,
$$
其中，$l$ 为层数，$t$ 为 token 的位置，$i$ 为专家的编号。$\mathbf{u}^l_t$ 是 token $t$ 在第 $l$ 层的输入表示，$\mathbf{e}^l_i$ 是第 $l$ 层第 $i$ 个专家的权重参数。
2. 选出分数最高的 k 个专家：
$$
g\_{i,t} = \begin{cases}
s\_{i,t} & \text{if } i \in \mathrm{top\_k}(s\_{:,t}) \\\\
0 & \text{otherwise}
\end{cases},
$$
3. 对选出的 k 个专家的分数进行求和：
$$
\mathbf{h}^l_t = \sum\_{i=1}^N (g\_{i,t} \cdot \mathrm{FFN}_i(\mathbf{u}^l_t)) + \mathbf{u}^l_t,
$$
其中 $g\_{i,t}$ 可视作门控值（gating value）。

#### 2.2.2. Hashing 路由

{{< figure src="hashing.png" alt="Hashing 路由示例">}}

1. 获得 token 的某个 id，可以是：token id、token 字符串等；
2. 对它做哈希；
3. 对专家数量取模，即：

$$
\mathrm{expert\_{id}} = \mathrm{hash}(\mathrm{token\_{id}}) \mod N
$$

#### 2.2.3. 利用强化学习学习路由

由于路由的过程是给 token 分配专家，此过程为离散过程，很适合使用强化学习训练路由函数。

{{< figure src="rl_to_learn_routes.png" alt="强化学习路由示例">}}

但是，强化学习训练路由函数的过程有两个缺点：
- 训练过程不稳定；
- 训练效率较低。

因此目前该方法并不常用。

#### 2.2.4. BASE 路由

它显式把“负载均衡”放进分配过程本身，而不是像普通 top-k routing 那样，先选完再额外加一个 balancing loss 去补救。

{{< figure src="linear_assignment.png" alt="BASE 路由示例">}}

一般步骤：
1. 计算路由分数（routing score）：和普通 router 一样，先用一个小 gating 网络给分；
2. 把整批 token 的分数拼成一个矩阵；
3. 给每个 expert 规定可接收 token 数；
4. 用某种 assignment / matching / optimization 算法求解。

### 2.3. MoE 的变体

{{< figure src="moe_variants.png" alt="MoE 变体示例">}}

MoE 的设计语言的变化主要如下（假设 FFN 原始的特征维度为 $d$）：

(a) 复制 $N$ 个特征维度为 $d$ 的 FFN；

(b) 将特征维度分成 $k$ 份，则原先一个 FFN 变为 $k$ 个特征维度为 $\frac{d}{k}$ 的 FFN；

(c) 固定一个 FFN 为共享专家，其他 $N-1$ 个 FFN 为路由选择专家；

DeepseekMoE[4] 的[消融实验](shared_moe_ablation.png)表明，(c) 的设计比 (b) 更优。

Olmoe[2] 的[消融实验](finegrained_moe_ablation.png)表明，(b) 的设计比 (a) 更优。

不同模型的路由配置如下表所示：

| Model | Routed | Active | Shared | Fine-grained | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| GShard | 2048 | 2 | 0 | 0 | - |
| Switch Transformer | 64 | 1 | 0 | 0 | - |
| ST-MOE | 64 | 2 | 0 | 0 | - |
| Mixtral | 8 | 2 | 0 | 0 | - |
| DBRX | 16 | 4 | 0 | 0 | - |
| Grok | 8 | 2 | 0 | 0 | - |
| DeepSeek v1 | 64 | 6 | 2 | 1 | 1/4 |
| Qwen 1.5 | 60 | 4 | 4 | 1 | 1/8 |
| DeepSeek v3 | 256 | 8 | 1 | 1 | 1/14 |
| OlMoE | 64 | 8 | 0 | 1 | 1/8 |
| MiniMax | 32 | 2 | 0 | 0 | 1/4 |
| Llama 4 (maverick) | 128 | 1 | 1 | 0 | 1/2 |

## 3. MoE 的训练技巧

### 3.1. 训练路由函数的方法

#### 3.1.1. 强化学习

结论：强化学习训练路由函数的过程不稳定，效率较低，性能没有显著提升，因此目前并不常用，如[图](rl_for_routing.png)所示。

#### 3.1.2. 随机扰动

这个方法也并未广泛采纳。

##### 3.1.2.1. 方法一

相比于普通的 top-k 路由，随机扰动（random perturbation）在训练过程中会给路由分数添加一些噪声，使得模型在训练过程中能够探索更多的专家组合，从而提高模型的泛化能力[5]。具体实现为：

$$
\mathbf{g}^l_t = \mathrm{softmax}(\mathrm{KeepTopK}(\mathbf{H}^l_t, k)),
$$

$$
H^l_{i,t} = (\mathbf{u}^l_t)^\top \mathbf{e}^l_i + \mathcal{N}(0,1) \cdot \mathrm{softplus}\big((\mathbf{u}^l_t)^\top \mathbf{w}^{l}_{\mathrm{noise},i}\big),
$$

$$
\mathrm{KeepTopK}(\mathbf{v}, k)_i =
\begin{cases}
v_i, & \text{if } v_i \text{ is in the top } k \text{ elements of } \mathbf{v}, \\\\
-\infty, & \text{otherwise}.
\end{cases}
$$

##### 3.1.2.2. 方法二

对路由分数添加一个正态分布扰动[1]，设路由逻辑值（logits）为 $\mathbf{l}^l_{i,t}$，则有：

$$
\mathbf{g}^l_t = \mathrm{softmax}(\mathbf{l}^l_t + \mathcal{N}(0,1) \cdot \sigma)
$$

#### 3.1.3. 负载均衡损失（Heuristic Balancing Loss）

一种广泛用于 MoE 模型训练的技巧，旨在“别让路由器总把大部分 token 都送给少数几个 expert”。负载均衡损失的计算方法[1]如下：

$$
\mathbf{loss} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i,
$$

其中，$N$ 是专家的数量。

$f_i$ 是第 $i$ 个专家实际接收到的 token 比例：

$$
f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathrm{1}\\{\arg\max p(x) = i\\} ,
$$

假设一批 100 个 token 里，30 个被送到 expert 2，那 $f_2=0.3$。

$P_i$ 是路由器对第 $i$ 个专家给出的平均概率质量：

$$
P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x),
$$

$\alpha$ 是一个超参数，用于控制负载均衡损失的权重。

> **为什么同时要 $f_i$ 和 $P_i$ 呢？**
>
> 因为在 Switch 里，真正派发 token 用的是 argmax/top-1，这是离散操作，不好直接反传梯度。所以作者用了两个量：
> - $f_i$：反映真实分配结果；
> - $P_i$：反映路由器的概率倾向，它是连续的，可训练。
>
> 这样就能构造一个既反映负载、又能训练路由器的目标。

> 以下是一些个人理解，不一定完全正确：
> 该 $\mathbf{loss}$ 是存在理论最小值的。我们假设 $P_i = f_i$，即路由器的概率分布与实际分配完全一致，那么 
> $$
> \mathbf{loss} = \alpha \cdot N \cdot \sum_{i=1}^N f_i^2.
> $$ 
> 由于 $f_i$ 是一个概率分布（$\sum_{i=1}^N f_i = 1$），我们可以令 $f(x) = x^2$，有 $f''(x) = 2 > 0$，因此 $f(x)$ 是一个凸函数。根据 Jensen 不等式，我们有：
> $$
> \frac{1}{N} \sum_{i=1}^N f_i^2 \geq \left(\frac{1}{N} \sum_{i=1}^N f_i \right)^2 = \frac{1}{N^2},
> $$
> 因此 $\mathbf{loss}$ 的理论最小值为 $\alpha$。

##### 3.1.3.1. DeepSeek v1 和 v2 的负载均衡损失

**专家平衡（Per-Expert Balance）**

$$
\mathbf{loss} = \alpha_1 \sum_{i=1}^{N'} f_i \cdot P_i,
$$

$$
f_i = \frac{N'}{K'T} \sum_{t=1}^{T} \mathrm{1}\\{\mathrm{Token}\ t \mathrm{selects\ Expert}\ i\\} ,
$$

$$
P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t},
$$

其中，$N'$ 是专家的数量，$K'$ 是每个 token 选择的专家数量，$T$ 是批次中 token 的总数，$s_{i,t}$ 是 token $t$ 对专家 $i$ 的路由分数。

**设备平衡（Per-Device Balance）**

$$
\mathbf{loss} = \alpha_2 \sum_{i=1}^{D} f'\_i \cdot P'\_i,
$$

$$
f'\_i = \frac{1}{|\mathcal{E}\_i|} \sum_{j \in \mathcal{E}\_i} f_j,
$$

$$
P'\_i = \sum_{j \in \mathcal{E}\_i} P_j,
$$

其中，$D$ 是设备的数量，$\mathcal{E}\_i$ 是第 $i$ 个设备上专家的集合，$f_j$ 和 $P_j$ 分别是第 $j$ 个专家的实际分配比例和路由器给出的平均概率质量。

##### 3.1.3.2. DeepSeek v3 的负载均衡损失

DeepSeek v3 设计了一种“无需辅助损失平衡（auxiliary-loss-free balancing）”方法，该方法将 top-k 路由的负载均衡直接融入路由分数的计算中，无需额外的负载均衡损失。具体实现如下：

$$
\mathbf{g}'\_{i,t} = \begin{cases}
s\_{i,t} & s\_{i,t} + b_i \in \mathrm{top\_k}(\{s\_{j,t} + b\_j|1 \leq j \leq N_r\}, K_r) \\\\
0 & \text{otherwise}
\end{cases},
$$

其中，$b_i$ 是一个与专家 $i$ 相关的偏置项，$N_r$ 是路由器的专家数量，$K_r$ 是每个 token 选择的专家数量。

当长期没有 token 选择择专家 $i$ 时，$b_i$ 会逐渐增加，使得 $s\_{i,t} + b_i$ 更有可能进入 top-k，从而实现负载均衡。

当然，DeepSeek v3 还是添加专家平衡负载均衡损失，并非完全的“无需辅助损失”。

### 3.2. 计算效率的提升方法

#### 3.2.1. 并行计算设计

> **普通 dense FFN**：所有 token × 同一个权重矩阵
>
> **朴素 MoE**：每个专家各做一次小矩阵乘法

**块对角线矩阵乘法（Block Diagonal Matrix Multiplication）**：把所有专家权重拼成一个大块对角矩阵：
$$
\begin{bmatrix}
\mathrm{FFN}_1 & 0 & \cdots & 0 \\\
0 & \mathrm{FFN}_2 & \cdots & 0 \\\
\vdots & \vdots & \ddots & \vdots \\\
0 & 0 & \cdots & \mathrm{FFN}_N
\end{bmatrix}.
$$
这样做的好处是，只需一次矩阵乘法，就能完成所有的专家计算。然而，这样做会导致计算大量的零值，计算和存储效率低下。

**块稀疏矩阵乘法（Block Sparse Matrix Multiplication）**：不显式存整个大矩阵，只存和计算那些真正非零的块

#### 3.2.2. 潜层 MoE（Latent MoE）

{{<figure src="latent_moe.png" alt="潜层 MoE 示例">}}

对 token 特征维度下采样再上采样。

#### 3.2.3. 批次丢弃（Batch Dropping）

当所有的批次都倾向于使用同一个专家，就丢弃一部分批次，让它们不参与当前的专家计算，直接完成残差连接，从而减少计算量。

### 3.3. 初始化技巧

**upcycling**：用预训练好的 dense 模型来初始化 MoE 模型中的专家权重。

{{< figure src="upcycling.png" alt="upcycling 示例">}}

## 4. MoE 的问题

### 4.1. 稳定性

#### 4.1.1. 路由训练的稳定性

观察到路由分数的计算涉及到 $\mathrm{softmax}$ 操作，$\mathrm{softmax}$ 会引发训练不稳定的问题，见[博客](/posts/transformer_in_llm/#模型训练稳定性技巧)，因此引入了路由 z-loss 来稳定训练过程。

{{< figure src="router_z-loss.png" alt="路由 z-loss 示例">}}

#### 4.2. SFT 阶段的过拟合问题

MoE 模型在 SFT 阶段可能会过拟合规模较小的训练数据。

**解决办法一**：只训练非 MoE 架构的 FFN；

**解决办法二**：采用更大的数据集。



---

## 参考文献

[1] Fedus W, Zoph B, Shazeer N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity[J]. Journal of Machine Learning Research, 2022, 23(120): 1-39.

[2] Muennighoff N, Soldaini L, Groeneveld D, et al. Olmoe: Open mixture-of-experts language models[J]. arXiv preprint arXiv:2409.02060, 2024.

[3] Zoph B, Bello I, Kumar S, et al. St-moe: Designing stable and transferable sparse expert models[J]. arXiv preprint arXiv:2202.08906, 2022.

[4] Dai D, Deng C, Zhao C, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models[C]//Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024: 1280-1297.

[5] Shazeer N, Mirhoseini A, Maziarz K, et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer[J]. arXiv preprint arXiv:1701.06538, 2017.

[6] Stanford CS336 Course Staff. CS336: Language Modeling from Scratch[EB/OL]. https://cs336.stanford.edu/ [2026-04-23].