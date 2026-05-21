---
title: "Mixture of Experts（Mixture of Experts，MoE）"
date: 2026-04-22T10:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "large language model"
    subseries: "Architecture and Training"
categories: ["大语言模型"]
tags: ["架构", "训练"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "Stanford CS336 \"mixture of experts\" Course Notes"
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

<img src="moe_overall.png" alt="Mixture of Experts Architecture Diagram"/>

- Dense FFN: A large FFN that every token runs through.
- MoE FFN: $N$ expert FFNs, each token runs through only one of them.

---

## 1. Pros and Cons of MoE

**Advantages**:

1. The more experts, the better the model performance, while FLOPs remain unchanged. See [image](more_experts.png)[1].
2. When the active parameter count of an MoE model matches that of a dense model, the MoE model trains faster. See [image](moe_train.png)[2].
3. MoE models allow experts to be distributed across different devices, reducing the per-device memory requirement. See [image](moe_parallelization.png).

**Disadvantages**:

1. MoE's sweet spot is "large-scale distributed systems," not "small, simple single-machine training."
2. The training objective and process are heuristic and sometimes unstable, as shown in the following figure[3]:

<img src="moe_training_instabilities.png" alt="Training Instability in Mixture of Experts"/>

---

## 2. MoE Design Approaches

### 2.1. Routing Function

<img src="routing_function.png" alt="Router Function Example"/>

1. **Token-Choice Routing**: Each token selects the expert it most prefers (most common).
2. **Expert-Choice Routing**: Each expert selects the tokens it most prefers.
3. **Optimized Global Routing**: Optimization on top of the former two, considering load balancing, expert utilization, etc.

{{< figure src="routing_type.png" alt="Router Function Types" caption="Compared to Expert Choice, Token Choice is superior in all aspects [2]." >}}

### 2.2. Routing Type

#### 2.2.1. Top-k Routing

The most common routing type in MoE architectures.

{{< figure src="top-k.png" alt="Top-k Routing Example">}}

The k values for different models are shown in the table below:

| Model | k |
| --- | --- |
| Switch Transformer | 1 |
| Gshard | 2 |
| Grok | 2 |
| Mixtral | 2 |
| Qwen | 4 |
| DBRX | 4 |
| DeepSeek | 7 |

The general implementation steps for Top-k routing are as follows:
1. Compute the routing score of each token for each expert:
$$
s\_{i,t} = \mathrm{softmax}_i({\mathbf{u}^l_t}^\top \mathbf{e}^l_i)\_i,
$$
where $l$ is the layer index, $t$ is the token position, and $i$ is the expert index. $\mathbf{u}^l_t$ is the input representation of token $t$ at layer $l$, and $\mathbf{e}^l_i$ is the weight parameter of the $l$-th expert at layer $i$.
2. Select the top-k experts with the highest scores:
$$
g\_{i,t} = \begin{cases}
s\_{i,t} & \text{if } i \in \mathrm{top\_k}(s\_{:,t}) \\\\
0 & \text{otherwise}
\end{cases},
$$
3. Sum the scores of the selected k experts:
$$
\mathbf{h}^l_t = \sum\_{i=1}^N (g\_{i,t} \cdot \mathrm{FFN}_i(\mathbf{u}^l_t)) + \mathbf{u}^l_t,
$$
where $g\_{i,t}$ can be regarded as the gating value.

#### 2.2.2. Hashing Routing

{{< figure src="hashing.png" alt="Hashing Routing Example">}}

1. Obtain a certain id of the token, which could be: token id, token string, etc.
2. Hash it.
3. Take modulo with respect to the number of experts, i.e.:

$$
\mathrm{expert\_{id}} = \mathrm{hash}(\mathrm{token\_{id}}) \mod N
$$

#### 2.2.3. Learning Routing via Reinforcement Learning

Since routing involves assigning tokens to experts, which is a discrete process, reinforcement learning is well-suited for training the routing function.

{{< figure src="rl_to_learn_routes.png" alt="Reinforcement Learning Routing Example">}}

However, training the routing function with reinforcement learning has two drawbacks:
- Unstable training process.
- Low training efficiency.

Therefore, this method is not commonly used.

#### 2.2.4. BASE Routing

It explicitly incorporates "load balancing" into the assignment process itself, rather than first selecting and then adding a balancing loss as a remedy, unlike ordinary top-k routing.

{{< figure src="linear_assignment.png" alt="BASE Routing Example">}}

General steps:
1. Compute routing scores: same as ordinary router, first assign scores using a small gating network.
2. Concatenate the scores of the entire batch of tokens into a matrix.
3. Specify the number of tokens each expert can receive.
4. Solve using some assignment / matching / optimization algorithm.

### 2.3. MoE Variants

{{< figure src="moe_variants.png" alt="MoE Variant Example">}}

The main changes in MoE design language are as follows (assuming the original FFN feature dimension is $d$):

(a) Replicate $N$ FFNs with feature dimension $d$.

(b) Split the feature dimension into $k$ parts, turning the original one FFN into $k$ FFNs with feature dimension $\frac{d}{k}$.

(c) Fix one FFN as a shared expert, and the other $N-1$ FFNs are routing-selected experts.

DeepSeekMoE[4]'s [ablation study](shared_moe_ablation.png) shows that design (c) outperforms design (b).

Olmoe[2]'s [ablation study](finegrained_moe_ablation.png) shows that design (b) outperforms design (a).

The routing configurations for different models are shown in the table below:

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

## 3. Training Techniques for MoE

### 3.1. Methods for Training the Routing Function

#### 3.1.1. Reinforcement Learning

Conclusion: The process of training the routing function with reinforcement learning is unstable, inefficient, and does not yield significant performance improvements, so it is not commonly used, as shown in the [figure](rl_for_routing.png).

#### 3.1.2. Random Perturbation

This method has also not been widely adopted.

##### 3.1.2.1. Method One

Compared to ordinary top-k routing, random perturbation adds noise to the routing scores during training, enabling the model to explore more expert combinations and thus improving generalization[5]. The specific implementation is:

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

##### 3.1.2.2. Method Two

Add a normal distribution perturbation to the routing scores[1]. Let the routing logits be $\mathbf{l}^l_{i,t}$, then:

$$
\mathbf{g}^l_t = \mathrm{softmax}(\mathbf{l}^l_t + \mathcal{N}(0,1) \cdot \sigma)
$$

#### 3.1.3. Load Balancing Loss (Heuristic Balancing Loss)

A widely used technique for training MoE models, aiming to "prevent the router from always sending most tokens to just a few experts." The calculation method for the load balancing loss[1] is as follows:

$$
\mathbf{loss} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i,
$$

where $N$ is the number of experts.

$f_i$ is the proportion of tokens actually received by the $i$-th expert:

$$
f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathrm{1}\\{\arg\max p(x) = i\\} ,
$$

If out of 100 tokens in a batch, 30 are sent to expert 2, then $f_2=0.3$.

$P_i$ is the average probability mass assigned by the router to the $i$-th expert:

$$
P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x),
$$

$\alpha$ is a hyperparameter controlling the weight of the load balancing loss.

> **Why do we need both $f_i$ and $P_i$?**
>
> Because in Switch, token dispatch uses argmax/top-1, a discrete operation that is not easy to backpropagate through directly. So the authors used two quantities:
> - $f_i$: reflects the actual assignment result.
> - $P_i$: reflects the router's probability tendency, which is continuous and trainable.
>
> This constructs an objective that both reflects load and allows the router to be trained.

> The following is personal understanding and may not be entirely accurate:
> There exists a theoretical minimum for this $\mathbf{loss}$. Assume $P_i = f_i$, meaning the router's probability distribution exactly matches the actual assignment. Then 
> $$
> \mathbf{loss} = \alpha \cdot N \cdot \sum_{i=1}^N f_i^2.
> $$ 
> Since $f_i$ is a probability distribution ($\sum_{i=1}^N f_i = 1$), we can set $f(x) = x^2$, and we have $f''(x) = 2 > 0$, so $f(x)$ is a convex function. By Jensen's inequality, we have:
> $$
> \frac{1}{N} \sum_{i=1}^N f_i^2 \geq \left(\frac{1}{N} \sum_{i=1}^N f_i \right)^2 = \frac{1}{N^2},
> $$
> Therefore, the theoretical minimum of $\mathbf{loss}$ is $\alpha$.

##### 3.1.3.1. Load Balancing Loss in DeepSeek v1 and v2

**Per-Expert Balance**

$$
\mathbf{loss} = \alpha_1 \sum_{i=1}^{N'} f_i \cdot P_i,
$$

$$
f_i = \frac{N'}{K'T} \sum_{t=1}^{T} \mathrm{1}\\{\mathrm{Token}\ t \mathrm{selects\ Expert}\ i\\} ,
$$

$$
P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t},
$$

where $N'$ is the number of experts, $K'$ is the number of experts selected per token, $T$ is the total number of tokens in the batch, and $s_{i,t}$ is the routing score of token $t$ for expert $i$.

**Per-Device Balance**

$$
\mathbf{loss} = \alpha_2 \sum_{i=1}^{D} f'\_i \cdot P'\_i,
$$

$$
f'\_i = \frac{1}{|\mathcal{E}\_i|} \sum_{j \in \mathcal{E}\_i} f_j,
$$

$$
P'\_i = \sum_{j \in \mathcal{E}\_i} P_j,
$$

where $D$ is the number of devices, $\mathcal{E}\_i$ is the set of experts on the $i$-th device, and $f_j$ and $P_j$ are the actual allocation proportion and the average probability mass assigned by the router for the $j$-th expert, respectively.

##### 3.1.3.2. Load Balancing in DeepSeek v3

DeepSeek v3 designed an "auxiliary-loss-free balancing" method that directly incorporates the load balancing of top-k routing into the routing score calculation, without needing an additional load balancing loss. The specific implementation is as follows:

$$
\mathbf{g}'\_{i,t} = \begin{cases}
s\_{i,t} & s\_{i,t} + b_i \in \mathrm{top\_k}(\{s\_{j,t} + b\_j|1 \leq j \leq N_r\}, K_r) \\\\
0 & \text{otherwise}
\end{cases},
$$

where $b_i$ is a bias term associated with expert $i$, $N_r$ is the number of experts in the router, and $K_r$ is the number of experts selected per token.

When no token selects expert $i$ for a long time, $b_i$ gradually increases, making $s\_{i,t} + b_i$ more likely to enter the top-k, thus achieving load balancing.

Of course, DeepSeek v3 still adds an expert balance load balancing loss; it is not completely "auxiliary-loss-free."

### 3.2. Methods for Improving Computational Efficiency

#### 3.2.1. Parallel Computation Design

> **Ordinary dense FFN**: All tokens × the same weight matrix.
> **Naive MoE**: Each expert performs its own small matrix multiplication.

**Block Diagonal Matrix Multiplication**: Concatenate all expert weights into a large block diagonal matrix:
$$
\begin{bmatrix}
\mathrm{FFN}_1 & 0 & \cdots & 0 \\\
0 & \mathrm{FFN}_2 & \cdots & 0 \\\
\vdots & \vdots & \ddots & \vdots \\\
0 & 0 & \cdots & \mathrm{FFN}_N
\end{bmatrix}.
$$
The advantage is that all expert computations can be done with a single matrix multiplication. However, this leads to computing a large number of zeros, resulting in low computational and storage efficiency.

**Block Sparse Matrix Multiplication**: Do not explicitly store the entire large matrix; only store and compute the blocks that are actually non-zero.

#### 3.2.2. Latent MoE

{{<figure src="latent_moe.png" alt="Latent MoE Example">}}

Downsample and then upsample the token feature dimension.

#### 3.2.3. Batch Dropping

When all batches tend to select the same expert, drop a portion of the batches, letting them skip the current expert computation and directly complete the residual connection, thereby reducing computation.

### 3.3. Initialization Techniques

**Upcycling**: Use a pre-trained dense model to initialize the expert weights in an MoE model.

{{< figure src="upcycling.png" alt="Upcycling Example">}}

## 4. Issues with MoE

### 4.1. Stability

#### 4.1.1. Routing Training Stability

It has been observed that routing score computation involves the $\mathrm{softmax}$ operation, and $\mathrm{softmax}$ can cause training instability. See [blog](/posts/transformer_in_llm/#模型训练稳定性技巧). Therefore, routing z-loss is introduced to stabilize training.

{{< figure src="router_z-loss.png" alt="Router z-loss Example">}}

#### 4.2. Overfitting During SFT

MoE models may overfit small-scale training data during the SFT phase.

**Solution One**: Only train the FFNs that are not part of the MoE architecture.

**Solution Two**: Use a larger dataset.

---

## References

[1] Fedus W, Zoph B, Shazeer N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity[J]. Journal of Machine Learning Research, 2022, 23(120): 1-39.

[2] Muennighoff N, Soldaini L, Groeneveld D, et al. Olmoe: Open mixture-of-experts language models[J]. arXiv preprint arXiv:2409.02060, 2024.

[3] Zoph B, Bello I, Kumar S, et al. St-moe: Designing stable and transferable sparse expert models[J]. arXiv preprint arXiv:2202.08906, 2022.

[4] Dai D, Deng C, Zhao C, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models[C]//Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024: 1280-1297.

[5] Shazeer N, Mirhoseini A, Maziarz K, et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer[J]. arXiv preprint arXiv:1701.06538, 2017.

[6] Stanford CS336 Course Staff. CS336: Language Modeling from Scratch[EB/OL]. https://cs336.stanford.edu/ [2026-04-23].