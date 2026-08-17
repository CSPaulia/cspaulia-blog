---
title: "收集 N 种优化器"
date: 2026-08-07T16:00:00+08:00
series:
    main: "深度学习基础"
    subseries: "优化器"
categories: ["深度学习技巧"]
tags: ["优化器", "SGD", "AdamW", "Lion", "Muon"]
author: "CSPaulia"
math: true
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "从梯度下降、动量与自适应学习率出发，理解 SGD、AdaGrad、RMSProp、Adam、AdamW、Lion 和 Muon。"
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
cover:
    image: "cover.svg"
    alt: "SGD、AdamW 与 Muon 优化器示意图"
    caption: "从逐元素更新到矩阵级更新"
    relative: true
    hidden: false
    hiddenInList: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

训练神经网络，就是不断根据损失函数的梯度更新参数。优化器决定了两个问题：**沿哪个方向走，以及每一步走多远**。

记第 \(t\) 步的参数为 \(\theta_t\)，学习率为 \(\eta\)，当前梯度为

\[
g_t=\nabla_\theta \mathcal{L}_t(\theta_{t-1}).
\]

下面从最基础的随机梯度下降讲起，再逐步加入动量、自适应缩放和矩阵结构。

## 1. 随机梯度下降（Stochastic Gradient Descent，SGD）

随机梯度下降使用一个小批量样本估计完整数据集的梯度，并直接更新参数：

\[
\theta_t=\theta_{t-1}-\eta g_t.
\]

SGD 结构简单、显存开销低，但同一个学习率会作用于所有参数。损失曲面的不同方向若尺度差异很大，更新容易在陡峭方向来回震荡，在平缓方向前进缓慢。

### 1.1 动量法（Momentum）

动量法对历史梯度做指数移动平均：

\[
m_t=\beta m_{t-1}+(1-\beta)g_t,
\]

\[
\theta_t=\theta_{t-1}-\eta m_t.
\]

它会削弱方向反复变化的噪声，并累积方向一致的更新。直观上，SGD 只看当前一步，Momentum 还保留了过去一段时间的“惯性”。

## 2. AdaGrad：累计历史梯度

自适应梯度算法（Adaptive Gradient Algorithm，AdaGrad）为每个参数维护累计平方梯度：

\[
v_t=v_{t-1}+g_t^2,
\]

\[
\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}.
\]

梯度经常较大的参数会获得更小的有效学习率，稀疏或很少更新的参数则能保留较大的步长。这对稀疏特征有帮助，但 \(v_t\) 只增不减，训练后期的有效学习率可能过早趋近于零。

## 3. RMSProp：只关注近期尺度

均方根传播（Root Mean Square Propagation，RMSProp）不再累加全部历史，而是维护平方梯度的指数移动平均：

\[
v_t=\rho v_{t-1}+(1-\rho)g_t^2,
\]

\[
\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}.
\]

旧梯度会逐渐淡出，因此 RMSProp 避免了 AdaGrad 学习率持续衰减的问题。它仍然是**逐元素**调整：每个参数根据自己的近期梯度尺度决定有效步长。

## 4. Adam：动量与自适应学习率结合

自适应矩估计（Adaptive Moment Estimation，Adam）同时维护梯度的一阶矩与二阶矩：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

由于两个移动平均都从零开始，训练初期需要做偏差修正：

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
\]

最终更新为

\[
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

Adam 的一阶矩提供动量，二阶矩则按参数缩放更新。它通常比 SGD 更容易得到可用的训练配置，因此广泛用于 Transformer 和大语言模型训练。

## 5. AdamW：把权重衰减从梯度中解耦

在 Adam 中直接加入 \(L_2\) 正则项，会让正则梯度也经过二阶矩缩放，因此不再等价于通常意义上的权重衰减。AdamW 将两部分更新分开：

\[
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},
\]

其中 \(\lambda\) 是权重衰减系数。这样，参数收缩不再依赖其梯度历史，学习率与权重衰减也更容易分别调节。

在现代大语言模型中，AdamW 是最常见的基线优化器之一。不过，它需要为每个参数保存一阶矩和二阶矩，优化器状态通常带来可观的显存开销。

## 6. Lion：基于符号的动量更新

Lion 根据动量方向的符号更新参数，而不像 Adam 那样使用二阶矩缩放梯度。忽略学习率调度后，其核心更新为 [7]：

\[
\begin{aligned}
c_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
\theta_t&=\theta_{t-1}-\eta_t\left(\operatorname{sign}(c_t)+\lambda\theta_{t-1}\right),\\
m_t&=\beta_2m_{t-1}+(1-\beta_2)g_t.
\end{aligned}
\]

由于 \(\operatorname{sign}(c_t)\) 的每个坐标通常为 \(-1\) 或 \(+1\)，每个参数在单步中的更新幅度主要由学习率 \(\eta_t\) 决定。Lion 只需要保存一个动量状态，优化器状态少于 Adam；但符号操作会丢弃梯度幅值信息，因此学习率和权重衰减需要谨慎设置。

## 7. Muon：对矩阵更新方向做正交化

Muon 的名称来自 MomentUm Orthogonalized by Newton-Schulz。它面向隐藏层中的二维矩阵参数：先像 Momentum 一样累积梯度，再对整个更新矩阵做近似正交化。

理解下面的内容需要知道奇异值和奇异向量分别表示什么，可以先阅读[奇异值分解（SVD）](../svd/)。

<figure>
  <img src="muon-algorithm.png" alt="Muon 优化器算法：动量累积、Newton-Schulz 正交化与参数更新">
  <figcaption>Muon 的基本步骤：形成动量矩阵 \(B_t\)，通过 Newton-Schulz 迭代得到更新方向 \(O_t\)，再更新参数。</figcaption>
</figure>

设矩阵参数在第 \(t\) 步的梯度为 \(G_t\)，Muon 首先计算

\[
B_t=\mu B_{t-1}+G_t.
\]

### 7.1 为什么梯度本身也是一个矩阵

考虑线性层

\[
y=Wx.
\]

权重 \(W\) 是矩阵，它的梯度 \(G=\nabla_W\mathcal{L}\) 和更新量 \(\Delta W\) 也具有相同形状。更新权重以后，层输出的变化为

\[
\Delta y=\Delta W x.
\]

因此，\(\Delta W\) 不只是一组彼此无关的数字。它本身也是一个线性变换，决定了不同输入方向会引起多大的输出变化。

损失函数在当前位置的一阶近似为

\[
\mathcal{L}(W+\Delta W)
\approx
\mathcal{L}(W)+\langle G,\Delta W\rangle_F,
\]

<details>
<summary><strong>为什么一阶近似可以写成这个形式？</strong></summary>

因为 \(W\) 是矩阵，需要对其中的所有元素同时进行一阶泰勒展开。令

\[
G=\nabla_W\mathcal{L}(W),
\]

其中 \(G_{ij}=\frac{\partial\mathcal{L}}{\partial W_{ij}}\)，则

\[
\mathcal{L}(W+\Delta W)
\approx
\mathcal{L}(W)+
\sum_{i,j}G_{ij}\Delta W_{ij}.
\]

两个同形矩阵的弗罗贝尼乌斯内积（Frobenius Inner Product）定义为

\[
\begin{aligned}
\langle G,\Delta W\rangle_F
&=\sum_{i,j}G_{ij}\Delta W_{ij}\\
&=\operatorname{tr}(G^\top\Delta W).
\end{aligned}
\]

这里的 \(\operatorname{tr}\) 表示矩阵的迹（trace）：先计算 \(G^\top\Delta W\)，再把结果矩阵从左上到右下的主对角线元素相加。它恰好等于 \(G\) 与 \(\Delta W\) 对应元素乘积之和。

因此，\(\langle G,\Delta W\rangle_F\) 衡量更新方向与梯度方向的一致程度。内积为负时，一阶近似下损失下降。例如，梯度下降令 \(\Delta W=-\eta G\)，于是

\[
\begin{aligned}
\langle G,\Delta W\rangle_F
&=-\eta\lVert G\rVert_F^2\leq 0.
\end{aligned}
\]

该近似忽略了二阶及更高阶项，只有在 \(\Delta W\) 足够小时才可靠。

</details>

优化器的任务，就是根据梯度 \(G\) 构造一个能让这个内积为负的更新矩阵 \(\Delta W\)。

### 7.2 SVD 如何解释梯度矩阵

对动量矩阵 \(B_t\) 做奇异值分解：

\[
B_t=U\Sigma V^\top,
\]

也可以展开成

\[
B_t=\sum_i\sigma_i u_i v_i^\top.
\]

这里，\(v_i\) 表示输入空间中的一个方向，\(u_i\) 表示对应的输出方向，\(\sigma_i\) 表示这个方向在当前动量更新中的强度：

\[
B_tv_i=\sigma_i u_i.
\]

如果直接使用 Momentum 更新 \(\Delta W=-\eta B_t\)，那么沿 \(v_i\) 的输入会产生

\[
\Delta Wv_i=-\eta\sigma_i u_i.
\]

当最大的奇异值远大于其余奇异值时，更新会被少数方向主导；奇异值很小的方向几乎得不到更新。

### 7.3 为什么 \(UV^\top\) 可以作为更新方向

Muon 将 \(B_t\) 概念上变换为

\[
O_t=UV^\top
=\sum_i u_i v_i^\top.
\]

它保留 \(B_t\) 的左右奇异向量，但把非零奇异值从 \(\sigma_i\) 调整到 \(1\)：

\[
O_tv_i=u_i.
\]

因此，Muon 没有随意创造新的方向，也没有通过 SVD 做降维。它保留原更新矩阵中的奇异方向，只是削弱特别强的方向，并相对放大原本很弱的方向。

先忽略动量并令 \(B_t=G_t\)。梯度和正交化结果的内积为

\[
\langle G_t,U V^\top\rangle_F
=\operatorname{tr}(\Sigma)
=\sum_i\sigma_i>0.
\]

如果选择

\[
\Delta W=-\eta UV^\top,
\]

那么一阶损失变化为

\[
\mathcal{L}(W+\Delta W)-\mathcal{L}(W)
\approx
-\eta\sum_i\sigma_i<0.
\]

所以 \(-UV^\top\) 确实是一个下降方向，而不是与梯度无关的矩阵。加入 Momentum 后，\(B_t\) 不再等于当前梯度，但它是近期梯度的平滑方向；这与普通 Momentum 不保证每一步都严格下降是同一个问题。

### 7.4 谱范数视角：限制一次更新的最大影响

矩阵的谱范数（Spectral Norm）满足

\[
\begin{aligned}
\|\Delta W\|_2
&=\max_{\|x\|_2=1}\|\Delta Wx\|_2.
\end{aligned}
\]

它衡量权重更新对任意单位输入所能造成的最大输出变化。如果要求 \(\|\Delta W\|_2\leq\eta\)，同时希望一阶损失下降得尽可能多，那么问题可以写成

\[
\underset{\|\Delta W\|_2\leq\eta}{\arg\min}
\langle G,\Delta W\rangle_F.
\]

这个问题的一个解正是

\[
\Delta W=-\eta UV^\top.
\]

因此，\(UV^\top\) 可以理解为**谱范数约束下的最速下降方向**。它的谱范数为 \(1\)，使一次更新对所有输入方向的最大作用受到统一控制，同时充分利用梯度提供的全部奇异方向。

当更新矩阵满秩但不是方阵时，\(UV^\top\) 称为半正交矩阵：根据矩阵的形状，它满足 \((UV^\top)^\top(UV^\top)=I\) 或 \((UV^\top)(UV^\top)^\top=I\)。这里的“正交化”描述的是更新矩阵，不是把模型权重强制变成正交矩阵。

### 7.5 为什么使用 Newton-Schulz 而不是直接计算 SVD

从数学上看，显式计算 SVD 后直接构造 \(UV^\top\) 最容易理解，但每个训练步骤都这样做会很慢。Muon 改用 Newton-Schulz 迭代，通过矩阵乘法近似同一个结果。

一次迭代可以写成

\[
X_{k+1}
=aX_k+b(X_kX_k^\top)X_k
+c(X_kX_k^\top)^2X_k.
\]

若 \(X_k=U\Sigma_kV^\top\)，这次迭代不会改变 \(U\) 和 \(V\)，只会把每个奇异值 \(s\) 变换为

\[
\phi(s)=as+bs^3+cs^5.
\]

在迭代前先归一化矩阵，再选择合适的 \(a,b,c\)，重复数次后便能把不同奇异值推向接近 \(1\)。Muon 通常执行 5 次迭代，因此这一步常写作 `NewtonSchulz5`。它得到的是近似正交化结果，而不是精确 SVD。

### 7.6 Muon 与 AdamW 的关键差别

AdamW 根据每个参数自己的历史平方梯度进行**逐元素缩放**；Muon 则把一个权重矩阵看作整体，根据它的行、列结构调整更新方向。

Muon 也不是直接替代模型中的全部优化器：常见实现只把隐藏层的二维权重矩阵交给 Muon，嵌入、输出头、偏置和归一化参数仍交给 AdamW。

### 7.7 Muon 的代价与限制

- Newton-Schulz 迭代会增加矩阵乘法和通信开销。
- 需要按参数形状把 Muon 与 AdamW 分组使用，训练代码更复杂。
- 低精度、大规模分布式训练中的数值稳定性需要额外处理。
- 最佳学习率、动量和缩放规则仍与模型规模及训练配方有关。

## 8. 如何选择

| 优化器 | 核心机制 | 主要优点 | 需要注意 |
|---|---|---|---|
| SGD | 当前梯度 | 简单、状态少 | 对学习率和损失曲面尺度敏感 |
| SGD + Momentum | 梯度的一阶移动平均 | 降低震荡，加速一致方向 | 仍使用统一尺度的更新 |
| AdaGrad | 累计平方梯度 | 适合稀疏特征 | 有效学习率可能过早衰减 |
| RMSProp | 近期平方梯度平均 | 能适应非平稳梯度尺度 | 缺少 Adam 的一阶矩组合 |
| Adam | 一阶矩 + 二阶矩 | 容易训练，适用范围广 | 状态显存较大；权重衰减需谨慎 |
| AdamW | Adam + 解耦权重衰减 | Transformer 的常用基线 | 仍是逐元素预条件 |
| Lion | 动量 + 符号更新 | 状态少于 Adam，更新规则简单 | 丢弃梯度幅值；学习率较敏感 |
| Muon | 动量矩阵 + 近似正交化 | 利用二维权重的矩阵结构 | 参数分组、计算与分布式实现更复杂 |

如果只是建立可靠基线，通常先选择 AdamW；若显存充足且训练配方成熟，再比较 Muon。优化器不能脱离学习率、调度器、批大小、权重衰减与梯度裁剪单独评价。

## 参考文献

[1] J. Duchi, E. Hazan, and Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. [Online]. Available: https://jmlr.org/papers/v12/duchi11a.html

[2] T. Tieleman and G. Hinton. Lecture 6.5—RMSProp. [Online]. Available: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

[3] D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. [Online]. Available: https://arxiv.org/abs/1412.6980

[4] I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. [Online]. Available: https://arxiv.org/abs/1711.05101

[5] K. Jordan et al. Muon: An optimizer for hidden layers in neural networks. [Online]. Available: https://kellerjordan.github.io/posts/muon/

[6] J. Bernstein and L. Newhouse. Old Optimizer, New Norm: An Anthology. [Online]. Available: https://arxiv.org/abs/2409.20325

[7] Xiangning Chen et al. Symbolic Discovery of Optimization Algorithms. [Online]. Available: https://arxiv.org/abs/2302.06675
