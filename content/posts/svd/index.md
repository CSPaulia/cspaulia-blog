---
title: "奇异值分解（SVD）"
date: 2026-08-17T15:00:00+08:00
series:
    main: "线性变换基础"
    subseries: "矩阵分解"
categories: ["数学基础"]
tags: ["线性代数", "矩阵分解", "SVD"]
author: "CSPaulia"
math: true
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "从线性变换的角度理解奇异值分解，以及左右奇异向量和奇异值分别表示什么。"
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
    image: "cover.png"
    alt: "SVD 将单位圆依次经过 V 转置、奇异值矩阵 Sigma 和 U 变换为椭圆的几何示意图"
    caption: "SVD 的几何过程：Vᵀ 改变坐标方向，Σ 沿奇异方向缩放，U 映射到输出空间。图源：[知乎专栏](https://zhuanlan.zhihu.com/p/342922980)"
    relative: true
    hidden: false
    hiddenInList: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

奇异值分解（Singular Value Decomposition，SVD）是一种矩阵分解方法。它能够把任意矩阵表示成三个结构简单的矩阵：

\[
A=U\Sigma V^\top.
\]

## 1. 把矩阵看成线性变换

设

\[
A\in\mathbb{R}^{m\times n},
\qquad
x\in\mathbb{R}^{n},
\]

那么

\[
y=Ax
\]

表示矩阵 \(A\) 把一个 \(n\) 维输入映射成 \(m\) 维输出。这个变换可能同时包含旋转、反射和沿不同方向的缩放，因此直接观察 \(A\) 通常不容易看出它的几何作用。

SVD 将这个复杂过程拆成三步：

1. \(V^\top\)：把输入换到一组特殊的正交坐标轴上；
2. \(\Sigma\)：分别缩放每一个坐标轴；
3. \(U\)：把缩放后的结果换到输出空间的坐标轴上。

<figure>
  <img src="svd-transform.svg" alt="SVD 的三个步骤：V 转置变换、奇异值缩放和 U 变换">
  <figcaption>从右向左理解 \(A=U\Sigma V^\top\)：先由 \(V^\top\) 改变输入坐标，再由 \(\Sigma\) 缩放，最后由 \(U\) 映射到输出空间。</figcaption>
</figure>

## 2. 奇异向量和奇异值是什么

SVD 可以写成

\[
A=\sum_{i=1}^{r}\sigma_i u_i v_i^\top,
\]

其中 \(r=\operatorname{rank}(A)\)，并且

\[
\sigma_1\geq\sigma_2\geq\cdots\geq\sigma_r>0.
\]

这里：

- \(v_i\) 是第 \(i\) 个右奇异向量，表示输入空间中的一个方向；
- \(u_i\) 是第 \(i\) 个左奇异向量，表示相应的输出方向；
- \(\sigma_i\) 是第 \(i\) 个奇异值，表示该方向被放大的程度。

三者最直观的关系是

\[
Av_i=\sigma_i u_i.
\]

也就是说，当输入正好沿着 \(v_i\) 时，矩阵会将它映射到 \(u_i\) 的方向，并将长度放大 \(\sigma_i\) 倍。

例如，矩阵

\[
A=
\begin{bmatrix}
3&0\\
0&1
\end{bmatrix}
\]

会把水平方向放大 3 倍，竖直方向保持不变。它的两个奇异值就是 \(3\) 和 \(1\)。一般矩阵只是额外在缩放前后加入了坐标变换。

## 3. SVD 与特征值分解的关系

特征值分解通常要求矩阵是方阵，SVD 则适用于任意矩形矩阵。

右奇异向量是 \(A^\top A\) 的特征向量：

\[
A^\top A v_i=\sigma_i^2v_i.
\]

因此，\(A^\top A\) 的特征值是 \(\sigma_i^2\)，奇异值是这些特征值的非负平方根。得到 \(v_i\) 后，可以计算

\[
u_i=\frac{Av_i}{\sigma_i}.
\]

这也说明奇异值一定是非负数，而左右奇异向量分别描述输入空间和输出空间。

## 4. SVD 的计算过程

对于任意实矩阵

\[
A\in\mathbb{R}^{m\times n},
\]

都存在奇异值分解

\[
A=U\Sigma V^\top.
\]

它不要求 \(A\) 是方阵，也不要求 \(A\) 可逆。矩形矩阵、秩亏矩阵和零矩阵都可以做 SVD；如果矩阵的秩不足，只会出现一些值为 \(0\) 的奇异值。复数矩阵也存在 SVD，只需把普通转置 \(V^\top\) 换成共轭转置 \(V^*\)。

在数学上，可以通过下面的步骤计算 SVD。

### 4.1 计算 \(A^\top A\)

首先构造

\[
A^\top A\in\mathbb{R}^{n\times n}.
\]

它一定是对称半正定矩阵，因此特征值都是非负数，并且存在一组正交特征向量。

### 4.2 求右奇异向量和奇异值

对 \(A^\top A\) 做特征值分解：

\[
A^\top A v_i=\lambda_i v_i.
\]

将特征值从大到小排列。右奇异向量就是单位特征向量 \(v_i\)，奇异值为

\[
\sigma_i=\sqrt{\lambda_i}.
\]

把所有 \(v_i\) 按列排列，就得到矩阵 \(V\)；把奇异值放在对角线上，就得到 \(\Sigma\)。

### 4.3 求左奇异向量

对于每个非零奇异值 \(\sigma_i\)，利用

\[
u_i=\frac{Av_i}{\sigma_i}
\]

计算左奇异向量。把这些 \(u_i\) 按列排列，就得到 \(U\) 中与非零奇异值对应的部分。

如果需要完整 SVD，还要为零奇异值补充与已有 \(u_i\) 正交的单位向量，使 \(U\) 构成输出空间的一组完整正交基。

### 4.4 组成分解结果

最后将三个矩阵组合起来：

\[
A=U\Sigma V^\top.
\]

整个过程可以概括为

```text
A
→ 计算 AᵀA
→ 求 AᵀA 的特征值和特征向量
→ σᵢ = √λᵢ，得到 Σ 和 V
→ uᵢ = Avᵢ / σᵢ，得到 U
→ A = UΣVᵀ
```

<details>
<summary><strong>一个简单的计算例子</strong></summary>

考虑矩阵

\[
A=
\begin{bmatrix}
1&1\\
0&0
\end{bmatrix}.
\]

首先计算

\[
A^\top A=
\begin{bmatrix}
1&1\\
1&1
\end{bmatrix}.
\]

它的特征值为 \(\lambda_1=2\) 和 \(\lambda_2=0\)，对应的单位特征向量可以取为

\[
v_1=\frac{1}{\sqrt{2}}
\begin{bmatrix}1\\1\end{bmatrix},
\qquad
v_2=\frac{1}{\sqrt{2}}
\begin{bmatrix}1\\-1\end{bmatrix}.
\]

因此奇异值为

\[
\sigma_1=\sqrt{2},
\qquad
\sigma_2=0.
\]

对于非零奇异值，左奇异向量为

\[
\begin{aligned}
u_1
&=\frac{Av_1}{\sigma_1}\\
&=\begin{bmatrix}1\\0\end{bmatrix}.
\end{aligned}
\]

再选择一个与 \(u_1\) 正交的单位向量

\[
u_2=
\begin{bmatrix}0\\1\end{bmatrix},
\]

最终得到

\[
U=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
\Sigma=
\begin{bmatrix}
\sqrt{2}&0\\
0&0
\end{bmatrix},
\]

\[
V=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1&1\\
1&-1
\end{bmatrix}.
\]

将它们相乘，可以还原原矩阵：

\[
\begin{aligned}
U\Sigma V^\top
&=\begin{bmatrix}
1&1\\
0&0
\end{bmatrix}\\
&=A.
\end{aligned}
\]

</details>

### 4.5 实际程序通常不会这样计算

通过 \(A^\top A\) 推导 SVD 很适合帮助理解，但数值计算库通常不会先显式构造 \(A^\top A\)。这是因为

\[
\kappa(A^\top A)=\kappa(A)^2,
\]

其中 \(\kappa\) 是矩阵的条件数。显式计算 \(A^\top A\) 会放大数值误差，尤其容易影响较小奇异值。

实际实现通常先把矩阵化为双对角矩阵，再使用 QR 迭代、分治法或其他稳定算法求奇异值。调用 NumPy、PyTorch 等库时，这些细节由底层线性代数库完成。

## 5. 完整 SVD 与紧致 SVD

对于

\[
A\in\mathbb{R}^{m\times n},
\]

完整 SVD 中各矩阵的形状为

\[
U\in\mathbb{R}^{m\times m},
\qquad
\Sigma\in\mathbb{R}^{m\times n},
\qquad
V\in\mathbb{R}^{n\times n}.
\]

\(U\) 和 \(V\) 都是正交矩阵：

\[
U^\top U=I,
\qquad
V^\top V=I.
\]

如果 \(A\) 的秩为 \(r\)，那么只有 \(r\) 个奇异值非零。去掉只与零奇异值对应的列，可以写成紧致奇异值分解（Compact SVD）：

\[
A=U_r\Sigma_rV_r^\top,
\]

其中

\[
U_r\in\mathbb{R}^{m\times r},
\qquad
\Sigma_r\in\mathbb{R}^{r\times r},
\qquad
V_r\in\mathbb{R}^{n\times r}.
\]

紧致 SVD 仍然能够精确还原 \(A\)。它只是省略与零奇异值对应的冗余部分，没有改变矩阵表示的信息。

## 6. 小结

- 任意矩阵都可以分解为 \(A=U\Sigma V^\top\)。
- \(Av_i=\sigma_i u_i\)：\(v_i\) 是输入方向，\(u_i\) 是输出方向，\(\sigma_i\) 是缩放强度。
- 紧致 SVD 只省略零奇异值对应的部分，仍能精确还原原矩阵。

## 参考文献

[1] G. H. Golub and W. Kahan. Calculating the Singular Values and Pseudo-Inverse of a Matrix. [Online]. Available: https://epubs.siam.org/doi/10.1137/0702016

[2] G. H. Golub and C. F. Van Loan. Matrix Computations. [Online]. Available: https://jhupbooks.press.jhu.edu/title/matrix-computations
