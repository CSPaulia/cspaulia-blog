# Focal Loss

$$
\text{FL}(p_t) = (1-p_t)^\gamma\log(p_t)
$$

## 背景

Focal Loss是为了解决**样本数量不平衡**而提出的，还强调了样本的**难易性**。

## Balenced Cross Entropy

为了解决**样本数量不平衡**这个问题，我们可以选择给Cross Entropy添加权重。以二分类问题举例，[Cross Entropy & KL Divergence](CE.md)这篇博客已经介绍过Binary Cross Entropy：

$$
\text{L} = \sum_{i=1}^N [y_i\log p + (1-y_i)\log(1-p)]
$$

改写一下，

$$
\text{L}=\left\{
\begin{aligned}
& -log(p) & \text{if}~y=1 \\
& -log(1-p) & \text{otherwise}
\end{aligned}
\right.
$$

再改写一下，

$$
p_t=\left\{
\begin{aligned}
& p & \text{if}~y=1 \\
& 1-p & \text{otherwise}
\end{aligned}
\right.
$$

$$
\text{L} = -log(p_t)
$$

添加权重，

$$
\text{L} = -\alpha_tlog(p_t)
$$

其中$y=1$时$\alpha_t=\alpha$；$y=0$时$\alpha_t=1-\alpha$。$\frac{\alpha}{1-\alpha}=\frac{n}{m}$，$n$为$y=0$的样本（负样本）个数，$m$为$y=1$的样本（正样本）个数。

## 样本难易问题

Balenced Cross Entropy确实解决了样本不均衡问题，但并未解决样本难易问题。

![easyhard](Focal/easyhard.jpg)

## Focal Loss

$$
\text{FL}(p_t) = (1-p_t)^\gamma\log(p_t)
$$

$p_t$是模型预测的结果的类别概率值。$−\log(p_t)$和交叉熵损失函数一致，因此当前样本类别对应的那个$p_t$如果越小，说明预测越不准确，那么$(1-p_t)^{\gamma}$这一项就会增大，这一项也作为困难样本的系数，预测越不准，Focal Loss越倾向于把这个样本当作困难样本，这个系数也就越大，目的是让困难样本对损失和梯度的贡献更大。

![easyhard](Focal/Focal_exp.png)

前面的$\alpha_t$是类别权重系数。如果你有一个类别不平衡的数据集，那么你肯定想对数量少的那一类在loss贡献上赋予一个高权重，这个$\alpha_t$就起到这样的作用。因此，$\alpha_t$应该是一个**向量**，向量的长度等于类别的个数，用于存放各个类别的权重。一般来说$\alpha_t$中的值为**每一个类别样本数量的倒数**，相当于平衡样本的数量差距
