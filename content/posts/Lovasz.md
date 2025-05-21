# Lovasz Loss

## Lovasz Loss的推导

IoU (intersection-over-union，也叫jaccard index)是自然图像分割比赛中常用的一个衡量分割效果的评价指标，所以一个自然的想法就是能否将IoU作为loss function来直接优化。交并比公式：

$$
J_c(y^{*},\widetilde{y}) = \frac{\vert \{y^{*}=c\} \cap \{\widetilde{y}=c\}\vert}{\vert \{y^{*}=c\} \cup \{\widetilde{y}=c\}\vert}
$$

其中$y^{*}$表示Ground Truth标签，$\widetilde{y}$表示预测标签，$\vert \cdot \vert$表示集合中的元素个数。可以看出上式的值是介于0到1之间的，由此可以设计出损失函数：

$$
\Delta_{J_c}(y^{*},\widetilde{y})=1-J_c(y^{*},\widetilde{y})
$$

这个损失函数是离散的，无法直接求导，需要对其做**光滑延拓**。

改写一下$\Delta_{J_c}$,

$$
\Delta_{J_c} = 1-J_c(y^{*},\widetilde{y}) = \frac{\vert M_c \vert}{\vert \{y^{*}=c\} \cup M_c \vert} \tag{1}
$$

其中，$M_c(y^{*},\widetilde{y}) = \{y^{*}=c,\widetilde{y}\neq c\} \cup \{y^{*} \neq c,\widetilde{y}=c\}$，$M_c$是损失函数的自变量，它表达网络分割结果与Ground Truth标签不匹配的集合。$M_c$的定义域为$\{0,1\}^p$，即$M_c \in \{0,1\}^p$，$p$表示集合$M_c$中像素的个数。

由于(1)是次模（submodular）函数，故可以对其做**光滑延拓**。


**定义1** 若一个集合函数$\Delta:\{0,1\}^p \rightarrow \mathbb{R}$对于所有的集合$A,B \in \{0,1\}^p$满足

$$
\Delta(A) + \Delta(B) \geq \Delta(A \cup B) + \Delta(A \cap B)
$$

则我们称$\Delta$是**次模函数**。

**定义2** **Lovasz extension** 现存在一集合函数$\Delta:\{0,1\}^p \rightarrow \mathbb{R}$且$\Delta(\pmb{0})=0$，则其Lovasz extension为

$$
\overline{\Delta} = \sum_{i=1}^p m_i g_i(\pmb{m}) \tag{2}
$$

$$
g_i(m) = \Delta(\{\pi_1,\cdots,\pi_i\}) - \Delta(\{\pi_1,\cdots,\pi_{i-1}\})
$$

$\pi$是一个数组，根据元素$\pmb{m}$降序排序。例如，$x_{\pi_1} \geq x_{\pi_2} \geq \cdots \geq x_{\pi_p}$。

此时\overline{\Delta}已经是一个连续、分段线性的函数了，可以直接对误差$m$求导，导数为$g(m)$。

## Lovasz Loss在多类分割中的应用

假设$F_i(c)$表示的是模型最后输出的像素$i$预测为类别$c$的非归一化分数，则可以通过softmax函数将$F_i(c)$归一化得到像素$i$预测为类别$c$的概率：

$$
f_i(c) = \frac{e^{F_i(c)}}{\sum_{c' \in C} e^{F_i(c')}}
$$

那么(2)中的$m_i(c)$可以定义为

$$
m_i(c) = \left\{
\begin{aligned}
& 1-f_i(c) & \text{if}~c=y_i^{*} \\
& f_i(c) & \text{otherwise}
\end{aligned}
\right.
$$

那么损失函数为

$$
loss(\pmb{f}(c)) = \overline{\Delta_{J_c}}(\pmb{m}(c))
$$

考虑到类别平均mIoU的计算方式，最终的损失函数为

$$
loss(\pmb{f}) = \frac{1}{\vert C \vert} \sum_{c \in C} \overline{\Delta_{J_c}}(\pmb{m}(c))
$$

## Lovasz Loss在多类分割中的代码流程

**步骤1** 计算预测结果的误差

``` py
signs = 2. * predictions.float() - 1.
errors = (1. - logits * Variable(signs))
errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
```

这一步得到(2)中的$m_i$。

**步骤2** 计算IoU得分

``` py
gts = gt_sorted.sum()
intersection = gts - gt_sorted.float().cumsum(0)
union = gts + (1 - gt_sorted).float().cumsum(0)
jaccard = 1. - intersection / union
```

这一步得到(1)的值，即IoU得分。

**步骤3** 根据IoU得分计算Lovasz extension的梯度

``` py
jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
```

这一步得到(2)中的$g_i(\pmb{m})$。

**步骤4** 计算Loss

``` py
loss = torch.dot(F.relu(errors_sorted), Variable(grad))
```

这一步得到(2)的值。