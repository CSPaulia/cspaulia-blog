---
title: "记录 100 种评价指标"
date: 2025-07-09T14:05:03+08:00
# weight: 1
# aliases: ["/first"]
categories: ["深度学习技巧"]
tags: ["评价指标"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "[Epoch  10/100] Updating..."
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
    image: "metric_cover.png" # image path/url
    alt: "评价指标概览" # alt text
    caption: "评价指标" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "修改建议" # edit text
    appendFilePath: true # to append file path to Edit link
---

## 分类评估指标

> TP / TN / FP / FN 基本概念
>
> 假设我们做的是“检测是否生病”的任务，模型预测结果 vs 实际情况如下表：
>
> | 实际情况\预测结果   | 预测为阳性（生病） | 预测为阴性（没病） |
> |------------------|------------------|------------------|
> | 实际是阳性（生病） | ✅ TP（真正）       | ❌ FN（假负）       |
> | 实际是阴性（没病） | ❌ FP（假正）       | ✅ TN（真负）       |
>
> - **TP (True Positive)**：预测“有病”，实际也“有病” → 判断正确  
> - **TN (True Negative)**：预测“没病”，实际也“没病” → 判断正确  
> - **FP (False Positive)**：预测“有病”，实际却“没病” → 误报  
> - **FN (False Negative)**：预测“没病”，实际却“有病” → 漏报（更严重）

---

### 准确率（Accuracy）

- 模型判断正确的总体比例

$$
Accuracy = \frac{(TP + TN)}{(TP + TN + FP + FN)}
$$

---

### 召回率（Recall）

- 在所有真正例中，模型识别出来的比例（不漏掉）

$$
Recall = \frac{TP}{(TP + FN)}
$$

---

### 精确率（Precision）

- 模型预测为正中，确实为正的比例（不冤枉）

$$
Precision = \frac{TP}{(TP + FP)}
$$

---

### F1-score

- 精确率和召回率的调和平均

$$
F1 = \frac{2 × (Precision × Recall)}{(Precision + Recall)}
$$

#### 多分类变体

- **Macro-F1**：所有类别的 F1 取**算术平均**。
- **Micro-F1**：基于全局总 TP/FP/FN 计算 F1。
- **Weighted-F1**：按各类别的**样本数量加权**平均 F1。

---

## 排序质量评估指标

### 平均准确率（Average Precision，AP）

**一句话解释：一个类别的平均“精确率”**，以某一类别的目标检测为例：

#### 1. 对预测框按置信分数从高到低排序

每个预测结果都有：

- 位置框（bounding box）
- 分数（confidence）
- 预测标签

按预测分数从高到低排序

#### 2. 判断每个预测是 TP 还是 FP

依次遍历预测框：

- 如果与某个**尚未匹配的真实框** IoU ≥ 阈值（如 0.5），则为 TP
- 否则为 FP

记录每个预测的 TP/FP 标记

#### 3. 计算累计 Precision & Recall

遍历每个预测点，从第一个开始，每预测一个都更新：

- 累积 TP 数（TP_i）
- 累积 FP 数（FP_i）

然后算：

```python
precision_i = TP_i / (TP_i + FP_i)
recall_i = TP_i / GT_total  # GT_total 为总的真实框数量
```

#### 4. 画 PR 曲线 & 计算面积

逐步积分:

```python
# Recall, Precision 已按 recall 升序排列
AP = 0
for i in range(1, len(recall)):
    AP += precision[i] * (recall[i] - recall[i-1])
```

---

### mAP（mean Average Precision）

所有类别的 AP 平均值

$$
mAP = \sum_i AP_i
$$

---

### mAP\@IoU=0.5（mAP\@0.5）

- 预测的框和真实框的 IoU（交并比）≥ 0.5 时，才算正确（TP）
- 然后计算每个类的 AP，再平均，就是 mAP\@0.5

---

### mAP\@0.5:0.95

- IoU 从 0.5 到 0.95，每隔 0.05 计算一次（共 10 个 IoU 阈值）
- 平均这 10 个 AP → 得到最终 mAP

---

### mAP\@k

常用于 **图像检索 / 推荐系统 / 多标签排序**：

> 表示在每次检索的前 **k 个结果** 中，计算 AP，然后对所有查询求平均

#### 举例

你搜索“狗”，模型返回前10张图片：

- 有6张是狗，4张不是，且狗分布在第1、2、3、6、7、9位
- 你计算这些位置上的 Precision，再求平均 → 得到 AP@10
- 对所有用户查询求平均 → 得到 mAP@10

---

## Caption任务指标

### SPICE

#### 为什么需要 SPICE？

传统指标如：

- **BLEU**：关注 n-gram 匹配（像机器翻译）
- **ROUGE**：关注召回率（适合摘要）
- **CIDEr**：考虑 TF-IDF 加权的 n-gram 匹配

它们都看的是“词”和“短语”重不重复，却忽略了语义结构是否对。

而 SPICE 的理念是：

> “人类评价图像描述时，看的是你有没有说出对的对象、属性和关系。”

#### SPICE 计算方法

> **核心思想：** 把句子转换为一个语义图（scene graph）：对象 + 属性 + 关系，然后比较机器描述和参考描述的语义图有多相似

##### 语义图结构定义

一句话被表示为一组三元组（triples）：

```
G = {object, attribute, relation}
```

例如：

```
Sentence: "A red car is parked beside a white house"
G = {
  (object: car),
  (object: house),
  (attribute: car, red),
  (attribute: house, white),
  (relation: car, beside, house)
}
```

##### 数学公式定义

**输入：**

- $G$: 生成句子的语义三元组集合（Graph of candidate）
- $R$: 所有参考句子的语义三元组集合（Graph of references）

**目标：**

- 计算 $F_1$ score between $G$ and $R$

**匹配集合：**

- Precision:

$$
P = \frac{|\mathcal{G} \cap \mathcal{R}|}{|\mathcal{G}|}
$$
​
 
- Recall:

$$
R = \frac{|\mathcal{G} \cap \mathcal{R}|}{|\mathcal{R}|}
$$
 
- F1-score:

$$
F_1 = \frac{2PR}{P + R}
$$
 
即：$\text{SPICE} = F_1(G, R)$

分别统计：

- 对象匹配（object F1）

- 属性匹配（attribute F1）

- 关系匹配（relation F1）

- 甚至是颜色、数量、大小等细分类别

最终 SPICE 得分是它们的加权平均：

$$
\text{SPICE}=\sum_i w_i \cdot F_1^i
$$
 
其中 $F_1^i$ 是每种语义类别（对象、属性等）的 F1，$w_i$ 是每类的权重

---