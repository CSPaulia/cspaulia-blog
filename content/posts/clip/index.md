---
title: "CLIP 及其改进工作"
date: 2025-06-03T10:46:03+08:00
# weight: 1
# aliases: ["/first"]
categories: ["基础模型"]
tags: ["CLIP"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "CLIP 及其改进工作"
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
    image: "clip_cover.png" # image path/url
    alt: "CLIP 及其改进工作封面" # alt text
    caption: "CLIP 及其改进工作" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
    hiddenInList: false # hide on list pages and home
    # class: "post-cover"
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "建议修改" # edit text
    appendFilePath: true # to append file path to Edit link
---

## I. CLIP

<p align="center">
  {{< img src="clip.png" alt="clip" >}}
</p>

1. 对比预训练：
    
    - 从网络上收集N个图片-文本对（OpenAI收集了4亿个图片文本对）作为正样本；
        
    - N张图片与剩下的N-1张图片对应的文本组成数据对，作为负样本（即文本描述和图片内容不符）；
        
    - N张图片送入图像编码器，对应文本送入文本编码器，将图像特征与文本特征做点积，得到相似度矩阵；
        
    - 将矩阵的每一行当作是一个N类预测的结果，以第 i 行为例，为了使第 i 行、第 i 列的值最大（在第 i 行中相似度最大），我们的 label 应该也是 i，将第 i 行与 label 作 cross entropy，即可完成矩阵的优化；
        
    - 将矩阵的每一列当作是一个N类预测的结果，以第 i 列为例，为了使第 i 列、第 i 行的值最大（在第 i 列中相似度最大），我们的 label 应该也是 i，将第 i 列与 label 作 cross entropy，即可完成矩阵的优化。
        
2. 从 label 中构建数据分类器：
    
    - 用文本标签构建句子，送入文本编码器得到文本特征。
        
3. 用于zero-shot预测：
    
    - 将标签构建的文本特征与图像特征进行相似度匹配，从而完成预测。

## II. 利用CLIP做语义分割

### 2.1 LSeg

<p align="center">
  {{< img src="lseg.png" alt="lseg" >}}
</p>

1. 与CLIP的关系：
    
    - 利用已经对齐好的 CLIP 特征空间，将语义标签和像素特征映射到同一空间，通过相似性进行分割预测；
    
    - 文本编码器：与CLIP保持一致，训练时不更新参数:snowflake:；

2. 与CLIP不同的点：

| 方法         | 训练方式   | 文本编码器参数 | 相似度计算 |
|--------------|------------|---------------|------------|
| CLIP         | 对比学习   | 可训练        | 计算图像文本对的相似度 |
| LSeg         | 有监督学习 | 冻结          | 计算图像特征与文本特征之间的相似度 |

- CLIP的输入：多个图像文本对
- LSeg的输入：一张图像+标签（可看作图像的描述文本）

### 2.2 GroupViT

<p align="center">
  {{< img src="groupvit_overview.png" alt="groupvit_overview" width="50%" >}}
</p>

1. 与CLIP和LSeg的关系：

| 方法         | 训练方式   | 相似度计算 |
|--------------|------------|---------------|------------|
| CLIP         | 对比学习   | 计算图像文本对的相似度 |
| LSeg         | 有监督学习 | 计算图像特征与文本特征之间的相似度 |
| GroupViT         | 对比学习 | 计算图像文本对的相似度 |

- LSeg的训练方式：利用已经对齐好的 CLIP 特征空间进行有监督训练；
- GroupViT的训练方式：调整CLIP中视觉编码器的架构，以适应语义分割任务，进行与CLIP一致的对比学习。

2. 架构细节：

<p align="center">
  {{< img src="groupvit.png" alt="groupvit" >}}
</p>

- 模型输入：`图像Patchs` + `可学习Group Tokens`
- 分割流程：通过`Group Block`将`Patch特征`分配给`可学习Group Tokens`
- `可学习Group Tokens`：类似于聚类中心

## III. 利用CLIP做目标检测

### 3.1 ViLD

<p align="center">
  {{< img src="vild_compare.png" alt="vild_compare" >}}
</p>

- `Vanilla Detector` = `Head` + `Classifier` + 交叉熵有监督
- `ViLD-text` = `Head` +  `相似度匹配` + 交叉熵有监督
  - 相似度匹配流程（CLIP流程）：
    - n个标签通过提示词工程送入`文本编码器`得到n个`文本编码`；
    - 为防止`文本编码`无法描述所有`region embeddings`，引入`背景编码`描述剩余的`region embeddings`；
    - `region embeddings`与`文本编码`和`背景编码`做相似度匹配，得到n个`Text Embeddings`和一个`Background`，这一步替代`Classifier`，且在训练中冻结:snowflake:；
- `ViLD-image` = `教师网络` + `学生网络` + L1知识蒸馏
  - `教师网络`：CLIP图像编码器
  - `学生网络`：`Vanilla Detector`
  - 为了减少训练量，利用预训练检测模型提前提取m个`region embeddings`
- `ViLD` = `ViLD-text` + `ViLD-image`

## IV. 利用CLIP做Visual Grounding

### 4.1 GLIP

<p align="center">
  {{< img src="glip.png" alt="glip" >}}
</p>

- 本质为有监督训练；
- 计算`Regions`和`Words`之间的相似度，从而完成`Regions`的分类/caption。
- 模型的训练阶段，必须知道`Regions`和Caption中`Words`之间的对应关系，为此：
  - Detection数据集：利用Bounding Boxes的标注构造Caption（例如Banana-->There is a banana.）
  - Caption数据集：利用在Detection数据集上训练好的GLIP模型在Caption数据集中找到`Regions`和`Words`之间的关系，构造伪标签。

## V. 利用CLIP做图像生成

### 5.1 CLIPasso

<p align="center">
  {{< img src="clipasso.png" alt="clipasso" >}}
</p>

#### 工作出发点

发现问题：以往的简笔画生成方法，仅适用于某一特定类别

解决问题：利用CLIP模型强大的泛化性，完成对所有类别的简笔画生成

#### 工作流程

1. 生成简笔画：
   - 首先利用`图像编码器`，获取图像的热图
   - 根据热图采样点
   - 通过`可学习参数`聚合点，生成贝兹曲线，从而获得简笔画
2. 利用CLIP约束生成：
   - 将生成的简笔画和原图送入两个不同的CLIP图像编码器
   - $L_g$约束两张图的几何信息，越接近越好
   - $L_s$约束两张图的语义信息，越接近越好

## VI. 利用CLIP做视频检索

### 6.1 CLIP4Clip

#### 工作出发点

<p align="center">
  {{< img src="clip4clip.png" alt="clip4clip" >}}
</p>

- CLIP是为图像-文本对设计的模型。对于视频检索任务来说，其本质上是一段文本和多张图像（视频帧）进行匹配，找到最相关的视频帧
- CLIP4Clip探索了三种匹配方法;
  - Parameter-free Type：不需要参数的模块，例如均值池化（没考虑时序）
  - Sequential Type：时序模块，例如LSTM和Transformer
  - Tight Type：使用一个Transformer Encoder共同学习文本特征和视频特征，输出相似度

### 6.2 ActionCLIP

<p align="center">
  {{< img src="actionclip_overview.png" alt="actionclip_overview" width="50%" >}}
</p>

<p align="center">
  {{< img src="actionclip.png" alt="actionclip" >}}
</p>

与CLIP4Clip类似

## VII. 利用CLIP做语音识别

### 7.1 AudioCLIP

<p align="center">
  {{< img src="audioclip.png" alt="audioclip" >}}
</p>

添加音频编码器，仿照CLIP架构构造`音频-图像对比学习`和`音频-文本对比学习`

## VIII. 利用CLIP做三维理解

### 8.1 PointCLIP

<p align="center">
  {{< img src="pointclip.png" alt="pointclip" >}}
</p>

1. 将点云映射到二维空间；
2. 构造Prompt：Point Cloud Depth Map of a [CLASS]

## IX. 利用CLIP做深度估计

### 9.1 DepthCLIP

<p align="center">
  {{< img src="depthclip.png" alt="depthclip" >}}
</p>

- 构建七类文本Prompt："This object is [distance class]"，[distance class]：
  - ’giant’
  - ’extremely close’
  - ’close’
  - ’not in distance’
  - ’a little remote’
  - ’far’
  - ’unseen’
- 网络结构与LSeg类似，通过`Softmax`完成分类预测

---

<div class="zhihu-ref">
  <div class="zhihu-ref-title">参考文献</div>
  <ol>
    <li><a href="https://www.bilibili.com/video/BV1FV4y1p7Lm?spm_id_from=333.788.videopod.sections&vd_source=9e4f1724ef60547fa31e3c8270245ff8" target="_blank">CLIP 改进工作串讲（上）【论文精读·42】</a></li>
    <li><a href="https://www.bilibili.com/video/BV1gg411U7n4?spm_id_from=333.788.videopod.sections&vd_source=9e4f1724ef60547fa31e3c8270245ff8" target="_blank">CLIP 改进工作串讲（下）【论文精读·42】</a></li>
  </ol>
</div>