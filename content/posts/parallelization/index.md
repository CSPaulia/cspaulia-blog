---
title: "并行化"
date: 2026-07-18T10:00:00+08:00
series:
  main: "大语言模型"
  subseries: "系统与硬件"
draft: false
categories: ["大语言模型", "系统"]
tags: ["GPU", "CUDA", "并行计算", "训练"]
author: "CSPaulia"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "CS336 Lecture 7 & 8 学习笔记。"
disableHLJS: true
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
disableShare: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

上一讲讨论的是单个 GPU 内部的并行化；这一讲进一步讨论多个 GPU 之间的并行化。

## 1. 计算与数据的层级

从访问速度来看，常见的硬件层级如下：

- 单节点、单 GPU：L1 缓存（L1 Cache）或共享内存（Shared Memory），速度最快；
- 单节点、单 GPU：高带宽内存（High Bandwidth Memory，HBM）；
- 单节点、多 GPU：通过 NVLink 或 NVSwitch 连接；
- 多节点、多 GPU：通过 InfiniBand 或以太网（Ethernet）连接，速度最慢。

层级越远，数据传输通常越慢，通信成本也越高。单个 GPU 内部可以通过算子融合（Operator Fusion）和分块（Tiling）减少内存访问；扩展到多个 GPU 或多个节点后，则需要通过复制（Replication）和分片（Sharding）减少 GPU 之间的通信。


---

## 参考文献

[1] Stanford CS336, "Lecture 7: Parallelism." [Online]. Available: https://cs336.stanford.edu/lectures/?trace=lecture_07.
