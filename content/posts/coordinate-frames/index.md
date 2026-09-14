---
title: "机器人坐标系：ENU、NED、FLU 与 FRD"
date: 2026-09-10T16:00:00+08:00
series:
  main: "机器人学"
  subseries: "坐标与变换"
categories: ["机器人学"]
tags: ["坐标系", "ENU", "NED", "FLU", "FRD", "ROS", "PX4"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "梳理机器人系统中的 ENU、NED、FLU 与 FRD 坐标约定，以及世界系转换、机体系转换和偏航旋转。"
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
  image: "coordinate-frames-cover.png"
  alt: "无人机周围的 ENU、FLU、NED 与 FRD 坐标轴转换示意图"
  caption: "ROS 与 PX4 使用不同的世界系和机体系约定"
  relative: true
  hidden: false
  hiddenInList: false
editPost:
  URL: "https://cspaulia.github.io/cspaulia-blog/content/"
  Text: "建议修改"
  appendFilePath: true
---

坐标变换最容易出错的地方，往往不是矩阵乘法，而是**没有先说明每个量属于哪个坐标系**。同一个三维数组，在不同约定下可能分别表示向东、向北、向上，也可能表示向前、向右、向下。

机器人系统中还要区分两类坐标系：

- <strong>世界系（world frame）</strong>固定在环境中，用于描述位置、轨迹和世界坐标下的速度。
- <strong>机体系（body frame）</strong>固定在机器人本体上，会随机器人一起平移和旋转，用于描述机体速度、推力和角速度等量。

## 1. 四种常见坐标约定

ENU 与 NED 通常用于世界系，FLU 与 FRD 通常用于机体系。

| 坐标系 | (X) 轴 | (Y) 轴 | (Z) 轴 | 常见用途 |
| --- | --- | --- | --- | --- |
| ENU 世界系 | East（东） | North（北） | Up（上） | ROS 局部世界系 |
| NED 世界系 | North（北） | East（东） | Down（下） | 航空与 PX4 局部世界系 |
| FLU 机体系 | Forward（前） | Left（左） | Up（上） | ROS 机体系 |
| FRD 机体系 | Forward（前） | Right（右） | Down（下） | PX4 机体系 |

[ROS REP-103](https://www.ros.org/reps/rep-0103.html) 推荐采用右手坐标系：机体系取 (x) 向前、(y) 向左、(z) 向上；短距离地理坐标表示采用 ENU。与之不同，[PX4 的 ROS 2 接口文档](https://docs.px4.io/main/en/ros2/user_guide#ros-2-px4-frame-conventions)说明，PX4 消息默认采用 NED 世界系和 FRD 机体系，除非消息定义另有说明。

因此，ROS 与 PX4 对接时不能只写“转换坐标系”，而应明确区分：

- 世界系向量执行 ENU 与 NED 之间的转换；
- 机体系向量执行 FLU 与 FRD 之间的转换；
- 姿态同时涉及世界系和机体系，不能只交换四元数的几个分量。

## 2. 世界系转换：ENU 与 NED

### 2.1 ENU 转为 NED

设一个向量在 ENU 世界系中的坐标为

\[
\mathbf v_{ENU}=
\begin{bmatrix}
E\\
N\\
U
\end{bmatrix}.
\]

同一个几何向量在 NED 世界系中的坐标为

\[
\begin{aligned}
\mathbf v_{NED}
&=\begin{bmatrix}
N\\
E\\
-U
\end{bmatrix} \\
&=\underbrace{
\begin{bmatrix}
0 & 1 & 0\\
1 & 0 & 0\\
0 & 0 & -1
\end{bmatrix}
}_{\mathbf R_{ENU\rightarrow NED}}
\mathbf v_{ENU}.
\end{aligned}
\]

直观上，这一步做了两件事：交换东、北分量，并将“向上为正”改成“向下为正”。例如：

```text
ENU [1, 2, 3] → NED [2, 1, -3]
```

### 2.2 NED 转回 ENU

该变换矩阵满足

\[
\mathbf R_{ENU\rightarrow NED}^{-1}
=\mathbf R_{ENU\rightarrow NED}^{\mathsf T}
=\mathbf R_{ENU\rightarrow NED}.
\]

因此，逆变换使用同一个矩阵：

\[
\mathbf v_{ENU}
=\mathbf R_{ENU\rightarrow NED}\mathbf v_{NED}.
\]

这也提供了一个简单的单元测试：任意向量连续转换两次后，应当回到原值。

## 3. 机体系转换：FLU 与 FRD

FLU 和 FRD 的 (X) 轴都指向前方，区别在于 (Y) 轴和 (Z) 轴的正方向相反。因此

\[
\mathbf v_{FRD}
=\underbrace{
\begin{bmatrix}
1 & 0 & 0\\
0 & -1 & 0\\
0 & 0 & -1
\end{bmatrix}
}_{\mathbf R_{FLU\rightarrow FRD}}
\mathbf v_{FLU}.
\]

例如，ROS 机体系中的“左移 (2\ \mathrm{m/s})、上升 (1\ \mathrm{m/s})”写作

```text
FLU [0, 2, 1] → FRD [0, -2, -1]
```

这个矩阵同样等于自身的逆矩阵，所以 FRD 转 FLU 也使用相同操作。

> ENU 转 NED 与 FLU 转 FRD 不是一回事。前者交换水平轴并翻转竖直轴，后者保留前向轴并翻转另外两轴。

## 4. 机体系转世界系：偏航决定水平朝向

坐标约定统一之后，才轮到用姿态把机体系向量旋转到世界系。先考虑只有偏航角（yaw）、忽略滚转（roll）和俯仰（pitch）的情况。

假设：

- 输入向量采用 FRD 机体系；
- 输出向量采用 NED 世界系；
- 偏航角为 \(\psi\)，\(\psi=0\) 表示机头朝北；
- 从北向东转动为偏航角正方向。

则

\[
\mathbf v_{NED}
=\mathbf R_{body\rightarrow NED}(\psi)\mathbf v_{body},
\]

其中

\[
\mathbf R_{body\rightarrow NED}(\psi)
=\begin{bmatrix}
\cos\psi & -\sin\psi & 0\\
\sin\psi & \cos\psi & 0\\
0 & 0 & 1
\end{bmatrix}.
\]

当 \(\psi=90^\circ\) 时，机头朝东。此时 FRD 机体系中的单位前向速度

\[
\mathbf v_{body}=
\begin{bmatrix}
1\\0\\0
\end{bmatrix}
\]

会变成 NED 世界系中的

\[
\mathbf v_{NED}=
\begin{bmatrix}
0\\1\\0
\end{bmatrix},
\]

即向东运动。

完整三维姿态还需要同时考虑滚转、俯仰和偏航。此时必须先声明欧拉角的旋转顺序，或直接使用带有明确变换方向的旋转矩阵或四元数；仅凭 “roll、pitch、yaw” 三个名称无法唯一确定旋转。

## 5. ROS 与 PX4 对接：先判断量属于哪个坐标系

一个实用的处理顺序是：

1. **读消息定义**：确认字段表达的是位置、速度、加速度、推力、角速度还是姿态。
2. **标注源坐标系和目标坐标系**：例如 \(\mathbf v_{ENU}\rightarrow\mathbf v_{NED}\)，不要只写含义模糊的 \(\mathbf v\)。
3. **区分世界量与机体量**：轨迹点通常按世界系转换，机体推力通常按机体系转换。
4. **最后处理姿态**：姿态描述两个坐标系之间的关系，转换时要同时改变参考系和机体系约定。
5. **用基向量测试**：分别检查 \([1,0,0]^\mathsf T\)、\([0,1,0]^\mathsf T\) 和 \([0,0,1]^\mathsf T\) 的输出，比只测随机向量更容易发现轴交换和符号错误。

对于偏航角本身，也不能直接复制数值。若 \(\psi_{ENU}\) 是 ROS/ENU 下从东向北为正的偏航角，而 \(\psi_{NED}\) 是 PX4/NED 下从北向东为正的偏航角，在只有偏航的前提下有

\[
\psi_{NED}=\frac{\pi}{2}-\psi_{ENU},
\]

结果还应归一化到系统采用的角度区间。
