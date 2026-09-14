---
title: "Robotics Coordinate Frames: ENU, NED, FLU, and FRD"
date: 2026-09-10T16:00:00+08:00
series:
  main: "Robotics"
  subseries: "Coordinates and Transforms"
categories: ["Robotics"]
tags: ["Coordinate Frames", "ENU", "NED", "FLU", "FRD", "ROS", "PX4"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "An introduction to ENU, NED, FLU, and FRD conventions, including world-frame conversion, body-frame conversion, and yaw rotation."
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
  alt: "ENU, FLU, NED, and FRD coordinate axes around a quadrotor"
  caption: "ROS and PX4 use different world-frame and body-frame conventions"
  relative: true
  hidden: false
  hiddenInList: false
editPost:
  URL: "https://cspaulia.github.io/cspaulia-blog/content/"
  Text: "Suggest Changes"
  appendFilePath: true
---

The hardest part of coordinate transformation is often not the matrix multiplication, but **failing to state which frame each quantity belongs to**. The same three numbers may mean east, north, and up under one convention, but forward, right, and down under another.

A robotics system also distinguishes two kinds of frames:

- A **world frame** is fixed to the environment and describes positions, trajectories, and world-frame velocities.
- A **body frame** is attached to the robot and moves and rotates with it. It describes quantities such as body velocity, thrust, and angular velocity.

## 1. Four Common Frame Conventions

ENU and NED are normally used as world frames, while FLU and FRD are normally used as body frames.

| Frame | (X)-axis | (Y)-axis | (Z)-axis | Typical use |
| --- | --- | --- | --- | --- |
| ENU world | East | North | Up | ROS local world frame |
| NED world | North | East | Down | Aviation and PX4 local world frame |
| FLU body | Forward | Left | Up | ROS body frame |
| FRD body | Forward | Right | Down | PX4 body frame |

[ROS REP-103](https://www.ros.org/reps/rep-0103.html) recommends right-handed frames: a body frame uses (x) forward, (y) left, and (z) up, while short-range geographic representations use ENU. In contrast, the [PX4 ROS 2 interface documentation](https://docs.px4.io/main/en/ros2/user_guide#ros-2-px4-frame-conventions) states that PX4 topics use NED world and FRD body conventions unless a message definition says otherwise.

A ROS–PX4 interface should therefore distinguish three operations:

- Convert world-frame vectors between ENU and NED.
- Convert body-frame vectors between FLU and FRD.
- Convert attitudes with both the world-frame and body-frame conventions in mind; merely rearranging quaternion components is not sufficient.

## 2. World-Frame Conversion: ENU and NED

### 2.1 ENU to NED

Let a vector have the following coordinates in the ENU world frame:

\[
\mathbf v_{ENU}=
\begin{bmatrix}
E\\
N\\
U
\end{bmatrix}.
\]

The coordinates of the same geometric vector in the NED world frame are

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

Intuitively, this swaps the east and north components and changes “up positive” into “down positive.” For example:

```text
ENU [1, 2, 3] → NED [2, 1, -3]
```

### 2.2 NED Back to ENU

The transformation matrix satisfies

\[
\mathbf R_{ENU\rightarrow NED}^{-1}
=\mathbf R_{ENU\rightarrow NED}^{\mathsf T}
=\mathbf R_{ENU\rightarrow NED}.
\]

The inverse transformation therefore uses the same matrix:

\[
\mathbf v_{ENU}
=\mathbf R_{ENU\rightarrow NED}\mathbf v_{NED}.
\]

This gives a convenient unit test: applying the conversion twice to any vector should recover the original value.

## 3. Body-Frame Conversion: FLU and FRD

FLU and FRD share the same forward (X)-axis, while their (Y)- and (Z)-axes point in opposite directions. Therefore,

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

For example, “move left at (2\ \mathrm{m/s}) and climb at (1\ \mathrm{m/s})” in an ROS body frame becomes

```text
FLU [0, 2, 1] → FRD [0, -2, -1]
```

This matrix is also its own inverse, so the same operation converts FRD back to FLU.

> ENU-to-NED and FLU-to-FRD are not the same transformation. The former swaps the horizontal axes and flips the vertical axis; the latter preserves the forward axis and flips the other two.

## 4. Body to World: Yaw Determines the Horizontal Heading

Once the frame conventions are consistent, the attitude can rotate a body-frame vector into the world frame. First consider a yaw-only attitude, ignoring roll and pitch.

Assume that:

- The input vector uses the FRD body frame.
- The output vector uses the NED world frame.
- The yaw angle is \(\psi\), with \(\psi=0\) meaning that the nose points north.
- A turn from north toward east is positive yaw.

Then

\[
\mathbf v_{NED}
=\mathbf R_{body\rightarrow NED}(\psi)\mathbf v_{body},
\]

where

\[
\mathbf R_{body\rightarrow NED}(\psi)
=\begin{bmatrix}
\cos\psi & -\sin\psi & 0\\
\sin\psi & \cos\psi & 0\\
0 & 0 & 1
\end{bmatrix}.
\]

When \(\psi=90^\circ\), the vehicle points east. A unit forward velocity in FRD,

\[
\mathbf v_{body}=
\begin{bmatrix}
1\\0\\0
\end{bmatrix},
\]

becomes

\[
\mathbf v_{NED}=
\begin{bmatrix}
0\\1\\0
\end{bmatrix},
\]

which is eastward motion in NED.

A full three-dimensional attitude also includes roll and pitch. In that case, the Euler-angle rotation order must be declared, or a rotation matrix or quaternion with a clearly specified transformation direction should be used. The names “roll, pitch, and yaw” alone do not uniquely define the rotation.

## 5. ROS–PX4 Integration: Classify the Quantity Before Converting It

A practical workflow is:

1. **Read the message definition**: determine whether the field is a position, velocity, acceleration, thrust, angular velocity, or attitude.
2. **Label the source and target frames**: write \(\mathbf v_{ENU}\rightarrow\mathbf v_{NED}\), rather than an ambiguous \(\mathbf v\).
3. **Separate world and body quantities**: trajectory points are usually converted as world-frame quantities, while body thrust is converted as a body-frame quantity.
4. **Handle attitude last**: an attitude relates two frames, so the reference-frame and body-frame conventions must both be changed.
5. **Test the basis vectors**: check \([1,0,0]^\mathsf T\), \([0,1,0]^\mathsf T\), and \([0,0,1]^\mathsf T\) separately. These tests reveal axis swaps and sign errors more clearly than a random vector does.

A yaw value should not be copied directly either. If \(\psi_{ENU}\) is ROS/ENU yaw, positive from east toward north, and \(\psi_{NED}\) is PX4/NED yaw, positive from north toward east, then under the yaw-only assumption,

\[
\psi_{NED}=\frac{\pi}{2}-\psi_{ENU}.
\]

The result should also be normalized to the angular range expected by the system.
