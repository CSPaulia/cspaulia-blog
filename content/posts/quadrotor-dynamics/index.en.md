---
title: "Quadrotor Dynamics and Control: From Flight Architecture to a Six-DOF Model"
date: 2026-09-10T17:30:00+08:00
series:
  main: "Robotics"
  subseries: "Dynamics and Control"
categories: ["Robotics"]
tags: ["Quadrotor", "Dynamics", "Attitude Motion", "Coordinate Frames", "Newton-Euler Equations", "Flight Control", "Paper Reading"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Develop a six-degree-of-freedom quadrotor model from the flight architecture, coordinate frames, Newton-Euler equations, and rotor torques, then connect it to attitude and altitude control."
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
disableAnchoredHeadings: true
cover:
  image: "quadrotor-control-subsystems.png"
  alt: "Connection between the quadrotor angular and translational subsystems"
  caption: "The angular subsystem changes attitude, while attitude and total thrust jointly determine translation"
  relative: true
editPost:
  URL: "https://cspaulia.github.io/cspaulia-blog/content/"
  Text: "Suggest Changes"
  appendFilePath: true
---

This article refers to [Bouabdallah et al.'s indoor micro-quadrotor study](https://doi.org/10.1109/ROBOT.2004.1302409).

## 1. Quadrotor Architecture: Four Rotors Control Lift and Attitude

### 1.1 Cross Layout: Two Rotor Pairs Spin in Opposite Directions

A quadrotor consists of a central body, four arms, and four motor–rotor groups mounted at the arm tips in a cross configuration. Opposite rotors form two pairs: rotors 1 and 3 form one pair, while rotors 2 and 4 form the other.

The two pairs spin in opposite directions. Their reaction torques can therefore cancel during ordinary flight, preventing the body from continuously yawing when no yaw command is applied.

The thrust of rotor \(i\) is approximated by

\[
T_i=b\Omega_i^2,
\]

where \(T_i\) is rotor thrust, \(\Omega_i\) is angular speed, and \(b\) is the thrust coefficient. The controller changes the four \(\Omega_i\) values to control total thrust and the three body torques indirectly.

### 1.2 Four Motion Modes: Different Speed Combinations Produce Different Effects

<figure>
  <img src="../../../posts/quadrotor-dynamics/quadrotor-motion-modes.png" alt="A quadrotor changes four rotor speeds to yaw left or right, climb, and move laterally">
  <figcaption>Basic quadrotor motions. A wider arrow indicates a higher rotor speed. Source: Bouabdallah et al., <a href="https://doi.org/10.1109/ROBOT.2004.1302409"><em>Design and Control of an Indoor Micro Quadrotor</em></a>, Fig. 2.</figcaption>
</figure>

The central idea is that a quadrotor does not use aerodynamic control surfaces to steer. Instead, it coordinates increases and decreases in the four rotor speeds to produce vertical, roll, pitch, and yaw motion. The combinations are summarized below:

| Rotor-speed change | Primary motion |
| --- | --- |
| Increase or decrease all four together | Change total lift and vertical motion |
| Change rotors 2 and 4 oppositely | Produce roll, followed by lateral motion |
| Change rotors 1 and 3 oppositely | Produce pitch, followed by longitudinal motion |
| Change the reaction-torque difference between the counter-rotating groups | Produce yaw |

Roll and pitch do not directly change position. They first tilt the body, which tilts total thrust and creates a horizontal force component.

### 1.3 Underactuation: Four Inputs Control Six Degrees of Freedom

A quadrotor has six spatial degrees of freedom:

\[
(x,y,z,\phi,\theta,\psi),
\]

but only four independent control channels: total thrust, roll torque, pitch torque, and yaw torque. It is therefore an <strong>underactuated</strong> system and cannot independently specify the acceleration of all six degrees of freedom at the same instant.

The architecture avoids the mechanically complex rotor system of a conventional helicopter and offers useful payload capacity. The trade-offs are high energy consumption and open-loop dynamic instability, which requires continuous feedback control.

## 2. Coordinate Frames and Attitude: Separate Earth Motion from Body Direction

Only two coordinate frames are needed for the following model:

- **Earth frame \(E\)**: fixed to the environment and used for position \(\zeta=[x,y,z]^T\) and linear velocity \(\nu\).
- **Body frame \(B\)**: fixed to the quadrotor and used for body angular velocity \(\omega\), rotor thrust, and body torque.

<figure>
  <img src="../../../posts/quadrotor-dynamics/quadrotor-coordinate-frames.png" alt="Earth frame E, body frame B, four rotor thrusts, and gravity on a quadrotor">
  <figcaption>Quadrotor frames and forces: \(E\) is the earth-fixed inertial frame, \(B\) is the body-fixed frame, \(F_1\)–\(F_4\) are the four rotor thrusts, and \(mg\) is gravity. Source: Bouabdallah et al., <a href="https://doi.org/10.1109/ROBOT.2004.1302409"><em>Design and Control of an Indoor Micro Quadrotor</em></a>, Fig. 3.</figcaption>
</figure>

Assume that the body-frame origin coincides with the center of mass. The rotation matrix \(R\) maps a body-frame vector into the earth frame:

\[
\mathbf v_E=R\mathbf v_B.
\]

Let

\[
e_3=\begin{bmatrix}0&0&1\end{bmatrix}^{T}.
\]

Then \(e_3\) is the body \(z\)-axis direction, while \(R e_3\) is that direction expressed in the earth frame.

## 3. Rigid-Body Dynamics: From Newton–Euler Equations to State Evolution

### 3.1 Newton–Euler Equations: One Model for Translation and Rotation

In the body frame, rigid-body dynamics can be written as

\[
\begin{aligned}
m\dot V+\omega\times(mV)&=F,\\
I\dot\omega+\omega\times(I\omega)&=\tau.
\end{aligned}
\]

Here:

- \(m\): body mass;
- \(V\): body-frame linear velocity;
- \(\omega\): body angular velocity;
- \(I\): inertia matrix about the center of mass;
- \(F\): net external force;
- \(\tau\): net external torque.

The inertia matrix is the rotational counterpart of mass. If the body axes are principal axes, it normally takes the diagonal form

\[
I=
\begin{bmatrix}
I_x&0&0\\
0&I_y&0\\
0&0&I_z
\end{bmatrix}.
\]

Larger \(I_x,I_y,I_z\) values make angular acceleration about the corresponding axes harder to produce.

<details>
<summary>Mathematical supplement: expand the definitions of cross and dot products</summary>

#### Cross Product: Perpendicular Direction and Rotational Coupling

For two three-dimensional vectors

\[
\mathbf a=
\begin{bmatrix}a_x&a_y&a_z\end{bmatrix}^{T},
\qquad
\mathbf b=
\begin{bmatrix}b_x&b_y&b_z\end{bmatrix}^{T},
\]

the determinant expression of the cross product is

\[
\begin{aligned}
\mathbf a\times\mathbf b
&=
\begin{vmatrix}
\mathbf e_x&\mathbf e_y&\mathbf e_z\\
a_x&a_y&a_z\\
b_x&b_y&b_z
\end{vmatrix}\\
&=
\begin{bmatrix}
a_yb_z-a_zb_y\\
a_zb_x-a_xb_z\\
a_xb_y-a_yb_x
\end{bmatrix}.
\end{aligned}
\]

The result is another vector. It is perpendicular to the plane containing \(\mathbf a\) and \(\mathbf b\), its direction follows the right-hand rule, and its magnitude is

\[
\lVert\mathbf a\times\mathbf b\rVert
=\lVert\mathbf a\rVert\lVert\mathbf b\rVert\sin\theta.
\]

Two properties are especially useful:

\[
\mathbf a\times\mathbf b=-\mathbf b\times\mathbf a,
\qquad
\mathbf a\times\mathbf a=0.
\]

In rigid-body dynamics, a cross product often describes how rotation changes the direction of another vector. The term \(\omega\times(mV)\) is the effect of frame rotation on the linear-momentum derivative, while \(\omega\times(I\omega)\) is the coupling between angular velocity and angular momentum.

Their dimensions are

\[
\begin{aligned}
[\omega\times(mV)]&=\mathrm N,\\
[\omega\times(I\omega)]&=\mathrm{N\,m}.
\end{aligned}
\]

#### Dot Product: A Supplementary Projection Operation

The dot product produces a scalar:

\[
\begin{aligned}
\mathbf a\cdot\mathbf b
&=a_xb_x+a_yb_y+a_zb_z\\
&=\lVert\mathbf a\rVert\lVert\mathbf b\rVert\cos\theta.
\end{aligned}
\]

The dot product is zero when the vectors are perpendicular. It describes the projection of one vector onto another and is also used to compute power:

\[
P=\mathbf F\cdot\mathbf v.
\]

The dot product is not the central operation in the current dynamics, but it clarifies the geometric distinction: a dot product measures directional alignment, while a cross product captures a perpendicular direction and rotational tendency.

</details>

### 3.2 Derivatives in a Rotating Frame: Cross Terms Come from Rotating Axes

Newton's second law uses the momentum derivative in the inertial frame:

\[
\mathbf F=\left(\frac{d(mV)}{dt}\right)_E.
\]

The body axes rotate with angular velocity \(\omega\). Over a short interval \(dt\), any body-axis vector \(\mathbf e\) changes approximately by

\[
d\mathbf e=(\omega\,dt)\times\mathbf e.
\]

Therefore,

\[
\dot{\mathbf e}=\omega\times\mathbf e.
\]

For any vector \(\mathbf a\), derivatives in the inertial and body frames satisfy

\[
\begin{aligned}
\left(\frac{d\mathbf a}{dt}\right)_E
&=
\left(\frac{d\mathbf a}{dt}\right)_B
+\omega\times\mathbf a.
\end{aligned}
\]

Setting \(\mathbf a=mV\) gives

\[
\mathbf F=m\dot V+\omega\times(mV).
\]

Thus, \(\omega\times(mV)\) is the derivative correction introduced by the rotating body frame, not an additional external force acting on the vehicle.

### 3.3 Rigid-Body State Evolution: Connect Position, Attitude, and Their Rates

Introducing position, earth-frame velocity, and the rotation matrix gives

\[
\begin{aligned}
\dot\zeta&=\nu,\\
m\dot\nu&=R F_b,\\
\dot R&=R\hat\omega,\\
J\dot\omega&=-\omega\times(J\omega)+\tau_a.
\end{aligned}
\]

**Position Kinematics: Position Rate Equals Velocity**

\[
\dot\zeta=\nu.
\]

The vector \(\zeta=[x,y,z]^T\) is earth-frame position, and \(\nu=[\dot x,\dot y,\dot z]^T\) is the corresponding linear velocity. This line is purely kinematic and does not involve force or mass.

**Translational Dynamics: Transform the Force Before Applying Newton's Law**

\[
m\dot\nu=R F_b.
\]

The force \(F_b\) is expressed in the body frame, while \(\dot\nu\) is expressed in the earth frame. The matrix \(R\) therefore transforms \(F_b\) into the earth frame before Newton's second law is applied.

**Attitude Kinematics: Angular Velocity Determines Rotation-Matrix Rate**

\[
\dot R=R\hat\omega.
\]

If \(\omega=[p,q,r]^T\), the corresponding skew-symmetric matrix is

\[
\hat\omega=
\begin{bmatrix}
0&-r&q\\
r&0&-p\\
-q&p&0
\end{bmatrix},
\qquad
\hat\omega\mathbf v=\omega\times\mathbf v.
\]

The columns of \(R\) are the three body-axis directions expressed in the earth frame. Combining the derivative rule for all three axes gives \(\dot R=R\hat\omega\).

**Rotational Dynamics: Torque Changes Angular Momentum**

\[
J\dot\omega=-\omega\times(J\omega)+\tau_a.
\]

The term \(J\dot\omega\) is the inertia-weighted angular acceleration, \(-\omega\times(J\omega)\) is rigid-body rotational coupling, and \(\tau_a\) is the net rotor torque acting on the body. Here \(J\) denotes body inertia; the next model uses \(I\) for the same type of physical quantity, while \(J_r\) specifically denotes rotor inertia.

## 4. Quadrotor Dynamics: From Rotor Inputs to Six-DOF Motion

### 4.1 Force Model: Add Gravity, Rotor Thrust, and Gyroscopic Effects

Substituting the quadrotor's physical forces into the rigid-body model gives

\[
\begin{aligned}
\dot\zeta&=\nu,\\
\dot\nu&=-g e_3+R e_3\left(\frac{b}{m}\sum_{i=1}^{4}\Omega_i^2\right),\\
\dot R&=R\hat\omega,\\
I\dot\omega&=-\omega\times(I\omega)
-\sum_{i=1}^{4}J_r(\omega\times e_3)\Omega_i+\tau_a.
\end{aligned}
\]

**Position Kinematics: Retain the Earth-Frame Position Definition**

\[
\dot\zeta=\nu.
\]

This line is unchanged from the general rigid-body model. Rotor thrust does not change position directly: it first changes velocity, and velocity then integrates into position.

**Translational Dynamics: Gravity and Thrust Form the Total Acceleration**

\[
\dot\nu=-g e_3+R e_3\left(\frac{b}{m}\sum_{i=1}^{4}\Omega_i^2\right).
\]

The first term, \(-g e_3\), is gravitational acceleration in the earth frame. The second is thrust acceleration: every rotor produces \(b\Omega_i^2\), the four thrusts are summed and divided by mass \(m\), and \(R e_3\) gives the body \(z\)-axis direction in the earth frame.

At level hover, \(R e_3=e_3\) and acceleration is zero, so

\[
b\sum_{i=1}^{4}\Omega_i^2=mg.
\]

**Attitude Kinematics: Attitude Still Integrates Angular Velocity**

\[
\dot R=R\hat\omega.
\]

Making the force model more specific does not change the attitude kinematics. Angular velocity states how quickly the body is rotating now; integrating this equation gives its attitude over time.

**Rotational Dynamics: Body Coupling and Rotor Gyroscopic Effects**

\[
I\dot\omega=-\omega\times(I\omega)
-\sum_{i=1}^{4}J_r(\omega\times e_3)\Omega_i+\tau_a.
\]

The right-hand side contains three torque contributions:

1. \(-\omega\times(I\omega)\): rigid-body coupling between angular velocity and angular momentum.
2. \(-\sum J_r(\omega\times e_3)\Omega_i\): the gyroscopic reaction caused when body motion changes the direction of rotor angular momentum.
3. \(\tau_a\): the commanded body torque generated by thrust differences and rotor reaction torques.

Here \(J_r\) is the rotational inertia of one rotor. The sign of \(\Omega_i\) must encode the corresponding rotor's spin direction in the gyroscopic sum.

### 4.2 Control Inputs: Combine Four Rotor Speeds into Thrust and Body Torques

Under the rotor numbering and sign convention used here, the total rotor torque is

\[
\tau_a=
\begin{bmatrix}
lb(\Omega_4^2-\Omega_2^2)\\
lb(\Omega_3^2-\Omega_1^2)\\
d(\Omega_2^2+\Omega_4^2-\Omega_1^2-\Omega_3^2)
\end{bmatrix}.
\]

**Roll and Pitch: Thrust Difference Times Lever Arm**

The thrust difference between rotors 2 and 4 produces roll torque:

\[
\tau_x=lT_4-lT_2=lb(\Omega_4^2-\Omega_2^2).
\]

The thrust difference between rotors 1 and 3 produces pitch torque:

\[
\tau_y=lT_3-lT_1=lb(\Omega_3^2-\Omega_1^2).
\]

Both terms follow from \(\tau=\mathbf r\times\mathbf F\), where \(l\) is the lever-arm distance from a rotor to the body center.

**Yaw: Difference Between the Two Rotor Groups' Reaction Torques**

Yaw comes from the aerodynamic reaction torques of the rotors. The two groups spin in opposite directions, so the net yaw torque is

\[
\tau_z=d(\Omega_2^2+\Omega_4^2-\Omega_1^2-\Omega_3^2),
\]

where \(d\) is the drag-torque coefficient. There is no factor \(l\) because yaw torque is not generated by an offset thrust force.

For compact notation, define four control inputs:

\[
\begin{aligned}
U_1&=b(\Omega_1^2+\Omega_2^2+\Omega_3^2+\Omega_4^2),\\
U_2&=b(\Omega_4^2-\Omega_2^2),\\
U_3&=b(\Omega_3^2-\Omega_1^2),\\
U_4&=d(\Omega_2^2+\Omega_4^2-\Omega_1^2-\Omega_3^2).
\end{aligned}
\]

Here \(U_1\) is total thrust, \(lU_2\) and \(lU_3\) are roll and pitch torque, and \(U_4\) is yaw torque.

### 4.3 Six-DOF Equations: Attitude Projection Connects Rotation and Translation

Using roll \(\phi\), pitch \(\theta\), and yaw \(\psi\) to represent attitude, the model expands to

\[
\begin{aligned}
\ddot{x}&=\left(\cos\phi\sin\theta\cos\psi+\sin\phi\sin\psi\right)\frac{U_1}{m},\\
\ddot{y}&=\left(\cos\phi\sin\theta\sin\psi-\sin\phi\cos\psi\right)\frac{U_1}{m},\\
\ddot{z}&=-g+\left(\cos\phi\cos\theta\right)\frac{U_1}{m},\\
\ddot\phi&=\dot\theta\dot\psi\frac{I_y-I_z}{I_x}
-\frac{J_r}{I_x}\dot\theta\Omega+\frac{l}{I_x}U_2,\\
\ddot\theta&=\dot\phi\dot\psi\frac{I_z-I_x}{I_y}
+\frac{J_r}{I_y}\dot\phi\Omega+\frac{l}{I_y}U_3,\\
\ddot\psi&=\dot\phi\dot\theta\frac{I_x-I_y}{I_z}
+\frac{1}{I_z}U_4.
\end{aligned}
\]

The first three lines are earth-frame translational accelerations. Their trigonometric factors are the projections of the thrust direction \(R e_3\) onto the \(x,y,z\) axes. The final three lines are angular accelerations containing rigid-body inertia coupling, rotor gyroscopic effects, and control torque.

<details>
<summary>Expand the derivation of the \(x\)-direction equation</summary>

The translational equation is

\[
\dot\nu=-g e_3+\frac{U_1}{m}R e_3.
\]

Use the Euler-angle rotation order

\[
R=R_z(\psi)R_y(\theta)R_x(\phi),
\]

where

\[
R_x(\phi)=
\begin{bmatrix}
1&0&0\\
0&\cos\phi&-\sin\phi\\
0&\sin\phi&\cos\phi
\end{bmatrix},
\]

\[
R_y(\theta)=
\begin{bmatrix}
\cos\theta&0&\sin\theta\\
0&1&0\\
-\sin\theta&0&\cos\theta
\end{bmatrix},
\]

\[
R_z(\psi)=
\begin{bmatrix}
\cos\psi&-\sin\psi&0\\
\sin\psi&\cos\psi&0\\
0&0&1
\end{bmatrix}.
\]

Because \(e_3=[0,0,1]^T\), only the third column of \(R\) is needed:

\[
R e_3=
\begin{bmatrix}
\cos\phi\sin\theta\cos\psi+\sin\phi\sin\psi\\
\cos\phi\sin\theta\sin\psi-\sin\phi\cos\psi\\
\cos\phi\cos\theta
\end{bmatrix}.
\]

Since

\[
\dot\nu=\begin{bmatrix}\ddot x&\ddot y&\ddot z\end{bmatrix}^{T},
\]

and gravity has no \(x\)-component, taking the first component of \(R e_3\) gives the horizontal equation.

</details>

<details>
<summary>Expand the derivation of the roll-angle \(\phi\) equation</summary>

Start from the rotational dynamics introduced in Section 4.1:

\[
I\dot\omega=-\omega\times(I\omega)
-J_r(\omega\times e_3)\Omega+\tau_a,
\]

where the signed combined rotor speed is

\[
\Omega=\Omega_2+\Omega_4-\Omega_1-\Omega_3.
\]

Choose the body axes as principal inertia axes and write

\[
I=\operatorname{diag}(I_x,I_y,I_z),
\qquad
\omega=\begin{bmatrix}p&q&r\end{bmatrix}^{T}.
\]

First compute the rigid-body coupling term:

\[
I\omega=
\begin{bmatrix}
I_xp\\
I_yq\\
I_zr
\end{bmatrix},
\]

\[
\omega\times(I\omega)=
\begin{bmatrix}
(I_z-I_y)qr\\
(I_x-I_z)pr\\
(I_y-I_x)pq
\end{bmatrix}.
\]

Its body-\(x\) component is therefore

\[
\left[-\omega\times(I\omega)\right]_x
=(I_y-I_z)qr.
\]

Next consider the rotor gyroscopic term. Since

\[
e_3=\begin{bmatrix}0&0&1\end{bmatrix}^{T},
\qquad
\omega\times e_3=
\begin{bmatrix}q&-p&0\end{bmatrix}^{T},
\]

its body-\(x\) component is

\[
\left[-J_r(\omega\times e_3)\Omega\right]_x
=-J_rq\Omega.
\]

From the control-input definition in Section 4.2, the roll control torque is

\[
[\tau_a]_x=lb(\Omega_4^2-\Omega_2^2)=lU_2.
\]

Combining the three body-\(x\) components gives

\[
I_x\dot p=(I_y-I_z)qr-J_rq\Omega+lU_2.
\]

The paper's expanded equations directly associate the body-rate components \(p,q,r\) with the Euler-angle rates. This is a first-order approximation near hover and at small attitude angles:

\[
p\approx\dot\phi,
\qquad q\approx\dot\theta,
\qquad r\approx\dot\psi,
\]

we obtain

\[
I_x\ddot\phi
=(I_y-I_z)\dot\theta\dot\psi
-J_r\dot\theta\Omega+lU_2.
\]

Finally, divide by \(I_x\):

\[
\ddot\phi
=\dot\theta\dot\psi\frac{I_y-I_z}{I_x}
-\frac{J_r}{I_x}\dot\theta\Omega
+\frac{l}{I_x}U_2.
\]

The three terms represent rigid-body inertia coupling, the rotor gyroscopic effect, and roll control torque. Strictly speaking, away from hover, \([p,q,r]^T\) is not identical to the Euler-angle rate vector and the two must be related through the Euler-rate transformation matrix.

</details>

### 4.4 Hover Approximation: Coupled Dynamics Reduce to Four Intuitive Channels

Near level hover, let

\[
\phi=\theta=0,
\qquad
\dot\phi=\dot\theta=\dot\psi=0.
\]

The six-DOF model reduces to

\[
\begin{aligned}
\ddot x&=0,\\
\ddot y&=0,\\
\ddot z&=-g+\frac{U_1}{m},\\
\ddot\phi&=\frac{l}{I_x}U_2,\\
\ddot\theta&=\frac{l}{I_y}U_3,\\
\ddot\psi&=\frac{1}{I_z}U_4.
\end{aligned}
\]

The four control inputs can then be interpreted as:

- \(U_1\) mainly controls vertical motion.
- \(U_2\) mainly controls roll.
- \(U_3\) mainly controls pitch.
- \(U_4\) mainly controls yaw.

This is only a local interpretation near hover. As attitude angles and angular rates grow, coupling between translation, rotation, and the three rotational axes becomes significant again.

## 5. Quadrotor Control: Stabilize Attitude Before Altitude

The six-degree-of-freedom dynamics are now available. The control problem is to choose \(U_1,U_2,U_3,U_4\) from the desired attitude and altitude so that the actual state approaches the target. The goal here is not to redesign the flight controller, but to clarify the <strong>controller inputs and outputs and the way the attitude subsystem affects translation</strong>.

### 5.1 State Space: Convert Six Second-Order Equations into Twelve First-Order Equations

Position and attitude dynamics are second-order equations. To describe them uniformly, retain both each quantity and its rate in a twelve-dimensional state vector:

\[
X=\begin{bmatrix}x_1&x_2&\cdots&x_{12}\end{bmatrix}^{T}.
\]

| Degree of freedom | State variables |
| --- | --- |
| \(x\) direction | \(x_1=x,\quad x_2=\dot x\) |
| \(y\) direction | \(x_3=y,\quad x_4=\dot y\) |
| \(z\) direction | \(x_5=z,\quad x_6=\dot z\) |
| Roll | \(x_7=\phi,\quad x_8=\dot\phi\) |
| Pitch | \(x_9=\theta,\quad x_{10}=\dot\theta\) |
| Yaw | \(x_{11}=\psi,\quad x_{12}=\dot\psi\) |

The complete model can then be written as

\[
\dot X=f(X,U),
\qquad
U=\begin{bmatrix}U_1&U_2&U_3&U_4\end{bmatrix}^{T}.
\]

This step introduces no new physics. It only reorganizes the six second-order equations in Section 4.3 into twelve first-order equations. The resulting form is convenient for numerical integration, state estimation, and control design.

<details>
<summary>Expand the complete twelve-state model</summary>

\[
\begin{aligned}
\dot x_1&=x_2,\\
\dot x_2&=(\cos x_7\sin x_9\cos x_{11}+\sin x_7\sin x_{11})\frac{U_1}{m},\\
\dot x_3&=x_4,\\
\dot x_4&=(\cos x_7\sin x_9\sin x_{11}-\sin x_7\cos x_{11})\frac{U_1}{m},\\
\dot x_5&=x_6,\\
\dot x_6&=-g+(\cos x_7\cos x_9)\frac{U_1}{m},\\
\dot x_7&=x_8,\\
\dot x_8&=x_{10}x_{12}\frac{I_y-I_z}{I_x}
-\frac{J_r}{I_x}x_{10}\Omega+\frac{l}{I_x}U_2,\\
\dot x_9&=x_{10},\\
\dot x_{10}&=x_8x_{12}\frac{I_z-I_x}{I_y}
+\frac{J_r}{I_y}x_8\Omega+\frac{l}{I_y}U_3,\\
\dot x_{11}&=x_{12},\\
\dot x_{12}&=x_8x_{10}\frac{I_x-I_y}{I_z}+\frac{1}{I_z}U_4.
\end{aligned}
\]

Odd-numbered states are positions or attitude angles, and each following even-numbered state is its rate. The derivatives of the even-numbered states are determined by forces or torques.

</details>

### 5.2 Subsystem Structure: Attitude Determines the Thrust Direction

<figure>
  <img src="../../../posts/quadrotor-dynamics/quadrotor-control-subsystems.png" alt="The quadrotor angular subsystem supplies roll, pitch, and yaw to the translation subsystem">
  <figcaption>Connection between the angular and translational subsystems: \(U_2,U_3,U_4\) change attitude, which then combines with total thrust \(U_1\) to determine translation. Source: Bouabdallah et al., <a href="https://doi.org/10.1109/ROBOT.2004.1302409"><em>Design and Control of an Indoor Micro Quadrotor</em></a>, Fig. 6.</figcaption>
</figure>

The twelve-state equations have a one-way dependency structure:

- The angular subsystem \(X_\alpha=[\phi,\dot\phi,\theta,\dot\theta,\psi,\dot\psi]^T\) is driven by \(U_2,U_3,U_4\) and does not depend on position or linear velocity.
- The translational subsystem \(X_\Delta=[x,\dot x,y,\dot y,z,\dot z]^T\) is driven by \(U_1\), but also depends on \(\phi,\theta,\psi\) because attitude determines the direction of total thrust.

This structure suggests stabilizing attitude first and then using the controlled attitude to regulate translation. The split is an ideal modeling and control structure, not a claim that attitude and position are physically decoupled: \(U_2,U_3,U_4\) still affect translation indirectly through attitude.

### 5.3 Attitude Control: Angle Error and Angular Rate Determine Torque Together

Let the desired attitude state be

\[
X_\alpha^d=
\begin{bmatrix}
x_7^d&0&x_9^d&0&x_{11}^d&0
\end{bmatrix}^{T}.
\]

The desired angular rates are zero, meaning that the vehicle should not only reach the target attitude but also stop rotating. Choose the following Lyapunov function to measure the combined attitude error and rotational motion:

\[
\begin{aligned}
V(X_\alpha)=\frac{1}{2}\big[&
(x_7-x_7^d)^2+x_8^2
+(x_9-x_9^d)^2+x_{10}^2\\
&+(x_{11}-x_{11}^d)^2+x_{12}^2
\big].
\end{aligned}
\]

The three angle-error terms measure how far the body is from the target; the three squared angular-rate terms measure how quickly it is rotating. Thus \(V>0\) whenever the state is away from the desired equilibrium.

For an ideal symmetric cross configuration, the paper assumes \(I_x=I_y\) and chooses

\[
\begin{aligned}
U_2&=-\frac{I_x}{l}(x_7-x_7^d)-k_1x_8,\\
U_3&=-\frac{I_y}{l}(x_9-x_9^d)-k_2x_{10},\\
U_4&=-I_z(x_{11}-x_{11}^d)-k_3x_{12},
\end{aligned}
\]

where \(k_1,k_2,k_3>0\). Each row has two effects:

- The term proportional to angle error pulls the body toward the desired attitude.
- The term opposing angular rate damps continued rotation and reduces oscillation around the target.

The laws therefore resemble three PD controllers rather than full PID controllers: they contain proportional and derivative terms but no integral term. The inputs \(U_2,U_3,U_4\) act simultaneously, and the three rotational axes remain dynamically coupled. Under the symmetry assumption, however, the coupling terms cancel in the Lyapunov derivative.

<details>
<summary>Why the attitude error decreases under this control law</summary>

Differentiating \(V\) along the system trajectory gives

\[
\begin{aligned}
\dot V={}&(x_7-x_7^d)x_8+x_8\frac{l}{I_x}U_2\\
&+(x_9-x_9^d)x_{10}+x_{10}\frac{l}{I_y}U_3\\
&+(x_{11}-x_{11}^d)x_{12}+x_{12}\frac{1}{I_z}U_4.
\end{aligned}
\]

Substitution of the control law cancels the angle-error terms and leaves

\[
\dot V
=-\frac{lk_1}{I_x}x_8^2
-\frac{lk_2}{I_y}x_{10}^2
-\frac{k_3}{I_z}x_{12}^2
\leq 0.
\]

This derivative is negative semidefinite because it can be zero whenever all three angular rates are zero, even if an angle error momentarily remains. LaSalle's invariance principle then shows that the only state that can remain indefinitely in the set \(\dot V=0\) is the target equilibrium, so the attitude converges asymptotically to the target.

</details>

### 5.4 Altitude Control: Compensate Gravity and Attitude Tilt

The altitude subsystem contains only \(x_5=z\) and \(x_6=\dot z\):

\[
\begin{aligned}
\begin{bmatrix}
\dot x_5\\
\dot x_6
\end{bmatrix}
&=
\begin{bmatrix}
x_6\\
-g+\cos x_7\cos x_9\dfrac{U_1}{m}
\end{bmatrix}.
\end{aligned}
\]

The factor \(\cos x_7\cos x_9\) shows that only the vertical component of total thrust can oppose gravity when the body rolls or pitches. Let

\[
U_1=
\frac{mg}{\cos x_7\cos x_9}
+\frac{m\widehat U_1}{\cos x_7\cos x_9}
=\frac{m(g+\widehat U_1)}{\cos\phi\cos\theta}.
\]

The first term compensates gravity and the loss of vertical thrust caused by tilt, while the second introduces a new virtual input \(\widehat U_1\). Substitution gives

\[
\begin{aligned}
\begin{bmatrix}
\dot x_5\\
\dot x_6
\end{bmatrix}
&=
\begin{bmatrix}
x_6\\
\widehat U_1
\end{bmatrix}.
\end{aligned}
\]

or simply

\[
\ddot z=\widehat U_1.
\]

The nonlinear altitude dynamics have become a double integrator. Using an intuitive positive-gain convention, choose

\[
\widehat U_1=-K_p(z-z_d)-K_d\dot z,
\qquad K_p,K_d>0,
\]

so that the altitude error obeys

\[
\ddot e_z+K_d\dot e_z+K_pe_z=0,
\qquad e_z=z-z_d.
\]

The same state-feedback idea can also be written as \(\widehat U_1=k_4x_5+k_5x_6\), with coefficients chosen so that the closed-loop poles lie in the left half of the complex plane. The apparent sign difference comes from the choice of error coordinates and gain convention.

This compensation requires

\[
\cos\phi\cos\theta\neq 0.
\]

As roll or pitch approaches \(90^\circ\), the required \(U_1\) tends toward infinity and the compensation fails. Stabilizing attitude before altitude is therefore not an arbitrary ordering; it is an important condition for the altitude controller to remain valid.

This controller stabilizes attitude and altitude but does not yet include an \(x,y\) position outer loop. Autonomous waypoint flight additionally requires a position or velocity controller that generates desired roll, pitch, and total thrust.

## 6. Summary: From Rotor Speed to Position, Attitude, and Closed-Loop Control

Quadrotor dynamics can be read as one causal chain:

\[
\text{rotor speed}
\longrightarrow
\text{thrust and torque}
\longrightarrow
\text{attitude change}
\longrightarrow
\text{thrust-direction change}
\longrightarrow
\text{position change}.
\]

The architecture limits the vehicle to four direct control inputs. The rotation matrix converts directions between the body and earth frames. The Newton–Euler equations describe how forces and torques change linear and angular velocity. Finally, the six-DOF equations expand those relationships into a form suitable for simulation and control design. The controller follows the causal chain in reverse: it computes \(U_1,U_2,U_3,U_4\) from desired attitude and altitude, and the rotors realize the corresponding forces and torques.
