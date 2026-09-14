---
title: "PID Control: From Intuition to Tuning"
date: 2026-09-14T12:00:00+08:00
series:
  main: "Robotics"
  subseries: "Dynamics and Control"
categories: ["Robotics"]
tags: ["PID Control", "Feedback Control", "Control Systems", "Robotics"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Understand PID through cruise control, then learn discrete implementation, manual tuning, and practical safeguards."
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
  image: "pid-parallel-structure.png"
  alt: "Parallel proportional, integral, and derivative structure of a PID controller"
  caption: "The error enters the P, I, and D branches, whose outputs are summed and applied to the plant"
  relative: true
editPost:
  URL: "https://cspaulia.github.io/cspaulia-blog/content/"
  Text: "Suggest Changes"
  appendFilePath: true
---

PID is a way to watch the result and correct the next action. This article uses <strong>cruise control</strong> to explain the three terms, then shows how to implement PID in code. The presentation follows the intuition-first approach of the [MathWorks PID introduction](https://www.mathworks.com/videos/series/understanding-pid-control.html) and the [University of Michigan PID tutorial](https://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=ControlPID).

## 1. Closed-loop feedback: observe the error and correct the next action

A car should travel at \(60\,\mathrm{km/h}\), but its measured speed is only \(50\,\mathrm{km/h}\). The error is

\[
e(t)=r(t)-y(t)=10\,\mathrm{km/h}.
\]

The controller uses this error to calculate the throttle command \(u(t)\). After the car accelerates, the sensor measures speed again and the controller repeats the calculation. This continuous cycle is <strong>closed-loop feedback</strong>.

<figure>
  <img src="../../../posts/pid-control/pid-parallel-structure.png" alt="The error enters proportional, integral, and derivative branches whose outputs are combined and applied to the plant">
  <figcaption>The error \(e(t)\) enters the P, I, and D branches in parallel. Their sum forms the control signal \(u(t)\), and the output \(y(t)\) is fed back to the input.</figcaption>
</figure>

If the car climbs a hill or faces a headwind, the old throttle position is no longer enough. Feedback notices the falling speed and compensates automatically.

## 2. Three actions: present, past, and trend

The PID control law is

\[
u(t)=K_pe(t)+K_i\int_0^t e(\tau)\,\mathrm d\tau+K_d\frac{\mathrm d e(t)}{\mathrm dt}.
\]

<figure>
  <img src="../../../posts/pid-control/pid-three-actions.svg" alt="Proportional, integral, and derivative actions in a cruise-control example">
  <figcaption>Three judgments in the same cruise-control task: distance from the target, duration of the error, and speed of approach.</figcaption>
</figure>

### 2.1 Proportional action: press harder when farther away

\[
u_P(t)=K_pe(t).
\]

Apply more throttle at \(40\,\mathrm{km/h}\) and less at \(58\,\mathrm{km/h}\). Increasing \(K_p\) accelerates the response, but too much can drive the speed past the target and cause oscillation.

### 2.2 Integral action: keep adding effort while an error persists

\[
u_I(t)=K_i\int_0^t e(\tau)\,\mathrm d\tau.
\]

On a hill, the car may remain at \(58\,\mathrm{km/h}\). Each error is small, but the integral accumulates it and gradually adds enough throttle to remove the <strong>steady-state error</strong>.

If the throttle is already saturated while the integral keeps growing, <strong>integral windup</strong> occurs. The stored integral can keep accelerating the car after the hill ends, so the implementation needs output limits and anti-windup.

### 2.3 Derivative action: back off early when approaching too quickly

\[
u_D(t)=K_d\frac{\mathrm d e(t)}{\mathrm dt}.
\]

The speed may still be below \(60\,\mathrm{km/h}\) while rising rapidly. Derivative action reduces throttle early and behaves like damping. A large \(K_d\) also amplifies sensor fluctuations, so derivative action normally needs filtering.

> In one sentence: \(P\) sees the present, \(I\) remembers the past, and \(D\) watches the trend.

## 3. Manual tuning: P first, D second, I last

Watch four properties while tuning: response speed, overshoot, settling time, and final offset.

<figure>
  <img src="../../../posts/pid-control/pid-response-metrics.svg" alt="Rise time, overshoot, settling time, and steady-state error on a step response">
  <figcaption>A typical response after a target change. Tuning balances speed, overshoot, and final error.</figcaption>
</figure>

A useful sequence is:

1. Set \(K_i=K_d=0\), then verify that a positive control action really reduces the error.
2. Increase \(K_p\) from a small value until the response is fast enough without sustained oscillation.
3. Add \(K_d\) to reduce overshoot and back-and-forth motion.
4. Add \(K_i\) slowly and only to remove the remaining persistent offset.
5. Change the target, add a disturbance, and test again.

| Observed behavior | First adjustment |
| --- | --- |
| Very slow response | Increase \(K_p\) |
| Overshoot or oscillation | Reduce \(K_p\) or \(K_i\); add a modest \(K_d\) |
| Persistent final offset | Increase \(K_i\) slightly |
| High-frequency output chatter | Reduce \(K_d\); improve derivative filtering |
| Slow recovery after saturation | Add anti-windup |

Every system does not need full PID. PD may be sufficient when no persistent offset exists; PI is often better when a reliable derivative is difficult to obtain.

## 4. Discrete implementation: what each control-loop iteration does

For sampling period \(T_s\), define \(P_k\), \(I_k\), and \(D_k\) directly as their contributions to the control signal so that the equations match the implementation below:

\[
e_k=r_k-y_k,
\]

\[
P_k=K_pe_k,
\]

\[
I_k=I_{k-1}+K_ie_kT_s,
\]

\[
D_k=-K_d\frac{y_k-y_{k-1}}{T_s},
\]

\[
u_k=P_k+I_k+D_k.
\]

The derivative is taken from the measurement \(y\). When the target is constant, \(\dot e=-\dot y\), which explains the minus sign. This form also avoids derivative kick when the target changes suddenly.

Each iteration has only five steps: <strong>read the measurement → compute error → update P/I/D → constrain the output → command the actuator</strong>.

<details>
<summary>Show a Python implementation with derivative filtering and anti-windup</summary>

```python
class PID:
    def __init__(self, kp, ki, kd, dt, limits,
                 derivative_filter_time=0.02,
                 anti_windup_gain=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.u_min, self.u_max = limits
        self.tau_d = derivative_filter_time
        self.kaw = anti_windup_gain
        self.integral = 0.0
        self.d_state = 0.0
        self.previous_measurement = None

    def update(self, setpoint, measurement, feedforward=0.0):
        error = setpoint - measurement
        p = self.kp * error

        # Differentiate measurement to avoid setpoint-induced kick.
        if self.previous_measurement is None:
            d_raw = 0.0
        else:
            d_raw = -(measurement - self.previous_measurement) / self.dt

        alpha = self.tau_d / (self.tau_d + self.dt)
        self.d_state = alpha * self.d_state + (1 - alpha) * d_raw
        d = self.kd * self.d_state

        u_raw = feedforward + p + self.integral + d
        u = min(max(u_raw, self.u_min), self.u_max)

        # Pull the integrator back toward the usable output range.
        self.integral += (
            self.ki * error + self.kaw * (u - u_raw)
        ) * self.dt

        self.previous_measurement = measurement
        return u
```

The implementation adds three features absent from the ideal equation: actuator limits, a low-pass-filtered derivative, and anti-windup when the output saturates.

</details>

## 5. Preflight check: four common failure points

1. <strong>Sign:</strong>does a positive command actually reduce the error? A wrong sign turns negative feedback into positive feedback.
2. <strong>Sampling:</strong>does \(T_s\) match the real loop period?
3. <strong>Physical limits:</strong>are the output and integral contribution bounded?
4. <strong>Logging:</strong>are target, measurement, error, and the individual P/I/D terms recorded? Without curves, it is difficult to know which gain to change.

The central idea is simple: use \(P\) to pull the system toward the target, \(D\) to keep it from rushing past, and a small amount of \(I\) to remove the final persistent offset.
