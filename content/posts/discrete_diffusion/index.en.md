---
title: "Discrete Diffusion Models and Discrete Flow Matching"
date: 2026-09-24T09:00:00+08:00
series:
    main: "Generative Models"
    subseries: "Fundamentals"
categories: ["Generative Models"]
tags: ["Discrete Diffusion", "Flow Matching", "Diffusion Models", "Language Models"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Notes for Lecture 5 of MIT's “Introduction to Flow Matching and Diffusion Models 2026”: continuous-time Markov chains (CTMCs), discrete diffusion models, discrete flow matching, and masked diffusion language models."
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
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

The previous lectures all dealt with processes on continuous spaces: data can be written as continuous vectors, and both noising and denoising can be described by SDEs or ODEs. Language and protein sequences are different. They are discrete data, where a token can only take one of the values in the vocabulary and there is no intermediate state "partway between" two tokens.

Discrete diffusion models carry the learning principles of denoising and flow matching over to discrete states, and the direction has been showing up in the news: Google's [Gemini Diffusion](https://blog.google/technology/google-deepmind/gemini-diffusion/), announced at I/O, generates text with a diffusion process at roughly 1500 tokens/s in the official demo; Inception Labs claims over 1000 tokens/s on NVIDIA H100s; ByteDance released Seed Diffusion, a large language model built on discrete diffusion.

<figure>
  <img src="../../../posts/discrete_diffusion/discrete_diffusion_in_the_news.png" alt="Collage of news coverage about text-producing diffusion models: Forbes and Fortune headlines, an Inception Labs output-speed chart, and a ByteDance Seed Diffusion demo frame" width="100%" />
  <figcaption>Figure 1: Discrete diffusion models in the news. Figure source: MIT 6.S184 Lecture 5; collage materials from public announcements by Forbes, Fortune, Inception Labs, and ByteDance Seed.</figcaption>
</figure>

Unlike autoregressive models, diffusion language models do not generate strictly from left to right. They generate text in arbitrary order: at every step the model can see all positions of the sequence and decides which positions to update. This line of work goes back to Austin et al., 2021, [<em>Structured Denoising Diffusion Models in Discrete State-Spaces</em>](https://arxiv.org/abs/2107.03006), which introduced Discrete Denoising Diffusion Probabilistic Models (D3PM) and generalized diffusion processes to discrete state spaces.

## 1. Continuous-Time Markov Chains (CTMC)

A discrete state space has neither SDEs nor ODEs: on a continuous space, evolution can be described as "moving a little along a vector field," whereas a discrete state can only change from one value to another as a whole. The replacement is the continuous-time Markov chain (CTMC), which treats time as continuous and state changes as instantaneous jumps.

### 1.1. State Space and Rate Matrix

The notation used in this section is collected below:

- <strong>State space</strong> \(S\): the set of all states the system can be in.
- \(d\): the sequence length. A state is written \(x = (x_1, \dots, x_d) \in S\), that is, a sequence of length \(d\).
- <strong>Vocabulary</strong> \(V\): the set of values each position \(x_i\) can take, so the state space has size \(|V|^d\).
- \((X_t)_{t \ge 0}\): the \(S\)-valued stochastic process, where \(X_t\) denotes the state of the system at time \(t\).
- \(p_{t+h|t}\): the transition probability from time \(t\) to time \(t+h\), where \(h \ge 0\) is the time interval.
- \(Q_t(y \mid x)\): the rate of jumping from state \(x\) to state \(y\).

All the information about a CTMC is packed into the <strong>rate matrix</strong> formed by these rates. It varies with time \(t \in [0,1]\):

\[
Q: S \times S \times [0,1] \to \mathbb{R}_{\ge 0}, \qquad (x, y, t) \mapsto Q_t(y \mid x)
\]

These rates must satisfy two constraints:

- Constraint 1: \(Q_t(y \mid x) \ge 0\); jump rates are non-negative.
- Constraint 2: \(Q_t(x \mid x) = -\sum_{y \ne x} Q_t(y \mid x)\); the diagonal entries are negative, with magnitude equal to the total rate of leaving \(x\), so every row of the matrix sums to zero.

A rate matrix also has an equivalent infinitesimal characterization: it is the derivative of the transition probability at \(h = 0\).

\[
\left.\frac{d}{dh} p_{t+h|t}(X_{t+h} = y \mid X_t = x)\right|_{h=0} = Q_t(y \mid x)
\]

The equation says that for small \(h\), the probability of jumping from \(x\) to a different state \(y \ne x\) is roughly \(h \cdot Q_t(y \mid x)\), so \(Q_t(y \mid x)\) is the slope at which that probability grows with \(h\). For \(y = x\) the derivative on the left is negative, matching the diagonal entries in Constraint 2.

The \(Q_t\) defined this way is also called the <strong>infinitesimal generator</strong> of the process: it does not give the transition probability over a finite time, only where the next jump goes and how fast. Discretizing the evolution equation with a single Euler step yields exactly the same approximation \(p_{t+h|t}(y \mid x) \approx h \cdot Q_t(y \mid x)\) for \(y \ne x\).

### 1.2. Sample Paths: Jumps at Random Times

Fixing one realization gives a staircase-shaped <strong>sample path</strong> \(t \mapsto X_t\): the chain stays in one state for a random holding time, then jumps instantaneously to another state at a random time.

<figure>
  <img src="../../../posts/discrete_diffusion/ctmc_sample_path.png" alt="A CTMC sample path: the state jumps between S1, S2, and S3 at the four random times t1 through t4" width="100%" />
  <figcaption>Figure 2: A CTMC sample path. The chain passes through \(S_3 \to S_1 \to S_2 \to S_1 \to S_3\) at \(t_1, t_2, t_3, t_4\), and the state stays fixed between jumps. Figure source: MIT 6.S184 Lecture 5; figure: Andrew Campbell.</figcaption>
</figure>

### 1.3. Two-State Example: Transition Probabilities Converge to Uniform

Take the simplest state space \(S = \{a, b\}\) with rate matrix

\[
Q = \begin{pmatrix} -\lambda & \lambda \\ \lambda & -\lambda \end{pmatrix}
\]

Rows index the source state and columns the target state, so both directions jump at rate \(\lambda\). Solving the evolution equation (by taking derivatives with respect to \(h\)) gives the transition probabilities over an interval \(h\):

\[
\begin{pmatrix}
p(X_{t+h} = a \mid X_t = a) & p(X_{t+h} = a \mid X_t = b) \\
p(X_{t+h} = b \mid X_t = a) & p(X_{t+h} = b \mid X_t = b)
\end{pmatrix}
= \frac{1}{2}\begin{pmatrix}
1 + e^{-2\lambda h} & 1 - e^{-2\lambda h} \\
1 - e^{-2\lambda h} & 1 + e^{-2\lambda h}
\end{pmatrix}
\]

<details>
<summary>Derivation: solving for the transition probabilities over an interval \(h\)</summary>

Write \(a(h) = p(X_{t+h} = a \mid X_t = a)\) and \(b(h) = p(X_{t+h} = b \mid X_t = a)\), which always sum to 1. Start with the probability of being at \(a\) after a small extra step \(dh\), having started at \(a\). The probability of two or more jumps within that step is \(o(dh)\) and can be ignored, so only two paths remain: no jump at all, with probability \(1 - \lambda\,dh\); or a jump to \(b\) followed by a jump back to \(a\), with probability \(b(h)\,\lambda\,dh\). Adding the two cases gives

\[
a(h + dh) = a(h)\,(1 - \lambda\,dh) + b(h)\,\lambda\,dh + o(dh)
\]

Subtracting \(a(h)\), dividing by \(dh\), and letting \(dh \to 0\) gives the evolution equation written by the derivative relation in Section 1.1:

\[
a'(h) = \lambda\,(b(h) - a(h)) = \lambda\,(1 - 2a(h))
\]

The initial value is \(a(0) = 1\). This is the first-order linear equation \(a'(h) + 2\lambda\,a(h) = \lambda\); multiplying both sides by the integrating factor \(e^{2\lambda h}\) turns the left-hand side into a single derivative,

\[
\frac{d}{dh}\left(e^{2\lambda h} a(h)\right) = \lambda\,e^{2\lambda h}
\]

Integrating over \(h\) gives \(e^{2\lambda h} a(h) = \frac{1}{2}e^{2\lambda h} + C\), and \(a(0) = 1\) fixes \(C = \frac{1}{2}\), so

\[
a(h) = \frac{1}{2} + \frac{1}{2}e^{-2\lambda h} = \frac{1}{2}\left(1 + e^{-2\lambda h}\right)
\]

Since \(a(h) + b(h) = 1\), we immediately get \(b(h) = \frac{1}{2}\left(1 - e^{-2\lambda h}\right)\). Swapping the starting state to \(b\) exchanges the roles of \(a\) and \(b\), which gives the transition probability matrix in the main text.

</details>

As \(h \to \infty\) the exponential terms decay to 0 and the matrix converges to the all-\(1/2\) matrix: starting from either \(a\) or \(b\), after a long enough time the probability of landing in either state is \(1/2\).

<figure>
  <img src="../../../posts/discrete_diffusion/ctmc_two_state_convergence.png" alt="Transition probabilities of a two-state CTMC against the interval h, all converging to 1/2" width="100%" />
  <figcaption>Figure 3: Transition probabilities of a two-state CTMC as a function of the interval \(h\). The diagonal entries decay from 1 to \(1/2\) and the off-diagonal entries grow from 0 to \(1/2\). Figure source: MIT 6.S184 Lecture 5.</figcaption>
</figure>

In this example the information about the initial state is gradually "forgotten," and the distribution converges to a uniform distribution independent of where it started. Generative modeling with CTMCs exploits exactly this: the initial distribution serves as a noise distribution that is independent of the data, and the model learns to turn it into data step by step.

## 2. CTMC Models

The previous section treated the rate matrix as a given object. Generative modeling also needs it to be learnable: a neural network outputs all the jump rates.

### 2.1. Model: Parameterizing the Rate Matrix with a Network

Hand the rate matrix to a parameterized neural network and write its output as \(Q_t^\theta(y \mid x)\), where \(\theta\) are the network parameters.

The difficulty is scale. The state space has size \(|S| = |V|^d\), so a complete rate matrix has \(|V|^{2d}\) entries, which is impractical to store or to learn. The model needs a constraint that sets the vast majority of rates directly to zero, and this is the <strong>factorization condition</strong>:

Whenever \(x\) and \(y\) differ in more than one position, set \(Q_t^\theta(y \mid x) = 0\).

In other words, <strong>jumps are allowed only between neighbors</strong>. The network then does not have to describe the relationship between arbitrary pairs of states; it only has to describe jumps of the form "replace one position with a different token."

The network takes the current state and time \((x, t)\) as input, and one forward pass outputs a \(d \times |V|\) array, where row \(i\) corresponds to position \(i\) and column \(j\) to the token \(v_j\) in the vocabulary; written in the abbreviated index notation, it is

\[
\begin{pmatrix}
Q_t^\theta(v_1, 1 \mid x) & Q_t^\theta(v_2, 1 \mid x) & \cdots & Q_t^\theta(v_{|V|}, 1 \mid x) \\
Q_t^\theta(v_1, 2 \mid x) & Q_t^\theta(v_2, 2 \mid x) & \cdots & Q_t^\theta(v_{|V|}, 2 \mid x) \\
\vdots & \vdots & \ddots & \vdots \\
Q_t^\theta(v_1, d \mid x) & Q_t^\theta(v_2, d \mid x) & \cdots & Q_t^\theta(v_{|V|}, d \mid x)
\end{pmatrix}
= \left(Q_t^\theta(v, i \mid x)\right)_{\substack{v \in V \\ i \in [d]}} \in \mathbb{R}^{d \times |V|}
\]

Here \(Q_t^\theta(v, i \mid x)\) is the rate of replacing the token at position \(i\) with \(v\): the index \(v\) ranges over the vocabulary and \(i\) over the positions, and filling in both indices recovers the matrix on the left. This is the notation behind \(Q_t^\theta(\cdot \mid X_t)\) in the algorithms below.

The factorization condition has already ruled out jumps that change several positions at once, so the network only needs to output one \(|V|\)-dimensional vector per position, \(d \times |V|\) numbers in total; when \(v\) is the token already sitting at position \(i\), that entry corresponds to "leave this position unchanged." The diagonal rates of jumping back to the same state are determined by Constraint 2 and do not need to be output separately.

### 2.2. Factorization Condition: Jumps Only Between Neighbors

**Definition 1 (Neighbors)**: two states \(x\) and \(y\) are neighbors if they differ in the value at only one position, that is, if there is some \(i \in [d]\) with \(x_i \ne y_i\) while \(x_j = y_j\) for every \(j \ne i\).

With the notion of neighbors, the factorization condition becomes a single sentence: \(Q_t^\theta(y \mid x)\) can be non-zero only when \(x\) and \(y\) are neighbors.

<figure>
  <img src="../../../posts/discrete_diffusion/ctmc_neighbors_example.png" alt="An example of neighbors: states x and y differ only in the value at position 4, y and z only at position 3, while z and x differ at both positions 3 and 4" width="100%" />
  <figcaption>Figure 4: An example of neighbors. \(x\) and \(y\) differ only at position 4 and \(y\) and \(z\) only at position 3, so both pairs are neighbors; \(z\) and \(x\) differ at both positions 3 and 4, so they are not neighbors and \(Q_t^\theta(z \mid x) = 0\). Figure source: MIT 6.S184 Lecture 5.</figcaption>
</figure>

For a sequence of length \(d\) with vocabulary size \(|V|\), each state \(x\) has \(d(|V| - 1)\) neighbors: each of the \(d\) positions can be replaced by any of the remaining \(|V| - 1\) tokens. The factorization condition shrinks the set of possible jump targets from \(|V|^d\) down to that order.

### 2.3. Factorized CTMC vs. General CTMC

- A <strong>general CTMC</strong> allows a single jump to change several positions at once, so a rate \(Q_t(y \mid x)\) must be specified for every pair of states \((x, y)\) in the state space.
- A <strong>factorized CTMC</strong> restricts each jump to a single position, and the rate is written \(Q_t(v_i, i \mid x)\), that is, replace the token at position \(i\) with \(v_i\).

<figure>
  <img src="../../../posts/discrete_diffusion/factorized_vs_general_ctmc.png" alt="Left: jumps from x in a general CTMC can point to any state; right: jumps in a factorized CTMC can only follow the horizontal and vertical lines through x" width="100%" />
  <figcaption>Figure 5: A general CTMC compared with a factorized CTMC. In the left figure the arrows leaving \(x\) can point to any state in the grid, with rates written \(Q_t(y \mid x)\); in the right figure the arrows can only follow the horizontal and vertical lines through \(x\), that is, they change a single position, with rates written \(Q_t(v_i, i \mid x)\). Figure source: MIT 6.S184 Lecture 5; figure: Yaron Lipman.</figcaption>
</figure>

The difference finally shows up in the number of parameters: a general CTMC needs \(|V|^{2d}\) rates, while a factorized CTMC needs only \(d|V|\). This is also the precondition that makes discrete diffusion models trainable.

## 3. Sampling: Generating Samples from a CTMC Model

Once the model is trained, the remaining question is how to draw samples from it. That takes only two ingredients: the distribution to start from, and how to update the state at every step.

### 3.1. Sampling Formula: Approximating the Transition Probability with One Expansion Step

The starting point is the <strong>limit distribution</strong> \(p_{init} = \mathrm{Unif}_S\), which puts equal probability on all \(|V|^d\) states of the state space.

> **Delta function**: \(\delta_y(X_t) = \begin{cases} 1, & y = X_t \\ 0, & y \ne X_t \end{cases}\). It describes a point mass — all the probability sits on the single state \(X_t\), and every other state gets zero.

Sampling starts from \(X_0 \sim p_{init}\) and advances with a step size \(h > 0\). The transition probability \(p_{t+h|t}(\cdot \mid X_t)\) needed at each step cannot be read off the network directly, so it takes one Taylor expansion step in \(h\):

\[
\begin{aligned}
X_{t+h} \sim p_{t+h|t}(y \mid X_t) &= p_{t|t}(y \mid X_t) + h\left.\frac{d}{dh}p_{t+h|t}(y \mid X_t)\right|_{h=0} \\
&= \delta_y(X_t) + h\,Q_t(y \mid X_t)
\end{aligned}
\]

The first term \(p_{t|t}(y \mid X_t)\) is the transition probability over a zero-length interval: no time passes and the state cannot change, so it equals the delta function \(\delta_y(X_t)\).

This distribution is non-zero on only finitely many states: the probability of staying at \(X_t\) is \(1 + h\,Q_t(X_t \mid X_t)\), and the probability of jumping to some \(y \ne X_t\) is \(h\,Q_t(y \mid X_t)\).

### 3.2. Sampling Algorithm: One Euler Update per Position, in Parallel

| **Algorithm 7** Sampling from a Factorized CTMC Model (Euler / τ-leaping) |
|-------------------------------|
| **Input**: factorized rate network \(Q_t^\theta\), initial distribution \(p_{init}\), number of steps \(n\) |
| 1: Set \(t \leftarrow 0\) |
| 2: Set step size \(h \leftarrow \frac{1}{n}\) |
| 3: Draw a sample \(X_0 \sim p_{init}\), where \(X_0 = (X_0^{(1)}, \dots, X_0^{(d)}) \in V^d\) |
| 4: **for** \(i = 1, \dots, n\) **do** |
| 5: \(~~~~\)Read the factorized jump rates off the network: \(\{q_j(v)\}_{j = 1..d,\ v \in V} \leftarrow Q_t^\theta(\cdot \mid X_t)\) |
| 6: \(~~~~\)**for** \(j = 1, \dots, d\) (in parallel) **do** |
| 7: \(~~~~~~~~\)\(x \leftarrow X_t^{(j)}\), the current token at position \(j\) |
| 8: \(~~~~~~~~\)Define the per-position Euler transition probabilities \(\tilde p_{j,t}(\cdot \mid X_t^{(j)} = x)\) by |
| 9: \(~~~~~~~~\)\(\tilde p_{j,t}(v \mid x) = \begin{cases} h\,q_j(v), & v \ne x \\ 1 - h\sum_{v' \in V \setminus \{x\}} q_j(v'), & v = x \end{cases}\) |
| 10: \(~~~~~~~~\)Sample \(X_{t+h}^{(j)} \sim \mathrm{Categorical}\left(\{\tilde p_{j,t}(v \mid x)\}_{v \in V}\right)\) |
| 11: \(~~~~\)**end for** |
| 12: \(~~~~\)Set \(t \leftarrow t + h\) |
| 13: **end for** |
| **Output**: \(X_1\) |

<figure>
  <img src="../../../posts/discrete_diffusion/mdlm_sampling.gif" alt="Animation of MDLM sampling: the sequence starts almost entirely masked (shown as blank), some positions are filled with real tokens at each step, ending in a complete passage" width="100%" />
  <figcaption>Figure 6: Sampling from a masked diffusion language model (MDLM). The sequence starts from masks, and each step replaces some positions with real tokens in random order (masked positions appear blank in the animation). Figure source: the <a href="https://s-sahoo.com/mdlm/">MDLM project page</a>; paper: Sahoo et al., <a href="https://arxiv.org/abs/2406.07524"><em>Simple and Effective Masked Diffusion Language Models</em></a> (NeurIPS 2024).</figcaption>
</figure>

## 4. Generative Modeling: Turning Noise into Data with a CTMC

- <strong>Data distribution</strong> \(p_{data}(z)\), \(z \in S\): the distribution we ultimately want the model to generate, say the distribution of text on the internet.
- <strong>Initial distribution</strong> \(p_{init}(z)\), \(z \in S\): where sampling starts, for example the uniform distribution \(p_{init}(z) = \frac{1}{|S|}\).
- <strong>Goal</strong>: use a CTMC to turn "noise" into data,

\[
X_0 \sim p_{init} \xrightarrow{\ \text{CTMC}\ } X_1 \sim p_{data}
\]

## 5. Discrete Flow Matching

### 5.1. Roadmap: From Conditional Probability Path to Training Loss

<figure>
  <img src="../../../posts/discrete_diffusion/continuous_flow_matching.png" alt="Two rows of continuous flow matching: conditional probability path → conditional vector field → conditional flow matching loss; marginal probability path → marginal vector field → marginal flow matching loss" width="100%" />
  <figcaption>Figure 7: The three steps of continuous flow matching. The conditional row on top and the marginal row below correspond one for one. Figure source: MIT 6.S184 Lecture 5.</figcaption>
</figure>

<figure>
  <img src="../../../posts/discrete_diffusion/discrete_flow_matching_matrix.png" alt="Two rows of discrete flow matching: conditional probability path → conditional rate matrix → discrete flow matching loss; marginal probability path → marginal rate matrix → discrete flow matching loss" width="100%" />
  <figcaption>Figure 8: Discrete flow matching swaps the vector field of the middle link for a rate matrix, with everything else corresponding one for one. Figure source: MIT 6.S184 Lecture 5.</figcaption>
</figure>

### 5.2. Conditional and Marginal Probability Paths

A <strong>conditional probability path</strong> \(p_t(x \mid z)\) (\(0 \le t \le 1\), \(x, z \in S\)) is the path that ends at the data point \(z\). For each fixed \(z\) it is a valid probability distribution over \(S\):

\[
p_t(x \mid z) \ge 0, \qquad \sum_{x \in S} p_t(x \mid z) = 1
\]

Both endpoints are known: at \(t = 0\) the data has not come in yet, and at \(t = 1\) the data is settled:

\[
p_0(x \mid z) = p_{init}(x), \qquad p_1(x \mid z) = \delta_z(x)
\]

A <strong>marginal probability path</strong> \(p_t(x)\) averages the data points under the data distribution \(p_{data}\), which gives the unconditional distribution of \(X_t\):

\[
p_t(x) = \sum_{z \in S} p_t(x \mid z)\, p_{data}(z) = \mathbb{E}_{z \sim p_{data}}\left[p_t(x \mid z)\right]
\]

Its endpoints line up as well:

\[
p_0 = p_{init}, \qquad p_1 = p_{data}
\]

One common concrete example is the <strong>factorized mixture path</strong>. A <strong>scheduler</strong> \(\kappa_t\) sets the mixing ratio: \(0 \le \kappa_t \le 1\), \(\kappa_0 = 0\), \(\kappa_1 = 1\), for example \(\kappa_t = t^{0.7}\) (Figure 9).

<figure>
  <img src="../../../posts/discrete_diffusion/factorized_mixture_scheduler.png" alt="Scheduler curves: the blue κ_t = t^0.7 rises from 0 to 1 while the orange 1 − κ_t falls from 1 to 0, crossing at t slightly below 0.4" width="100%" />
  <figcaption>Figure 9: The scheduler \(\kappa_t = t^{0.7}\) and \(1 - \kappa_t\) as functions of \(t\). Figure source: MIT 6.S184 Lecture 5.</figcaption>
</figure>

The path mixes noise and data independently at every position:

\[
p_t(x \mid z) = \prod_{j=1}^{d} \left[(1 - \kappa_t)\, p_{init}^{(j)}(x_j) + \kappa_t\, \delta_{z_j}(x_j)\right] \tag{5.2}
\]

Here \(p_{init}^{(j)}\) is the marginal distribution of the initial distribution at position \(j\). Both endpoints check out: at \(t = 0\), \(\kappa_0 = 0\) and the product degenerates to \(p_{init}(x)\); at \(t = 1\), \(\kappa_1 = 1\) and it degenerates to \(\delta_z(x)\).

Sampling is then an independent two-way choice per position: when \(m_j \sim \mathrm{Bernoulli}(\kappa_t)\) comes up 1 take the data token \(z_j\), otherwise draw a noise token \(\xi_j\) from \(p_{init}^{(j)}\), giving \(x_j = m_j z_j + (1 - m_j)\, \xi_j\).

### 5.3. Rate Matrices: Conditional and Marginal

A <strong>conditional rate matrix</strong> \(Q_t^z(y \mid x)\) (\(x, y \in S\), \(0 \le t \le 1\)) is the rate matrix belonging to the conditional probability path \(p_t(\cdot \mid z)\) that ends at the data point \(z\):

\[
X_0 \sim p_{init}, \quad X_t \sim \mathrm{CTMC}(Q_t^z) \implies X_t \sim p_t(\cdot \mid z)
\]

**Theorem 1 (Discrete Marginalization Trick)**: the <strong>marginal rate matrix</strong> \(Q_t(y \mid x)\) defined by

\[
Q_t(y \mid x) = \sum_{z \in S} Q_t^z(y \mid x)\, \frac{p_t(x \mid z)\, p_{data}(z)}{p_t(x)} \tag{5.3}
\]

fulfills

\[
X_0 \sim p_{init}, \quad X_t \sim \mathrm{CTMC}(Q_t) \implies X_t \sim p_t \implies X_1 \sim p_{data}
\]

> The fraction in the weights is the posterior probability \(p_t(z \mid x)\): given the current state \(x\), how likely it is to have come from the data point \(z\).

### 5.4. Kolmogorov Forward Equation (KFE): The Rate Matrix Determines How Probability Changes

The rate matrix \(Q_t\) only says where the chain jumps and how fast; the probability distribution \(p_t\) is the quantity we actually care about. A single equation ties the two together: a CTMC with rate matrix \(Q_t\) follows the probability path \(X_t \sim p_t\) (\(0 \le t \le 1\)) if and only if

\[
\frac{d}{dt} p_t(x) = \sum_{y \in S} Q_t(x \mid y)\, p_t(y) \tag{5.4}
\]

This is the <strong>Kolmogorov Forward Equation</strong> (KFE), the discrete analogue of the continuity equation — except that probability is not carried across space, it is teleported in one jump.

#### Proof of Theorem 1

With the KFE (5.4) in hand, Theorem 1 from §5.3 takes only a few lines. Start from the definition of the marginal path \(p_t(x) = \sum_{z \in S} p_t(x \mid z)\, p_{data}(z)\) and differentiate both sides with respect to \(t\):

\[
\begin{aligned}
\frac{d}{dt} p_t(x) &= \sum_{z \in S} \frac{d}{dt} p_t(x \mid z)\, p_{data}(z) \\
&= \sum_{z \in S} \left[\sum_{y \in S} Q_t^z(x \mid y)\, p_t(y \mid z)\right] p_{data}(z) \\
&= \sum_{y \in S} \left[\sum_{z \in S} Q_t^z(x \mid y)\, \frac{p_t(y \mid z)\, p_{data}(z)}{p_t(y)}\right] p_t(y) \\
&= \sum_{y \in S} Q_t(x \mid y)\, p_t(y)
\end{aligned}
\]

### 5.5. Conditional Rate Matrix for the Factorized Mixture Path: One Token at a Time

Section 5.3 defined the conditional rate matrix for an arbitrary path; for the factorized mixture path of §5.2 — equation (5.2) — it can be written in closed form, and it only updates a single token (factorized):

\[
Q_t^z(y \mid x) = \left(Q_t^z(v_i, j \mid x_j)\right)_{v_i, j}
\]

\[
Q_t^z(v_i, j \mid x_j) = \frac{\dot\kappa_t}{1 - \kappa_t}\left(\delta_{z_j}(v_i) - \delta_{x_j}(v_i)\right) \tag{5.5}
\]

Here \(v_i\) is the \(i\)-th token of the vocabulary and \(j\) a position; the subscript \(v_i, j\) selects the entry "position \(j\) takes the token \(v_i\)". Since only one token changes at a time, the rate depends only on the current token \(x_j\) at that position. Expanding the two delta functions gives four cases:

\[
Q_t^z(v_i, j \mid x_j) = \frac{\dot\kappa_t}{1 - \kappa_t}
\begin{cases}
0, & x_j = z_j \\
1, & v_i = z_j,\ x_j \ne z_j \\
0, & v_i \ne z_j,\ v_i \ne x_j,\ x_j \ne z_j \\
-1, & v_i = x_j,\ x_j \ne z_j
\end{cases}
\]

- The current token is already correct (\(x_j = z_j\)): rate 0, nothing to fix.
- The current token is wrong and the target is the correct token (\(v_i = z_j\)): jump there at rate \(\frac{\dot\kappa_t}{1 - \kappa_t}\).
- The current token is wrong and the target is a different wrong token (\(v_i \ne z_j,\ v_i \ne x_j\)): rate 0 — the chain never jumps from one wrong token to a different wrong token.
- The target is the current token itself (\(v_i = x_j\)): the negative diagonal entry, exactly the size of the outgoing rate.

<details>
<summary>Derivation: where the closed form (5.5) comes from</summary>

Fix one position \(j\). Equation (5.2) gives

\[
p_t(u \mid z_j) = (1 - \kappa_t)\, p_{init}^{(j)}(u) + \kappa_t\, \delta_{z_j}(u)
\]

For \(u\ne z_j\), \(\delta_{z_j}(u)=0\), so \(p_t(u\mid z_j)=(1-\kappa_t)p_{init}^{(j)}(u)\).

Let wrong tokens jump only to \(z_j\). By the diagonal constraint in §1.1 and the KFE (5.4), \(\dot p_t(u\mid z_j)=Q_t^z(u,j\mid u)p_t(u\mid z_j)\) for \(u\ne z_j\). Substituting the path above, for \(p_{init}^{(j)}(u)>0\) and \(t<1\), gives

\[
\begin{aligned}
Q_t^z(z_j,j\mid u)&=-Q_t^z(u,j\mid u) \\
&= -\frac{1}{p_t(u\mid z_j)}\frac{d}{dt}p_t(u\mid z_j) \\
&= -\frac{d}{dt}\log(1-\kappa_t) \\
&= \frac{\dot\kappa_t}{1-\kappa_t}.
\end{aligned}
\]

For a current token \(x_j\ne z_j\), the off-diagonal entry for jumping to \(z_j\) takes this rate, the diagonal entry takes its negative (§1.1), and all other entries are zero. The two delta functions select these entries:

\[
\begin{aligned}
Q_t^z(v_i,j\mid x_j)
&= \frac{\dot\kappa_t}{1-\kappa_t}\delta_{z_j}(v_i)
-\frac{\dot\kappa_t}{1-\kappa_t}\delta_{x_j}(v_i) \\
&= \frac{\dot\kappa_t}{1-\kappa_t}\left(\delta_{z_j}(v_i)-\delta_{x_j}(v_i)\right).
\end{aligned}
\]

When \(x_j=z_j\), the terms cancel and the rate is zero, so the same formula still applies. Substituting into the single-position KFE checks it:

\[
\begin{aligned}
\sum_{v\in V}Q_t^z(u\mid v)\,p_t(v\mid z_j)
&= \frac{\dot\kappa_t}{1-\kappa_t}\left[\delta_{z_j}(u)-p_t(u\mid z_j)\right] \\
&= \dot\kappa_t\left[\delta_{z_j}(u)-p_{init}^{(j)}(u)\right] \\
&= \frac{d}{dt}p_t(u\mid z_j).
\end{aligned}
\]

The right-hand side is the derivative of (5.2), so (5.5) satisfies the KFE. Restricting wrong tokens to jump only to \(z_j\) is the construction used here.

</details>

The rate explodes as \(t \to 1\): \(\kappa_1 = 1\) drives the denominator \(1 - \kappa_t\) to zero while the numerator \(\dot\kappa_t\) does not vanish at \(t = 1\) (with \(\kappa_t = t\), as in Figure 10, the numerator is exactly 1).

<figure>
  <img src="../../../posts/discrete_diffusion/conditional_rate_explosion.png" alt="Plot: kappa-dot divided by 1 minus kappa rises sharply as t approaches 1, with the vertical axis cut off at 20" width="100%" />
  <figcaption>Figure 10: The coefficient \(\dot\kappa_t / (1 - \kappa_t)\) of the rate as a function of \(t\), with \(\kappa_t = t\) and the vertical axis cut off at 20. Figure source: MIT 6.S184 Lecture 5.</figcaption>
</figure>

> The blow-up corresponds to the endpoint of the path: as \(\kappa_t \to 1\) all probability mass already sits on the data point \(z\), so a sample still holding a wrong token has to be corrected immediately.

### 5.6. Summary of Conditional/Marginal Paths and Rate Matrices

#### Conditional Objects: Closed Forms for the Factorized Mixture Path

| Object | Notation | Key property | Factorized mixture path |
| --- | --- | --- | --- |
| Conditional probability path | \(p_t(x \mid z)\) | Interpolates between \(p_{init}\) and a data point \(z\) | \(\prod_{j=1}^{d}\left[(1 - \kappa_t)\, p_{init}^{(j)}(x_j) + \kappa_t\, \delta_{z_j}(x_j)\right]\) |
| Conditional rate matrix | \(Q_t^z(y \mid x)\) | The CTMC follows the conditional path | \(Q_t^z(y \mid x) = \left(Q_t^z(v_i, j \mid x_j)\right)_{v_i, j}\)<br>\(Q_t^z(v_i, j \mid x_j) = \frac{\dot\kappa_t}{1 - \kappa_t}\left(\delta_{z_j}(v_i) - \delta_{x_j}(v_i)\right)\) |

#### Marginal Objects: Weighted Sums of the Conditional Objects

| Object | Notation | Key property | Formula |
| --- | --- | --- | --- |
| Marginal probability path | \(p_t\) | Interpolates between \(p_{init}\) and \(p_{data}\) | \(p_t(x) = \sum_{z \in S} p_t(x \mid z)\, p_{data}(z)\) |
| Marginal rate matrix | \(Q_t(y \mid x)\) | The CTMC follows the marginal path | \(Q_t(y \mid x) = \sum_{z \in S} Q_t^z(y \mid x)\, \frac{p_t(x \mid z)\, p_{data}(z)}{p_t(x)}\)<br>\(Q_t(v_i, j \mid x) = \frac{\dot\kappa_t}{1 - \kappa_t}\left(p_{1\vert t}(z_j = v_i \mid x) - \delta_{x_j}(v_i)\right)\) |

> \(p_{1\vert t}(z_j = v_i \mid x)\) is the probability that position \(j\) ends at \(v_i\) given the current state \(x\) — the one quantity in the marginal rate matrix that a network has to predict.

<details>
<summary>Derivation: the factorized closed form of the marginal rate matrix</summary>

There is only one thing to prove, the last row of the table: substitute the conditional closed form (5.5) into the marginalization trick (5.3).

Start with what survives the sum. The conditional rate matrix only moves a single token, so the only non-zero term is the one that puts \(v_i\) at position \(j\), namely \(y = (v_i, j)\), and on the conditional side the rate only depends on \(x_j\):

\[
Q_t(v_i, j \mid x) = \sum_{z \in S} Q_t^z(v_i, j \mid x_j)\, \frac{p_t(x \mid z)\, p_{data}(z)}{p_t(x)}
\]

Plugging in (5.5) leaves \(\delta_{z_j}(v_i)\) as the only factor that depends on \(z\), so everything independent of \(z\) comes out of the sum:

\[
Q_t(v_i, j \mid x) = \frac{\dot\kappa_t}{1 - \kappa_t} \sum_{z \in S} \left(\delta_{z_j}(v_i) - \delta_{x_j}(v_i)\right) \frac{p_t(x \mid z)\, p_{data}(z)}{p_t(x)}
\]

Split the sum in two. In the second term \(\delta_{x_j}(v_i)\) does not depend on \(z\), so it comes out and what is left is the posterior summing to one over all data points:

\[
\sum_{z \in S} \frac{p_t(x \mid z)\, p_{data}(z)}{p_t(x)} = \frac{p_t(x)}{p_t(x)} = 1
\]

The first term survives only when \(z_j = v_i\), and what is left is the posterior probability that the data point takes \(v_i\) at position \(j\) given the current state \(x\). Since \(z\) is the value at time 1, it is written \(p_{1\vert t}\):

\[
\sum_{z \in S} \delta_{z_j}(v_i)\, \frac{p_t(x \mid z)\, p_{data}(z)}{p_t(x)} = p_{1\vert t}(z_j = v_i \mid x)
\]

Putting the two terms together gives the closed form from the table:

\[
Q_t(v_i, j \mid x) = \frac{\dot\kappa_t}{1 - \kappa_t}\left(p_{1\vert t}(z_j = v_i \mid x) - \delta_{x_j}(v_i)\right)
\]

Unlike the conditional side, this posterior has no closed form: it is a weighted sum over all data points, which is exactly the quantity training has to fit with a network.

</details>

## 6. Training: Fitting the Posterior with Cross-Entropy

### 6.1. The Discrete Flow Matching Loss: Posterior Prediction as Classification

The closed form (5.5) for the marginal rate matrix leaves exactly one unknown: the posterior probability \(p_{1\vert t}(z_j = v_i \mid x)\), that is, "given the current state \(x\), what is the final value at position \(j\)?" A network is used to fit it, written \(p_{1\vert t}^\theta(z_j \mid x)\) and called the <strong>posterior probability network</strong>.

Training is classification: the noisy state \(x\) and the time \(t\) go into the network, every position emits a distribution over the vocabulary, and the label is the true token \(z_j\) at that position. One cross-entropy per position, added up, gives the Discrete Flow Matching loss

\[
\mathcal{L}_{DFM}(\theta) = \mathbb{E}_{z \sim p_{data},\ t \sim \mathrm{Unif}[0,1],\ x \sim p_t(\cdot \mid z)}\left[\sum_{j=1}^{d} -\log p_{1\vert t}^\theta(z_j \mid x)\right]
\tag{6.1}
\]

The \(x\) in the expectation is drawn from the conditional path \(p_t(\cdot \mid z)\): the conditional path has a closed form, so drawing an \(x\) needs no network.

### 6.2. The Training Algorithm: All Positions from One Forward Pass

| **Algorithm 8** Training a Factorized CTMC Model (Discrete Diffusion) |
|-------------------------------|
| **Input**: dataset of sequences \(z \sim p_{data}\), where \(z = (z_1, \dots, z_d) \in V^d\); initial (noise) token marginals \(p_{init}^{(j)}\) at every position; schedule \(\kappa_t \in [0, 1]\); posterior network \(f_\theta\) returning per-position logits over the vocabulary; optimizer \(\mathrm{OPT}\) |
| 1: **for** each training iteration **do** |
| 2: \(~~~~\)Draw a data point \(z \sim p_{data}\) |
| 3: \(~~~~\)Draw a time \(t \sim \mathrm{Unif}[0, 1]\) and compute \(\kappa \leftarrow \kappa_t\) |
| 4: \(~~~~\)Draw a noisy state \(x \sim p_t(\cdot \mid z)\) (factorized mixture path): |
| 5: \(~~~~~~~~\)**for** \(j = 1, \dots, d\) (in parallel) **do** |
| 6: \(~~~~~~~~~~~~\)Draw a mask \(m_j \sim \mathrm{Bernoulli}(\kappa)\) |
| 7: \(~~~~~~~~~~~~\)Draw a noise token \(\xi_j \sim p_{init}^{(j)}\) |
| 8: \(~~~~~~~~~~~~\)Set \(x_j \leftarrow m_j z_j + (1 - m_j)\xi_j\) |
| 9: \(~~~~~~~~\)**end for** |
| 10: \(~~~~~~~~\)Set \(x \leftarrow (x_1, \dots, x_d)\) |
| 11: \(~~~~\)Predict the terminal-token posteriors from the network logits: \(\ell_j(\cdot) \leftarrow f_\theta(x, t)_j \quad \Rightarrow \quad p_{1\vert t}^\theta(v \mid x)_j = \mathrm{Softmax}(\ell_j)(v)\) |
| 12: \(~~~~\)Discrete Flow Matching loss (token-wise negative log-likelihood of \(z\)): \(\mathcal{L}_{DFM}(\theta) \leftarrow \sum_{j=1}^{d}\left[-\log p_{1\vert t}^\theta(z_j \mid x)_j\right]\) |
| 13: \(~~~~\)Update the parameters: \(\theta \leftarrow \mathrm{OPT.STEP}(\nabla_\theta \mathcal{L}_{DFM}(\theta))\) |
| 14: **end for** |

## 7. Masked Diffusion Language Models (MDLM): Generating Text from All Masks

### 7.1. Vocabulary and Initial Distribution: Marking "Not Decided Yet" with [MASK]

Every \(p_{init}\) so far has been uniform, which for text means an initial state made of random tokens. Masked diffusion language models take a route that fits text better: a special token <strong>[MASK]</strong> is added to the vocabulary. It never appears in real text; it only says that a position is not decided yet. The initial distribution becomes

\[
p_{init} = \delta_{[MASK]}
\]

that is, the whole sequence starts out entirely masked.

The intuition: for text, "a token already decided" and "a position not thought out yet" are two fundamentally different states, and uniform random tokens mix the two together, forcing the model to spend capacity telling which tokens are really noise. Once the noise is marked explicitly, each position on the conditional path has only two possibilities — the true token \(z_j\) or [MASK]. This lines up with the factorized mixture path from §5.5: set \(p_{init}^{(j)} = \delta_{[MASK]}\) and position \(j\) is \(z_j\) with probability \(\kappa_t\), and [MASK] with probability \(1 - \kappa_t\). The term \(\frac{\dot\kappa_t}{1 - \kappa_t}p_{1\vert t}(z_j = v_i \mid x)\) in the marginal rate matrix (5.5) is then the rate at which a still-masked position jumps to token \(v_i\).

> The sampling procedure itself does not change: in Algorithm 7 only step 3, \(X_0 \sim p_{init}\), becomes "the whole sequence is [MASK]". As \(\kappa_t \to 1\) the coefficient \(\dot\kappa_t/(1 - \kappa_t)\) explodes (Figure 10), which corresponds to the remaining masked positions having to be filled in immediately.

### 7.2. The Generation Process: Masks Revealed in Batches into a Full Passage

Sampling starts at \(t = 0\), where \(X_0\) is a full sequence of [MASK]. Each step feeds the current sequence together with the time \(t\) into the network, which returns the posterior distribution \(p_{1\vert t}^\theta(\cdot \mid x)\) at every position; the rate from (5.5) then decides which masked positions jump and to which token, while already revealed tokens stay put.

<figure>
  <img src="../../../posts/discrete_diffusion/masked_lm_generation.gif" alt="Animation of masked diffusion language model generation: the sequence starts fully masked and is revealed step by step at t = 0.3, 0.6, 0.8 and 1.0, with dashes marking positions not yet revealed and a complete passage in the last frame" width="100%" />
  <figcaption>Figure 11: One full generation run of a masked diffusion language model. The sequence starts fully masked and positions are revealed in batches at \(t = 0.3\), \(t = 0.6\), \(t = 0.8\) and \(t = 1.0\); positions not yet revealed are drawn as dashes, whose length matches the length of the true token. Figure source: MIT 6.S184 Lecture 5.</figcaption>
</figure>

What this process looks like at the scale of a real large model is easiest to watch on [LLaDA](https://arxiv.org/abs/2502.09992) (Large Language Diffusion Model), an 8B masked diffusion language model:

<figure>
  <img src="../../../posts/discrete_diffusion/llada_sampling_process.gif" alt="Animation of LLaDA generation: a 7 by 10 grid of tokens starts with every cell marked MASK, each generation step reveals a batch of positions, and cells shade from pale to dark green according to when they were revealed" width="100%" />
  <figcaption>Figure 12: The generation process of LLaDA. Each cell in the grid is one token and each frame is one generation step; the shading runs from pale green (Step 0) to dark green (Step 64) according to when a token was revealed, and cells marked MASK are still hidden. Figure source: <a href="https://github.com/NVlabs/Fast-dLLM">NVlabs/Fast-dLLM</a>, model: LLaDA.</figcaption>
</figure>

The whole process is parallel: one step can reveal several positions at once, instead of pushing forward one token at a time the way an autoregressive model does. The price is that revealing a position depends on the entire current sequence, so each batch of tokens costs one full forward pass.
