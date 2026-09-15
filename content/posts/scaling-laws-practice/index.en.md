---
title: "Scaling Laws in Practice"
date: 2026-08-07T11:30:03+08:00
series:
  main: "Large Language Model"
  subseries: "Pre-training"
categories: ["Large Language Model", "Pre-training"]
tags: ["Scaling Law", "MiniCPM", "DeepSeek", "μP", "WSD", "Compute Optimality"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "How scaling laws guide practical language-model training, from hyperparameter transfer and configuration selection to joint data-model scaling."
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
cover:
    image: "cover.png"
    alt: "MiniCPM joint loss fitting over model parameter count and training compute"
    caption: "MiniCPM jointly fits model size, data volume, and loss from small-scale experiments, then uses the fit to select a compute-optimal configuration. Source: Hu et al., 2024."
    relative: true
    hidden: false
    hiddenInList: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

## 1. From Fitted Curves to Training Decisions

Scaling laws can answer more than how large a model should be. They can also help determine learning rate, batch size, training duration, and the ratio between data volume and model size. Their practical purpose is to turn stable patterns observed in small-scale experiments into decisions for large-scale training.

This article focuses on how scaling laws are used in real training rather than on one particular model. Studies differ in parametrization, experimental design, and fitting method. MiniCPM is the first case: it shows how a final training configuration can be determined step by step from small-model experiments.

## 2. Case Study 1: MiniCPM's Model Wind-Tunnel Experiments

MiniCPM calls this small-scale workflow **model wind-tunnel experiments (MWTE)**: run many inexpensive experiments on small models, then transfer stable relationships to the final model.

MiniCPM has two main versions, 1.2B and 2.4B. The original paper reports that these small language models (SLMs) outperform many similarly sized models on Chinese, English, coding, and mathematics evaluations, with some metrics approaching or exceeding contemporary 7B models.

<figure>
  <img src="../../../posts/scaling-laws-practice/minicpm-benchmark.png" alt="Benchmark comparison of MiniCPM with language models of different sizes">
  <figcaption>Selected benchmark results for MiniCPM-1.2B and MiniCPM-2.4B. Bold indicates the best result among the small models in each group. Source: Hu et al., 2024.</figcaption>
</figure>

This result did not come from merely training a small model for longer. MiniCPM's workflow includes:

1. using \(\mu\)P to stabilize hyperparameter transfer across model sizes;
2. fitting optimal learning rate and batch size with small-model experiments;
3. reducing the cost of joint data–model scaling experiments with a new learning-rate schedule;
4. selecting the final data volume and model size from the fitted relationships.

### 2.1 Designing a Scalable Model Family with \(\mu\)P

#### Determining the Base \(\mu\)P Configuration on a Small Model

When model width is enlarged under a standard parametrization, the same learning rate, initialization scale, and residual scale are usually no longer appropriate. Maximal update parametrization (\(\mu\)P) aims to make activations, update magnitudes, and outputs comparable across widths so that hyperparameters found on a small model transfer to a large one.

MiniCPM first searches for the following base configuration on a 9M-parameter model:

- embedding-output scale <code>scale_emb = 12</code>;
- residual-depth scale <code>scale_depth = 1.4</code>;
- base initialization standard deviation <code>init_std = 0.1</code>;
- base learning rate <code>lr = 0.01</code>.

MiniCPM uses both width and depth scaling. Let current model width be \(d_m\) and base width be \(d_{base}\). Key relationships include:

- reducing the initialization standard deviation of two-dimensional tensors by \(1/\sqrt{d_m/d_{base}}\);
- reducing their applied learning rate by \(1/(d_m/d_{base})\);
- scaling output-layer logits by the same \(1/(d_m/d_{base})\);
- normalizing each layer's residual increment using <code>scale_depth</code> and the number of layers.

<details>
<summary>A Concrete Width-and-Depth Scaling Example</summary>

Suppose the base width is \(d_{base}=512\) and the target width is \(d_m=2048\). The width ratio is

\[
r=\frac{d_m}{d_{base}}=4.
\]

Applying the base configuration to the scaling rules gives

\[
\text{two-dimensional tensor initialization standard deviation}
=\frac{0.1}{\sqrt{4}}
=0.05,
\]

\[
\text{two-dimensional tensor applied learning rate}
=\frac{0.01}{4}
=0.0025.
\]

The language-model output logits are also multiplied by \(1/4\). If the target has \(L=16\) layers, the applied residual-branch coefficient is

\[
\frac{\text{scale\_depth}}{\sqrt{L}}
=\frac{1.4}{\sqrt{16}}
=0.35.
\]

The model still uses the base configuration \(\text{init\_std}=0.1\), \(\text{lr}=0.01\), and \(\text{scale\_depth}=1.4\), but the values actually applied to individual tensors and residual branches shrink according to target width and depth.

</details>

Thus, saying that the optimal base learning rate remains constant does not mean every tensor receives exactly the same numerical learning rate. Rather, \(\mu\)P absorbs width-dependent changes into its parametrization rules.

#### Holding Shape Fixed while Scaling Overall Size

To avoid confounding model size with architectural shape, MiniCPM keeps the relative proportions of width and depth approximately fixed in its wind-tunnel experiments and scales the model as a whole. The experiments cover 9M, 30M, 70M, 0.1B, 0.17B, 0.2B, and 0.5B parameters.

<figure>
  <img src="../../../posts/scaling-laws-practice/scaling-model-configurations.png" alt="Model configurations of different sizes used in MiniCPM wind-tunnel experiments">
  <figcaption>As parameter count grows, hidden dimension, feed-forward dimension, attention-head count, and layer count increase together, keeping model shape broadly similar. Source: Hu et al., 2024.</figcaption>
</figure>

The largest wind-tunnel model has only about 0.5B parameters, while MiniCPM-2.4B is roughly five times larger. This gap tests the extrapolation: the final configuration is predicted from small-model runs instead of exhaustively searched at 2.4B scale.

#### Testing Whether the Optimal Learning Rate Remains Stable across Scale

\(\mu\)P predicts that, after correct scaling, the optimal base learning rate should remain approximately stable with model size. MiniCPM sweeps learning rates for 0.04B, 0.1B, 0.3B, and 0.5B models and validates the result again at 2.1B scale.

<figure>
  <img src="../../../posts/scaling-laws-practice/optimal-learning-rate.png" alt="Learning rate versus loss for MiniCPM models of different sizes">
  <figcaption>Colors denote model sizes. After model size grows by roughly an order of magnitude, loss minima remain concentrated near a base learning rate of 0.01. Source: Hu et al., 2024.</figcaption>
</figure>

The result agrees with \(\mu\)P's expectation: the optimal base learning rate remains around 0.01 as scale changes. The final model can therefore reuse the value found on small models without another expensive full-scale search.

### 2.2 Determining the Optimal Batch Size with Scaling Experiments

A batch that is too small increases the number of updates required to reach a target loss. A batch that is too large may consume more training tokens without a corresponding optimization benefit. MiniCPM tests six batch sizes on 9M-, 30M-, and 170M-parameter models.

<figure>
  <img src="../../../posts/scaling-laws-practice/optimal-batch-size.png" alt="Relationship among batch size, training tokens, and loss at different model sizes">
  <figcaption>The horizontal axis is batch size, the vertical axis processed tokens, and color denotes loss. The red line connects the batch size that consumes the fewest tokens at each target loss; the right panel aggregates these optima. Source: Hu et al., 2024.</figcaption>
</figure>

The three left panels correspond to the three model sizes. For a fixed batch, the vertical sequence of points is a training curve progressing through more tokens. To find the optimal batch:

1. choose a target loss;
2. measure how many tokens each batch size needs to reach it;
3. fit a parabola to these equal-loss points;
4. define the batch at minimum token consumption as optimal for that loss.

Fitting the optima on logarithmic axes gives

\[
B_{\mathrm{opt}}(L)\approx \frac{1.21\times 10^9}{L^{6.24}}.
\]

The optimal batch therefore grows as a power law when target loss falls. “Optimal” here specifically means reaching the target loss with as few training tokens as possible under MiniCPM's fixed compute resources.

<details>
<summary>Why Must the Target Loss Be Known before Estimating Batch Size?</summary>

This appears circular: how can final loss be known before the model is trained? In practice, a rough prediction comes from a smaller model, a shorter run, or a neighboring configuration, and is then used to estimate batch size. The prediction need not be exact because there is usually a relatively flat usable range around the optimum.

Learning rate and batch size are not independent either. MiniCPM first performs a rough learning-rate search, fits batch size using that rate, and then rechecks the learning rate—an alternating procedure similar to coordinate descent. The empirical formula is therefore not a universal law independent of the training configuration.

</details>

### 2.3 Reducing the Cost of Joint Scaling Experiments with WSD

#### Why Intermediate Checkpoints from a Cosine Schedule Are Not Directly Comparable

Chinchilla-style analysis compares many model sizes \(N\) and data volumes \(D\). Training every size from scratch at every target data volume creates \(m^2\) combinations for \(m\) model sizes and \(m\) data volumes.

A seemingly natural alternative is to train one long curve and treat intermediate checkpoints as final results for smaller data budgets. A cosine learning-rate schedule depends on its preset endpoint, however, so the learning rate at an intermediate checkpoint has not decayed fully. That checkpoint is not equivalent to a model whose run was designed to end there.

<figure>
  <img src="../../../posts/scaling-laws-practice/cosine-cycle-length.png" alt="Effect of cosine-cycle length on training and C4 loss">
  <figcaption>A longer cosine cycle leaves a higher learning rate near the target stopping point and produces worse final loss. Chinchilla's experiments show a clear degradation when expected training length is overestimated by more than roughly 25%. Source: Hoffmann et al., 2022.</figcaption>
</figure>

With a cosine schedule, a long run therefore cannot simply be truncated to replace a complete short run designed for that endpoint.

#### Reusing Training with a Warmup–Stable–Decay Schedule

MiniCPM partially solves this problem by explicitly dividing learning rate into warmup, stable, and decay phases:

\[
\operatorname{WSD}(T;s)=
\begin{cases}
\dfrac{s}{W}\eta, & s\lt W,\\
\eta, & W\le s\lt T,\\
f(s-T)\eta, & T\le s\lt S.
\end{cases}
\]

Here, \(s\) is the current step, \(W\) the end of warmup, \(T\) the end of the stable phase, \(S\) the end of training, \(\eta\) the maximum learning rate, and \(f\) a decreasing function of training progress.

<figure>
  <img src="../../../posts/scaling-laws-practice/wsd-schedule.png" alt="A cosine learning-rate curve and WSD curves with two endpoints">
  <figcaption>The WSD curves share the same stable phase and begin decay at different checkpoints, whereas the cosine schedule depends on its endpoint from the start. Source: Hu et al., 2024.</figcaption>
</figure>

The central advantage is that **the stable phase can be reused**. Researchers train one long stable prefix, branch at checkpoints such as 10N, 20N, and 30N, and apply a short decay to each branch. This yields several approximately completed models without restarting training for every data volume.

In MiniCPM's setup, this reduces the cost along the data axis of joint scaling experiments from approximately \(O(m^2C)\) to \(O(mC)\). It does not eliminate training cost; it reuses the most expensive common prefix among data budgets.

#### Empirical Training Behavior under WSD

<figure>
  <img src="../../../posts/scaling-laws-practice/wsd-loss.png" alt="C4 loss under WSD and cosine learning-rate schedules">
  <figcaption>Loss decreases more slowly during WSD's stable phase, then rapidly during decay, eventually matching or beating the corresponding cosine schedule. Source: Hu et al., 2024.</figcaption>
</figure>

MiniCPM reports two observations that directly affect experimental design:

- loss decreases relatively slowly while the stable phase keeps a high learning rate, then falls rapidly once decay begins;
- for checkpoints such as 40N, 60N, and 80N, a decay phase covering about 10% of total training tokens is generally sufficient, while 2.5% is often too short.

WSD's value is not that its entire curve always beats cosine scheduling. It allows the same stable-phase checkpoint either to continue training or to enter a short decay that measures its approximate final performance.

### 2.4 Estimating the Compute-Optimal Data–Model Ratio with Small Models

With WSD, MiniCPM trains six model sizes and decays multiple checkpoints from each stable phase. The resulting observations cover parameter count \(N\), training data \(D\), and final loss \(L\).

The authors use two Chinchilla-style methods: a lower-envelope method and a joint-function fit.

#### Lower Envelope: Directly Finding the Best Model at Fixed Compute

<figure>
  <img src="../../../posts/scaling-laws-practice/lower-envelope.png" alt="Final loss versus compute for MiniCPM models of different sizes">
  <figcaption>Colors denote model sizes. Within one color, additional compute mainly means more training data for the fixed model. The panels show measured losses on code, English WikiHow, and Chinese WikiHow. Source: Hu et al., 2024.</figcaption>
</figure>

Selecting the lowest-loss point across models at every compute budget gives the empirical lower envelope. As budget increases, the optimum switches gradually from small to large models. The switch points indicate the compute-optimal model size and data volume for each budget.

The segments show a clear downward trend on log–log axes but are not perfectly collinear. Within the covered range, experiments also show that adding data to a fixed model continues to produce meaningful gains; marginal returns on the data side diminish less quickly than expected.

#### Joint Function Fitting: Modeling Size and Data Together

MiniCPM's main method directly fits

\[
L(N,D)=C_NN^{-\alpha}+C_DD^{-\beta}+L_0.
\]

The term \(C_NN^{-\alpha}\) is error from finite model size, \(C_DD^{-\beta}\) is error from finite data, and \(L_0\) is the loss floor. The quantities \(C_N\), \(C_D\), \(\alpha\), \(\beta\), and \(L_0\) are fitted jointly from small-scale experiments rather than fixed as universal constants. With the compute approximation

\[
C\approx 6ND,
\]

the optimal fixed-budget configuration is

\[
(N^*,D^*)=\underset{N,D}{\arg\min}\ L(N,D)
\quad \text{s.t.}\quad 6ND\le C.
\]

The model–data ratio can be written as

\[
\frac{N_{\mathrm{opt}}}{D_{\mathrm{opt}}}
=K^2\left(\frac{C}{6}\right)^\eta,
\qquad
K=\left(\frac{\alpha C_N}{\beta C_D}\right)^{\frac{1}{\alpha+\beta}},
\qquad
\eta=\frac{\beta-\alpha}{\alpha+\beta}.
\]

When \(\alpha\approx\beta\), \(\eta\approx0\) and the optimal data–model ratio changes little with compute. When they differ, a growing budget gradually favors either data or model size.

<figure>
  <img src="../../../posts/scaling-laws-practice/joint-fit.png" alt="Joint loss contours over MiniCPM model size and training compute on six evaluation corpora">
  <figcaption>The horizontal axis is non-embedding parameter count and the vertical axis training compute. Black points are checkpoints actually decayed, while background contours come from the joint loss fit. Source: Hu et al., 2024.</figcaption>
</figure>

The fitted data–model ratios are substantially higher than the often cited Chinchilla value of roughly 20 tokens per parameter. In the UltraText panel, the ratio is about 95.6 tokens per parameter at \(C=10^{21}\) FLOPs; averaged over the six evaluation corpora, it is about 191.9.

This should not be treated as a new universal constant. It depends on MiniCPM's architecture, data, tokenizer, training recipe, and loss definition, and the paper does not fully validate the ratio on larger models. The safer conclusion is that **the optimal data volume under modern recipes may substantially exceed the early Chinchilla allocation, and WSD provides a cheaper way to remeasure it.**

## 3. Case Study 2: DeepSeek LLM's Direct Fitting Approach

DeepSeek LLM also uses small-scale experiments to guide large-model training, but follows a different route from MiniCPM. It does not transfer hyperparameters with \(\mu\)P. Instead, it directly estimates optimal batch size, learning rate, and data–model allocation at different compute budgets and extrapolates the fits to 7B and 67B models.

The original paper reports strong overall performance for DeepSeek LLM 7B and 67B among contemporary open models. More relevant here is how their large-scale configurations were obtained from cheaper scaling experiments.

### 3.1 Directly Fitting Batch Size and Learning Rate

DeepSeek first performs a grid search over batch size \(B\) and learning rate \(\eta\) in small-scale runs at \(10^{17}\) FLOPs. Every cell contains validation loss for one configuration; the horizontal axis is learning rate and the vertical axis token-based batch size.

<figure>
  <img src="../../../posts/scaling-laws-practice/deepseek-hyperparameter-grid.png" alt="DeepSeek grid searches over batch size and learning rate at two compute budgets">
  <figcaption>The left panel uses \(10^{17}\) FLOPs and 177M FLOPs per token. The right validates the fit at \(10^{20}\) FLOPs; the star lies in the low-loss region. Source: DeepSeek-AI, 2024.</figcaption>
</figure>

The area near minimum loss is broad and flat rather than a sharp point. Training does not need to hit a unique optimal pair exactly; a range of batch sizes and learning rates gives near-optimal results.

DeepSeek defines configurations within \(0.25\%\) of the minimum loss as **near-optimal hyperparameters**, then fits them as power laws of compute \(C\):

\[
\eta_{\mathrm{opt}}=0.3118\,C^{-0.1250},
\]

\[
B_{\mathrm{opt}}=0.2920\,C^{0.3271}.
\]

<figure>
  <img src="../../../posts/scaling-laws-practice/deepseek-hyperparameter-scaling.png" alt="DeepSeek fits optimal batch size and learning rate as functions of training compute">
  <figcaption>Gray points are near-optimal configurations from small-scale experiments, dashed lines are power-law fits, and gray bands show the broad near-optimal ranges. Blue stars are the configurations used for the 7B and 67B models. Source: DeepSeek-AI, 2024.</figcaption>
</figure>

The fitted trend is that **larger compute budgets favor larger batches and smaller learning rates**. The DeepSeek LLM 7B and 67B configurations also fall within the extrapolated near-optimal bands.

The coefficients and exponents are empirical values for this setup, not universal constants. In particular, the learning-rate experiments cover only several discrete values and the near-optimal band is wide, so the curve is better used to choose a workable range than as a high-precision prediction.

### 3.2 Reusing Scaling Experiments with a Multi-Step Learning-Rate Schedule

To reuse training across compute budgets, DeepSeek replaces an endpoint-dependent cosine schedule with a **multi-step learning-rate scheduler**:

1. warm the learning rate up to its maximum over 2,000 steps;
2. keep the first stage for the first 80% of training tokens;
3. after 80%, reduce the learning rate to 31.6% of the maximum;
4. after 90%, reduce it again to 10% of the maximum.

The last two stages each cover 10% of training, giving the default \(80\%+10\%+10\%\) split. The idea resembles WSD: reuse a long training prefix and decay near the target budget. It uses two discrete drops, however, and is not the same curve as MiniCPM's WSD.

<figure>
  <img src="../../../posts/scaling-laws-practice/deepseek-lr-schedule.png" alt="Loss curves for DeepSeek's multi-step and cosine learning-rate schedules">
  <figcaption>Left: the multi-step and cosine schedules follow different trajectories but reach similar final loss. Right: comparison of 80%+10%+10%, 70%+15%+15%, and 60%+20%+20% stage splits on a 1.6B model trained for 100B tokens. Source: DeepSeek-AI, 2024.</figcaption>
</figure>

A longer decay may slightly improve final loss but reduces the reusable fraction of the common prefix. DeepSeek selects \(80\%+10\%+10\%\) as a compromise between final performance and experimental reuse.

### 3.3 Extrapolating Data–Model Allocation and Final Loss with IsoFLOP Profiles

After obtaining empirical equations for batch size and learning rate, DeepSeek uses Chinchilla's **IsoFLOP profile** method to select model size and data volume.

Instead of parameter count \(N\), it measures model scale by non-embedding compute per token \(M\), or **non-embedding FLOPs per token**. This includes attention compute while excluding vocabulary computation, which contributes relatively little to model capability. Total training compute is

\[
C=MD,
\]

where \(D\) is training-token count. The experiments use eight budgets from \(10^{17}\) to \(3\times10^{20}\) FLOPs and test roughly ten model–data allocations at each budget. The lowest point on every IsoFLOP curve gives the optimal \(M\) and \(D\) for that budget.

<figure>
  <img src="../../../posts/scaling-laws-practice/deepseek-isoflop.png" alt="DeepSeek IsoFLOP curves and power-law extrapolation of optimal model compute and data volume">
  <figcaption>The left panel finds minimum validation loss at each fixed compute budget. The middle and right panels fit optimal per-token model compute and token count. Gray points are small-scale experiments and the blue line extrapolates to DeepSeek LLM 67B. Source: DeepSeek-AI, 2024.</figcaption>
</figure>

The fits are

\[
M_{\mathrm{opt}}=0.1715\,C^{0.5243},
\]

\[
D_{\mathrm{opt}}=5.8316\,C^{0.4757}.
\]

The exponents sum to approximately one, consistent with \(C=MD\), and both are close to \(0.5\). New compute is divided approximately evenly between model scale and training data. Extrapolating to a budget of \(4.5\times10^{23}\) FLOPs gives the figure's optimum of roughly \(4.3\times10^{11}\) FLOPs per token and \(1.04\times10^{12}\) training tokens.

Finally, DeepSeek fits optimal validation loss against compute and predicts the final losses of the 7B and 67B models from small-scale runs.

<figure>
  <img src="../../../posts/scaling-laws-practice/deepseek-loss-prediction.png" alt="DeepSeek predicts the loss of 7B and 67B models from a small-scale fitted curve">
  <figcaption>Gray points and the dashed curve come from small-scale experiments. Blue stars denote DeepSeek LLM 7B and 67B; both models' validation bits per byte lie close to the extrapolated curve. Source: DeepSeek-AI, 2024.</figcaption>
</figure>

Although this extrapolation spans roughly a thousand-fold compute increase, it predicts final loss accurately. Scaling experiments can therefore do more than select \(M\) and \(D\): when the training recipe, data distribution, and evaluation method remain consistent, the same small runs can estimate the performance of a large training run.

The accuracy remains an empirical result for this range. Changes to architecture, data quality, or training recipe require batch size, learning rate, optimal allocation, and loss curves to be recalibrated.

## 4. From Hyperparameters to Architecture Selection: More Applications

These cases share one process: define the relevant scale variables, run controlled experiments at affordable sizes, fit empirical relationships, and extrapolate to the target budget. The horizontal axis and optimization objective must be redefined for the actual engineering problem.

### 4.1 Qwen: Predicting Hyperparameters across Architectures and Training Stages

Qwen2.5 uses scaling laws to predict optimal learning rate \(\mu_{\mathrm{opt}}\) and batch size \(B_{\mathrm{opt}}\). It first runs small-scale experiments across model sizes, data volumes, and architectures, then selects training configurations separately for dense and mixture-of-experts (MoE) models.[6]

Qwen3 additionally includes the training stage, choosing learning-rate schedules and batch sizes separately for general pre-training, reasoning reinforcement learning, and long-context training.[7] The central idea is that **learning rate and batch size need not be copied as constants when scaling a model; they too can be fitted and extrapolated from small-scale experiments.** Neither report publishes the complete fitted equations and observations, however, so the detailed procedure is not currently reproducible.

### 4.2 Kimi K2: Selecting MoE Sparsity and Attention Heads with Scaling Experiments

For an MoE model, total parameters and the parameters active for one token are different. Kimi K2 defines **sparsity** as

\[
S=\frac{E_{\mathrm{total}}}{E_{\mathrm{active}}}.
\]

Its experiments hold eight active experts per token fixed and increase the total expert count. At comparable active parameters and training FLOPs, higher sparsity generally reduces validation loss, but also complicates routing, communication, and load balancing. Kimi K2 ultimately chooses sparsity 48, activating eight of 384 experts.[8]

<figure>
  <img src="../../../posts/scaling-laws-practice/kimi-k2-scaling-decisions.png" alt="Kimi K2 scaling experiments over MoE sparsity and attention-head count">
  <figcaption>Left: with active experts fixed, a larger total expert count and hence greater sparsity generally lower validation loss. Right: comparison of attention-head counts equal to the layer count and twice the layer count. Source: Kimi Team, 2025.</figcaption>
</figure>

Likewise, doubling the number of attention heads relative to layers improves validation loss by only about \(0.5\%\)–\(1.2\%\) while potentially increasing long-context inference cost substantially, so the final model uses 64 heads. The key point is that **scaling experiments measure the performance gain, but the final choice must still account for training and inference cost.**

### 4.3 Hunyuan and LLaMA 3: Compute-Optimal Allocation and Downstream Prediction

Hunyuan-Large applies the IsoFLOP method to an MoE model. Because each token passes through only some experts, model scale is measured by **activated parameters** rather than total parameters. The experiments identify the lowest-loss active count at every compute budget, then fit compute-optimal active parameters and training data.[9]

<figure>
  <img src="../../../posts/scaling-laws-practice/hunyuan-isoflop.png" alt="Hunyuan-Large fits optimal activated parameter count from IsoFLOP curves">
  <figcaption>The left panel uses quadratic curves to find the optimal active count at different budgets; the right fits the minima into a compute-to-active-parameter scaling relationship. Source: Hunyuan Team, 2024.</figcaption>
</figure>

The fitted optimum is approximately 58.1B active parameters and 5.6T tokens, or about 96 tokens per active parameter. Because the IsoFLOP curve is flat near its minimum, Hunyuan-Large ultimately uses 52B active parameters and roughly 7T tokens instead of copying the fitted point exactly.

LLaMA 3 conducts a similar IsoFLOP study for dense models.[10]

<figure>
  <img src="../../../posts/scaling-laws-practice/llama3-isoflop.png" alt="LLaMA 3 IsoFLOP curves and fitted optimal training-token count">
  <figcaption>Each curve on the left is a fixed compute budget and the diamond marks the minimum of a quadratic fit. The right panel fits optimal token count against compute. Source: Llama Team, 2024.</figcaption>
</figure>

Extrapolation recommends roughly 402B parameters and 16.55T tokens. The actual flagship model uses 405B parameters and 15.6T tokens, about 39 tokens per parameter. LLaMA 3 also observes that high-budget IsoFLOP curves are flat near their minima, so several nearby allocations can be near-optimal.

LLaMA 3 extends scaling prediction from pre-training loss to downstream tasks. It first fits training FLOPs against normalized negative log-likelihood (NLL) of the correct answer on ARC Challenge, then fits an S-shaped relationship from NLL to accuracy:

\[
\text{training compute}\longrightarrow\text{normalized NLL}\longrightarrow\text{downstream accuracy}.
\]

<figure>
  <img src="../../../posts/scaling-laws-practice/llama3-downstream-scaling.png" alt="LLaMA 3 predicts ARC Challenge accuracy from training compute and negative log-likelihood">
  <figcaption>The left panel maps training compute to normalized NLL on ARC Challenge; the right maps NLL to accuracy. Extrapolation is close to the measured LLaMA 3 405B result. Source: Llama Team, 2024.</figcaption>
</figure>

The method predicts 405B performance accurately. It does not assume that downstream accuracy directly follows a power law. Instead, it first predicts a stable loss metric and then converts it to task accuracy through a calibration relationship.

### 4.4 MiniMax-01: Comparing Attention Architectures with Scaling Laws

MiniMax-01 applies scaling laws directly to architecture selection, comparing[11]

- standard Softmax attention;
- pure Lightning Attention;
- Hybrid-lightning Attention, which retains one Softmax layer in every eight layers.

Experiments cover models from 70M to 7B parameters, each trained on up to 300B tokens. The study then uses Chinchilla's lower-envelope method to fit optimal loss, model size, and data volume against compute for each architecture.

<figure>
  <img src="../../../posts/scaling-laws-practice/minimax-architecture-scaling.png" alt="Scaling-law comparison of three attention architectures in MiniMax-01">
  <figcaption>The table gives fitted equations for Softmax, Lightning, and Hybrid-lightning Attention. The plots compare fixed-budget loss, optimal model size, and optimal token count. Source: MiniMax, 2025.</figcaption>
</figure>

All three loss–compute exponents are close to \(-0.08\), indicating that Lightning and Hybrid-lightning do not diverge clearly from Softmax scaling over the tested range. At equal budget, Hybrid-lightning has the lowest fitted loss, while pure Lightning favors more parameters and training tokens.

Architecture choice cannot rely on training loss alone. Pure Lightning Attention is compute-efficient but weaker on retrieval tasks. Periodically inserting Softmax attention restores long-context retrieval in the hybrid architecture. The scaling experiment does not independently prove that one architecture is best; it establishes that the hybrid does not degrade markedly as scale grows, after which downstream capability and speed tests guide the final choice.

Together, these cases show that scaling laws have evolved from answering “how large should the model be?” into a general experimental decision tool. The important step is not copying a fixed ratio but designing small-scale experiments around the actual bottleneck: training constraints motivate fits for learning rate and batch size, MoE constraints require sweeps over sparsity and active parameters, and deployment constraints require considering training loss together with inference cost and downstream performance.

## 5. Optimizer and Hyperparameter Scaling

The best learning rate, batch size, and optimizer also depend on scale. A configuration that is optimal for a small model may cease to be optimal after the model or training dataset grows; an optimizer that leads in small experiments may likewise lose its advantage at larger scales.

Optimizer studies therefore need to answer two questions: how can optimal hyperparameters from small experiments be extrapolated to the target training run, and how should optimizers be compared fairly across training scales?

### 5.1 Optimal Hyperparameter Scaling Laws: Step Law

Step Law expresses the peak learning rate \(\eta\) and batch size \(B\) jointly as functions of model size \(N\) and data volume \(D\).[12]

With \(N\) and \(D\) fixed, validation loss forms an approximately convex surface over learning rate and batch size. Moving away from its minimum generally increases loss, while the neighborhood around the minimum forms a relatively flat near-optimal region.

<figure>
  <img src="../../../posts/scaling-laws-practice/step-law-loss-landscape.png" alt="Loss surface over learning rate and batch size at fixed model size and data volume">
  <figcaption>After fixing either learning rate or batch size, loss slices are approximately bowl-shaped; the three-dimensional plot on the right shows the loss surface jointly determined by both variables. This convexity is an empirical observation from a large grid of experiments, not a mathematical theorem valid for every model and training recipe. Source: Li et al., 2025.</figcaption>
</figure>

The paper fits the following laws:

\[
\eta_{\mathrm{opt}}(N,D)=1.79N^{-0.713}D^{0.307},
\]

\[
B_{\mathrm{opt}}(D)=0.58D^{0.571}.
\]

Here, \(N\) is the number of model parameters excluding the vocabulary, \(D\) is the number of training tokens, and \(B\) is batch size measured in tokens. The equations reveal three trends: larger models favor smaller optimal learning rates; for a fixed model, more data raises the optimal learning rate; and optimal batch size primarily grows with data volume.

<figure>
  <img src="../../../posts/scaling-laws-practice/step-law-scaling-trends.png" alt="Step Law fits for optimal learning rate and batch size as model size and data volume change">
  <figcaption>The top row increases data volume at several fixed model sizes, while the bottom row increases model size at several fixed data volumes. Batch size is affected mainly by data volume, whereas learning rate depends on both model size and data volume. Dashed lines are fitted predictions and shaded regions indicate uncertainty. Source: Li et al., 2025.</figcaption>
</figure>

“More data requires a larger learning rate” is not a universal rule. Step Law uses AdamW, 2,000 warm-up steps, and cosine decay to a fixed minimum learning rate, so the positive exponent on \(D\) may partly reflect the schedule. A different schedule such as WSD must be validated again. The coefficients also depend on units, model, data, and training recipe and should not be copied directly into another project.

At the paper's 1B-parameter, 100B-token test point, the configuration predicted by the law produces a loss only about \(0.094\%\) above the global optimum found by exhaustive search. The authors also test MoE models at different sparsities and three data recipes; the predictions remain close to low-loss regions.

<figure>
  <img src="../../../posts/scaling-laws-practice/step-law-robustness.png" alt="Step Law loss contours under different MoE sparsities and data recipes">
  <figcaption>The top row compares different MoE sparsities, while the bottom row compares bilingual, code-augmented, and code-dominant data recipes. Red crosses mark grid-search minima and yellow stars mark Step Law predictions. The formula is reasonably robust within the tested range, but this does not justify applying it to arbitrary training runs without validation. Source: Li et al., 2025.</figcaption>
</figure>

### 5.2 Muon: From Matrix Updates to Large-Scale Training

The central conclusion of this section is that <strong>Muon has demonstrated an ability to scale from small models to large training runs, but its advantage over AdamW changes with training scale.</strong> Token efficiency, per-step overhead, and training stability must all be separated when evaluating that advantage.

Muon is primarily applied to two-dimensional weight matrices. Given gradient \(G_t\) at step \(t\), it first accumulates a momentum matrix \(B_t\), then uses Newton–Schulz iterations to approximate an orthogonalized update \(O_t\):[15]

\[
\begin{aligned}
B_t&=\mu B_{t-1}+G_t,\\
B_t&=U\Sigma V^\top,\\
O_t&\approx UV^\top,\\
W_t&=W_{t-1}-\eta O_t.
\end{aligned}
\]

The singular value decomposition (SVD) explains the update: in \(U\Sigma V^\top\), \(\Sigma\) represents update strength in different directions, whereas \(UV^\top\) retains those directions while bringing every nonzero singular value close to \(1\). An implementation does not compute an exact SVD at every step; a small number of Newton–Schulz iterations approximates \(UV^\top\).

<figure>
  <img src="../../../posts/scaling-laws-practice/optimizer-wallclock.png" alt="Validation loss against wall-clock time for Adam, Shampoo, Soap, and Muon">
  <figcaption>In small NanoGPT experiments, Muon takes about 142 ms per step, close to Adam's 139 ms, while reaching the same validation loss sooner. This shows a wall-clock advantage for Muon in this implementation, but it does not directly establish the same result for large-model training.</figcaption>
</figure>

Determining whether this advantage persists at scale requires tuning each optimizer separately and repeating the comparison across model sizes and data-to-model ratios.[13] Sharing one learning rate and weight decay is not a fair comparison because different optimizers may have very different near-optimal ranges for both hyperparameters.

<figure>
  <img src="../../../posts/scaling-laws-practice/optimizer-scaling-summary.png" alt="Summary of optimizer tuning and performance across model scales and Chinchilla ratios">
  <figcaption>The top row shows why learning rate and weight decay must be searched separately for each optimizer. The lower-left plot shows that Muon and Soap's token-efficiency advantages over AdamW shrink as model size grows; the lower-right plot compares matrix optimizers under different data-to-model ratios. Source: Wen et al., 2025.</figcaption>
</figure>

Experiments across scales reveal two patterns:

- On models smaller than 1B parameters, Muon and Soap achieve about \(1.3\text{--}1.4\times\) the token efficiency of AdamW; by 1.2B parameters, this advantage falls to roughly \(1.1\times\).
- The data-to-model ratio can change the ranking: Muon performs better at lower ratios, while Soap surpasses it in a 300M model trained at 16 times the Chinchilla ratio.

Kimi K2 supplies evidence at a much larger scale. It is a mixture-of-experts model with about 1.04T total parameters and roughly 32B active parameters per token. It completes pre-training on 15.5T tokens using MuonClip, which adds a QK-Clip stabilization mechanism.[8]

<figure>
  <img src="../../../posts/scaling-laws-practice/kimi-k2-training-loss.png" alt="Per-step training loss of Kimi K2 over 15.5 trillion training tokens">
  <figcaption>Kimi K2's unsmoothed, non-downsampled per-step training loss decreases throughout training without loss spikes. Source: Kimi Team, 2025.</figcaption>
</figure>

This curve demonstrates that MuonClip can train stably at very large scale. It does not, however, include an AdamW control run of the same scale and configuration, so it cannot by itself establish Muon's speedup ratio. A more precise conclusion is that <strong>Muon's scalability has been validated in engineering practice, while its relative benefit must still be measured in controlled experiments.</strong>

Furthermore, a reported “speedup” often means that fewer tokens are needed to reach the same validation loss; it does not imply that wall-clock training time falls by the same proportion. A reliable comparison should, at minimum, tune every optimizer thoroughly, cover multiple training scales, and report both token efficiency and wall-clock time.

### 5.3 Theoretical Basis and Applicability of μP

Maximal Update Parametrization (μP) can be understood through a spectral condition for feature learning.[18] Let the width of layer \(l\) be \(n_l\), with activation vector \(h_l\). As model width changes, μP aims to satisfy two conditions:

\[
\begin{aligned}
\text{A1: }&(h_l)_i=\Theta(1),\\
\text{A2: }&(\Delta h_l)_i=\Theta(1).
\end{aligned}
\]

Here, \(l\) indexes layers and \(i\in\{1,\ldots,n_l\}\) indexes a neuron or feature coordinate in layer \(l\). Thus, \((h_l)_i\) is the \(i\)-th activation in layer \(l\), while \((\Delta h_l)_i\) is its change after one gradient update.

A1 requires each activation at initialization to neither explode nor vanish as width grows. A2 imposes the same requirement on the change in each activation after a gradient update. Here, \(\Theta(1)\) only means remaining at the same order with respect to width; it does not require the numerical value to equal \(1\).

If every component of a width-\(n_l\) vector is \(\Theta(1)\), its Euclidean norm should satisfy

\[
\begin{aligned}
\lVert h_l\rVert_2&=\Theta(\sqrt{n_l}),\\
\lVert\Delta h_l\rVert_2&=\Theta(\sqrt{n_l}).
\end{aligned}
\]

These two vector scales become the targets of the initialization and parameter-update derivations below.

#### Condition A1: Deriving Initialization Scale from Activation Scale

First consider a deep linear network:

\[
\begin{aligned}
h_l&=W_lh_{l-1}.
\end{aligned}
\]

Let \(W_l\in\mathbb{R}^{n_l\times n_{l-1}}\), with each entry independently sampled from \(\mathcal{N}(0,\sigma_l^2)\). We use \(\lVert W_l\rVert_2\) for the matrix spectral norm.

<details>
<summary><strong>What is the spectral norm?</strong></summary>

The spectral norm of a matrix \(W\) is defined as

\[
\begin{aligned}
\lVert W\rVert_2
&=\max_{x\ne 0}\frac{\lVert Wx\rVert_2}{\lVert x\rVert_2}.
\end{aligned}
\]

It is the largest factor by which the matrix can stretch the length of an input vector. Consequently, every vector \(x\) satisfies

\[
\begin{aligned}
\lVert Wx\rVert_2
&\leq\lVert W\rVert_2\lVert x\rVert_2.
\end{aligned}
\]

The spectral norm equals the largest singular value:

\[
\begin{aligned}
\lVert W\rVert_2
&=\sigma_{\max}(W)
=\sqrt{\lambda_{\max}(W^{\top}W)}.
\end{aligned}
\]

The inequality becomes an equality only when \(x\) points along a right singular vector associated with the largest singular value. The subscript \(2\) is used here for the spectral norm to avoid confusion with the nuclear norm \(\lVert W\rVert_*\), which commonly denotes the sum of all singular values.

</details>

By random-matrix concentration, the spectral norm of \(W_l\) is approximately

\[
\begin{aligned}
\lVert W_l\rVert_2
&\approx
\sigma_l\left(\sqrt{n_{l-1}}+\sqrt{n_l}\right).
\end{aligned}
\]

We can therefore estimate the scale of the current layer's activations using

\[
\begin{aligned}
\lVert h_l\rVert_2
&\approx
\lVert W_l\rVert_2\lVert h_{l-1}\rVert_2.
\end{aligned}
\]

To map \(\Theta(\sqrt{n_{l-1}})\) in the preceding layer to \(\Theta(\sqrt{n_l})\) in the current layer, choose

\[
\begin{aligned}
\sigma_l
&=
\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}
\left(\sqrt{n_l}+\sqrt{n_{l-1}}\right)^{-1}\\
&=
\Theta\!\left(
\frac{1}{\sqrt{n_{l-1}}}
\min\!\left(1,\sqrt{\frac{n_l}{n_{l-1}}}\right)
\right).
\end{aligned}
\]

Now make the induction hypothesis

\[
\begin{aligned}
\lVert h_{l-1}\rVert_2
&=\Theta(\sqrt{n_{l-1}}).
\end{aligned}
\]

This initialization makes the spectral norm of the weight matrix satisfy

\[
\begin{aligned}
\lVert W_l\rVert_2
&\approx
\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}
\left(
\sqrt{n_l}+\sqrt{n_{l-1}}
\right)^{-1}
\left(
\sqrt{n_{l-1}}+\sqrt{n_l}
\right)\\
&=
\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}},
\end{aligned}
\]

and consequently

\[
\begin{aligned}
\lVert h_l\rVert_2
&=\sqrt{n_l}+o(\sqrt{n_l}).
\end{aligned}
\]

Each activation in the current layer therefore remains \(\Theta(1)\). When adjacent layers have equal width, \(n_l=n_{l-1}=n\), the expression simplifies to \(\sigma_l=\Theta(1/\sqrt n)\), matching the order of common width-aware initializations.

This spectral-norm argument is biased toward the worst case. Strictly speaking, \(\lVert W_lh_{l-1}\rVert_2\) need not equal \(\lVert W_l\rVert_2\lVert h_{l-1}\rVert_2\); the latter primarily provides an upper bound.

#### Condition A2: Deriving Weight-Update Scale from Activation Change

Next consider parameter updates. For a linear layer trained with stochastic gradient descent (SGD), the weight update is a rank-one outer product of the loss gradient and the preceding layer's activations:

\[
\begin{aligned}
\Delta W_l
&=-\eta_l\nabla_{h_l}\ell\,h_{l-1}^{\top}.
\end{aligned}
\]

<details>
<summary><strong>Why is \(\Delta W_l\) a rank-one outer product, and why is its right-hand direction \(h_{l-1}\)?</strong></summary>

Denote the gradient arriving at the current layer during backpropagation by

\[
\begin{aligned}
g_l&=\nabla_{h_l}\ell.
\end{aligned}
\]

Because \(h_l=W_lh_{l-1}\), its \(i\)-th output is

\[
\begin{aligned}
(h_l)_i
&=\sum_j(W_l)_{ij}(h_{l-1})_j.
\end{aligned}
\]

The gradient of each weight is therefore

\[
\begin{aligned}
\frac{\partial\ell}{\partial(W_l)_{ij}}
&=\frac{\partial\ell}{\partial(h_l)_i}(h_{l-1})_j\\
&=(g_l)_i(h_{l-1})_j.
\end{aligned}
\]

Writing all entries back into matrix form gives

\[
\begin{aligned}
\nabla_{W_l}\ell
&=g_lh_{l-1}^{\top},\\
\Delta W_l
&=-\eta_lg_lh_{l-1}^{\top}.
\end{aligned}
\]

Thus, \(\Delta W_l\) is a rank-one matrix of the form \(uv^\top\), with \(u=-\eta_lg_l\) and \(v=h_{l-1}\). The normalized \(h_{l-1}\) is its sole nonzero right singular direction. This is a simplified result for a single sample and a linear layer; a batch gradient is a sum of outer products and need not remain rank one.

</details>

In this single-sample rank-one derivation, \(h_{l-1}\) is a right singular direction of \(\Delta W_l\), so

\[
\begin{aligned}
\lVert\Delta W_lh_{l-1}\rVert_2
&=\lVert\Delta W_l\rVert_2\lVert h_{l-1}\rVert_2.
\end{aligned}
\]

<details>
<summary><strong>Why does the spectral-norm upper bound become an equality here?</strong></summary>

Using the outer-product form above, write \(\Delta W_l=uv^\top\), where \(u=-\eta_lg_l\) and \(v=h_{l-1}\). The spectral norm of a rank-one matrix \(uv^\top\) satisfies

\[
\begin{aligned}
\lVert uv^{\top}\rVert_2
&=\lVert u\rVert_2\lVert v\rVert_2.
\end{aligned}
\]

Substituting \(v=h_{l-1}\) yields

\[
\begin{aligned}
\lVert\Delta W_lh_{l-1}\rVert_2
&=\lVert uv^{\top}v\rVert_2\\
&=\lVert u\rVert_2\lVert v\rVert_2^2\\
&=\lVert\Delta W_l\rVert_2\lVert h_{l-1}\rVert_2.
\end{aligned}
\]

For a general matrix, only \(\lVert Ax\rVert_2\leq\lVert A\rVert_2\lVert x\rVert_2\) is guaranteed. Equality holds here because \(h_{l-1}\) points exactly along the leading right singular direction of the rank-one matrix \(\Delta W_l\).

</details>

After both the weights and preceding-layer activations are updated, the change in the current layer's activations is

\[
\begin{aligned}
\Delta h_l
&=W_l\Delta h_{l-1}
+\Delta W_l(h_{l-1}+\Delta h_{l-1}).
\end{aligned}
\]

<details>
<summary><strong>How is this activation-change equation expanded?</strong></summary>

The updated weights and preceding-layer activations are

\[
\begin{aligned}
W_l'&=W_l+\Delta W_l,\\
h_{l-1}'&=h_{l-1}+\Delta h_{l-1}.
\end{aligned}
\]

The updated current-layer activation is therefore

\[
\begin{aligned}
h_l'
&=(W_l+\Delta W_l)(h_{l-1}+\Delta h_{l-1})\\
&=W_lh_{l-1}
+W_l\Delta h_{l-1}
+\Delta W_lh_{l-1}
+\Delta W_l\Delta h_{l-1}.
\end{aligned}
\]

By definition, \(\Delta h_l=h_l'-h_l\). Subtracting the original activation \(h_l=W_lh_{l-1}\) gives

\[
\begin{aligned}
\Delta h_l
&=W_l\Delta h_{l-1}
+\Delta W_lh_{l-1}
+\Delta W_l\Delta h_{l-1}\\
&=W_l\Delta h_{l-1}
+\Delta W_l(h_{l-1}+\Delta h_{l-1}).
\end{aligned}
\]

The three terms respectively represent feature changes from earlier layers propagating into the current layer, the direct effect of updating the current layer's weights, and the cross term caused by changing both weights and inputs. If earlier layers do not change, so \(\Delta h_{l-1}=0\), only \(\Delta h_l=\Delta W_lh_{l-1}\) remains.

</details>

Assume the leading terms do not cancel one another. By the induction hypothesis and condition A1, the first term satisfies

\[
\begin{aligned}
\lVert W_l\Delta h_{l-1}\rVert_2
&=\Theta(\sqrt{n_l}).
\end{aligned}
\]

For the leading part of the second term, \(\Delta W_lh_{l-1}\), to also reach \(\Theta(\sqrt{n_l})\), we need

\[
\begin{aligned}
\lVert\Delta W_l\rVert_2\sqrt{n_{l-1}}
&=\Theta(\sqrt{n_l}),
\end{aligned}
\]

or equivalently

\[
\begin{aligned}
\lVert\Delta W_l\rVert_2
&=\Theta\!\left(\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}\right).
\end{aligned}
\]

The cross term \(\Delta W_l\Delta h_{l-1}\) is treated as lower order:

\[
\begin{aligned}
\lVert\Delta W_l\Delta h_{l-1}\rVert_2
&=o\!\left(\lVert\Delta W_l\rVert_2\sqrt{n_{l-1}}\right).
\end{aligned}
\]

Condition A2 has therefore been converted into a requirement on the spectral norm of the weight update. The change in each individual weight is not the final objective; what matters is whether applying \(\Delta W_l\) to the activations produces a nondegenerate feature change of order \(\Theta(\sqrt{n_l})\).

#### Deriving the SGD Learning Rate from Weight-Update Scale

Finally, choose \(\eta_l\) so that the update condition above holds. If the single-step loss change remains \(O(1)\), a first-order approximation gives

\[
\begin{aligned}
\Delta\ell
&\approx
\Theta\!\left(\left\langle\Delta W_l,\nabla_{W_l}\ell\right\rangle\right)\\
&=\Theta\!\left(\lVert\Delta W_l\rVert_F
\lVert\nabla_{W_l}\ell\rVert_F\right)\\
&=\Theta\!\left(\lVert\Delta W_l\rVert_2
\lVert\nabla_{W_l}\ell\rVert_2\right).
\end{aligned}
\]

The last equality uses the fact that both the gradient and update are rank-one matrices here, for which the Frobenius norm equals the spectral norm. Substituting \(\Delta\ell=O(1)\) and the update scale obtained above gives

\[
\begin{aligned}
\lVert\nabla_{W_l}\ell\rVert_2
&=\Theta\!\left(\frac{\sqrt{n_{l-1}}}{\sqrt{n_l}}\right).
\end{aligned}
\]

Standard SGD uses \(\Delta W_l=-\eta_l\nabla_{W_l}\ell\). To obtain

\[
\begin{aligned}
\lVert\Delta W_l\rVert_2
&=\Theta\!\left(\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}\right),
\end{aligned}
\]

the learning rate must scale as

\[
\begin{aligned}
\eta_l
&=\Theta\!\left(\frac{n_l}{n_{l-1}}\right).
\end{aligned}
\]

When input and output widths grow proportionally, \(n_l/n_{l-1}\) stays constant, so the order of the SGD learning rate with respect to width also remains constant. Adam changes the mapping from gradients to parameter updates, but the target condition remains

\[
\begin{aligned}
\lVert\Delta W_l\rVert_2\sqrt{n_{l-1}}
&=\Theta(\sqrt{n_l}).
\end{aligned}
\]

#### A Brief Comparison of μP and Standard Parametrization

The simplified derivation can be summarized as follows: μP controls the scales of both the weights \(W_l\) and their updates \(\Delta W_l\), keeping activations and activation changes stable as the model becomes wider.

| Setting | μP | Standard parametrization | Main difference |
|---|---:|---:|---|
| Initialization standard deviation | \(\Theta\!\left(\dfrac{1}{\sqrt{n_{l-1}}}\min\!\left(1,\sqrt{\dfrac{n_l}{n_{l-1}}}\right)\right)\) | \(1/\sqrt{n_{l-1}}\) | μP further reduces the initialization standard deviation when \(n_l\lt n_{l-1}\) |
| SGD learning rate | \(\Theta(n_l/n_{l-1})\) | \(\Theta(1)\) | Both remain constant-order when input and output widths scale proportionally |
| Adam learning rate | \(\Theta(1/n_{l-1})\) | \(\Theta(1)\) | μP's effective Adam learning rate decreases as input width grows |

The clearest differences are therefore that μP explicitly scales the Adam learning rate and, when fan-out \(n_l\) is smaller than fan-in \(n_{l-1}\), uses a different initialization scale from standard parametrization. These remain simplified rules for linear layers; a real Transformer must treat parameter types separately.

#### Transformer Parameters Must Be Scaled Separately

μP is a method for scaling hyperparameters with model width. Parameters in a real Transformer have different shapes and functions, so the same initialization and learning-rate rule cannot be applied to every tensor.[17]

Let \(M\) denote model width, \(H\) the number of attention heads, \(D\) the width of one head, \(F\) the MLP hidden width, \(P\) the width of the proxy model used for hyperparameter search, and \(\alpha\) the base learning rate. The parameter groups are:

- \(W^E\): token embedding matrix;
- \(W^{AQ},W^{AK},W^{AV},W^{AO}\): attention Query, Key, Value, and output projections;
- \(W^{FI},W^{FO}\): MLP input and output projections;
- \(W^U\): output matrix mapping hidden states to vocabulary logits.

The initialization quantities in the table are <strong>variances</strong>, not standard deviations. The asymptotic columns describe order as width changes, whereas the exact columns give the numerical rules used in the experiments.

| Parameter | Initialization variance (asymptotic) | Adam learning rate (asymptotic) | Initialization variance (exact) | Adam learning rate (exact) |
|---|---:|---:|---:|---:|
| \(W^E\) | \(1\) | \(1\) | \(1\) | \(\alpha\) |
| \(W^{AQ}\) | \(1/M\) | \(1/M\) | \(1/M\) | \(\alpha P/M\) |
| \(W^{AK}\) | \(1/M\) | \(1/M\) | \(1/M\) | \(\alpha P/M\) |
| \(W^{AV}\) | \(1/M\) | \(1/M\) | \(1/M\) | \(\alpha P/M\) |
| \(W^{AO}\) | \(1/(HD)\) | \(1/(HD)\) | \(1/M\) | \(\alpha P/M\) |
| \(W^{FI}\) | \(1/M\) | \(1/M\) | \(1/M\) | \(\alpha P/M\) |
| \(W^{FO}\) | \(1/F\) | \(1/F\) | \(0.25/M\) | \(\alpha P/M\) |
| \(W^U\) | \(1/M^2\) | \(1/M\) | \(1/M^2\) | \(\alpha P/M\) |

These experiments fix \(HD=M\), \(F=4M\), \(P=128\), and \(D=128\). Thus, \(1/(HD)\) for \(W^{AO}\) can be written as \(1/M\), and \(1/F\) for \(W^{FO}\) as \(0.25/M\). When target-model width equals proxy-model width, \(M=P\), the exact Adam learning rates for every parameter group except embeddings return to the base learning rate \(\alpha\); the embedding matrix always uses \(\alpha\).

Attention scores also require a separate scaling factor. μP recommends

\[
\begin{aligned}
\tau^{-1}&=\Theta(1/D),
\end{aligned}
\]

rather than the standard Transformer factor

\[
\begin{aligned}
\tau^{-1}&=1/\sqrt D.
\end{aligned}
\]

The experiments use \(\tau^{-1}=1/D\). Because \(D\) is fixed while \(M\) grows, any nonzero constant independent of \(M\) satisfies width μP asymptotically. Nevertheless, the experiments show that the exact constant used for attention scaling still materially affects model performance and learning-rate transfer.

#### Applicability of μP to Modern Training Components

Modern language models also vary activation functions, batch sizes, initialization, normalization parameters, optimizers, and regularization. The complete ablation table below tests whether μP can still transfer the best base learning rate at width \(M=128\) to \(M=512\) and \(M=2048\) under these components. Bold values mark the lowest validation loss at each width, and `Transfer` indicates whether the learning rate transfers successfully.[17]

<figure>
  <img src="../../../posts/scaling-laws-practice/mup-transfer-ablation-table.png" alt="Complete ablation table for μP learning-rate transfer">
  <figcaption>Complete ablation study of μP learning-rate transfer. Learnable RMSNorm gain, standard attention scaling, weight decay, and Lion do not transfer consistently in this set of single-run experiments; most other settings do. Source: Lingle, 2025.</figcaption>
</figure>

The meaning of RMSNorm gain is discussed in [Normalization Methods](../norm/), while the update mechanisms of Lion and weight decay are covered in [Optimizers](../optimizers/). Note that Lion recovers successful transfer in later multi-seed experiments, so the failure in this table should not be interpreted as structural incompatibility between Lion and μP.

### 5.4 A Good-Looking Fit Can Still Fail under Extrapolation

Well-fitted IsoFLOP curves at small scale do not guarantee stable training after compute is scaled up. The following case uses Cautious AdamC and scales learning rate with the square root of batch size. The equal-compute parabolas are orderly inside the fitting range, but held-out extrapolation points increasingly depart from the prediction.[14]

<figure>
  <img src="../../../posts/scaling-laws-practice/optimizer-extrapolation-failure.png" alt="IsoFLOP fitting gradually fails and training eventually diverges at larger compute budgets">
  <figcaption>In held-out extrapolation experiments, the result at approximately \(10^{21}\) FLOPs is \(0.8\%\) worse than predicted, the gap reaches \(2.5\%\) near \(10^{22}\) FLOPs, and training diverges outright near \(10^{23}\) FLOPs. Source: William Held, Delphi.</figcaption>
</figure>

This figure cannot identify a unique cause of divergence by itself, but it demonstrates that a scaling-law study must reserve <strong>held-out validation points</strong>. In practice, training should be enlarged in stages while monitoring loss deviation and numerical stability. Once the deviation starts growing systematically, parametrization, learning-rate scaling, batch-size rules, and the optimizer must be re-examined instead of continuing to trust the original fit.

### 5.5 Summary

- Optimal hyperparameters such as learning rate and batch size change with model size and training steps rather than remaining fixed constants; Step Law fits these changes from small-scale experiments.
- μP controls the scales of initialization, activations, and parameter updates so that base hyperparameters found on a small model can transfer to wider models.
- Different optimizers transform gradients differently and require scaling rules matched to their update geometry; one set of conclusions cannot be applied indiscriminately to all optimizers.
- Changes in architecture, normalization, regularization, or optimizer can break hyperparameter transfer and must be validated again.
- A good small-scale curve fit does not guarantee stable large-scale training. Extrapolations should be tested with larger held-out experiments.

## References

[1] Shengding Hu et al. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. [Online]. Available: https://arxiv.org/abs/2404.06395

[2] Greg Yang et al. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. [Online]. Available: https://arxiv.org/abs/2203.03466

[3] Jordan Hoffmann et al. Training Compute-Optimal Large Language Models. [Online]. Available: https://arxiv.org/abs/2203.15556

[4] Jared Kaplan et al. Scaling Laws for Neural Language Models. [Online]. Available: https://arxiv.org/abs/2001.08361

[5] DeepSeek-AI. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism. [Online]. Available: https://arxiv.org/abs/2401.02954

[6] Qwen Team. Qwen2.5 Technical Report. [Online]. Available: https://arxiv.org/abs/2412.15115

[7] Qwen Team. Qwen3 Technical Report. [Online]. Available: https://arxiv.org/abs/2505.09388

[8] Kimi Team. Kimi K2: Open Agentic Intelligence. [Online]. Available: https://arxiv.org/abs/2507.20534

[9] Hunyuan Team. Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent. [Online]. Available: https://arxiv.org/abs/2411.02265

[10] Llama Team. The Llama 3 Herd of Models. [Online]. Available: https://arxiv.org/abs/2407.21783

[11] MiniMax. MiniMax-01: Scaling Foundation Models with Lightning Attention. [Online]. Available: https://arxiv.org/abs/2501.08313

[12] Houyi Li et al. Predictable Scale: Part I, Step Law — Optimal Hyperparameter Scaling Law in Large Language Model Pre-training. [Online]. Available: https://arxiv.org/abs/2503.04715

[13] Kaiyue Wen et al. Fantastic Pretraining Optimizers and Where to Find Them. [Online]. Available: https://arxiv.org/abs/2509.02046

[14] William Held. Delphi. [Online]. Available: https://oa.williamheld.com/blog/delphi/

[15] Keller Jordan et al. Muon: An optimizer for hidden layers in neural networks. [Online]. Available: https://kellerjordan.github.io/posts/muon/

[16] Nolan Dey et al. Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster. [Online]. Available: https://arxiv.org/abs/2304.03208

[17] Lucas D. Lingle. An Empirical Study of μP Learning Rate Transfer. [Online]. Available: https://arxiv.org/abs/2404.05728

[18] Greg Yang, James B. Simon, and Jeremy Bernstein. A Spectral Condition for Feature Learning. [Online]. Available: https://arxiv.org/abs/2310.17813
