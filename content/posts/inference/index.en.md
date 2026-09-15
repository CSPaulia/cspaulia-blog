---
title: "Large Language Model Inference: Performance Metrics and System Optimization"
date: 2026-08-21T11:30:03+08:00
series:
  main: "Large Language Model"
  subseries: "Systems and Hardware"
categories: ["Large Language Model", "Systems"]
tags: ["Inference", "KV Cache", "Quantization", "Speculative Decoding", "PagedAttention"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Study notes for CS336 Lecture 10 on performance analysis and system optimization for large language model inference."
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
    image: "inference-schema.png"
    alt: "A model and prompt pass through inference to produce a response"
    caption: "A model receives a prompt and produces a response through inference. Source: Stanford CS336 Lecture 10."
    relative: true
    hidden: false
    hiddenInList: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

<figure>
  <img src="../../../posts/inference/inference-schema.png" alt="A model and prompt pass through inference to produce a response">
  <figcaption>A trained model receives a prompt and produces a response through inference. Source: Stanford CS336 Lecture 10.</figcaption>
</figure>

## 1. Understanding Large Language Model Inference Workloads

### 1.1 Applications and Efficiency Metrics for LLM Inference

Large language model (LLM) inference is not limited to conversations with chatbots. Whenever a model generates new tokens from an existing input, it performs inference. Common settings include:<sup><a href="#references">[1]</a></sup>

- **Applications:** chatbots, code completion, agents, and batch data processing;
- **Model evaluation:** generating responses and measuring properties such as instruction following;
- **Reinforcement learning (RL):** sampling several responses to one problem, scoring them, and then updating the model.

#### Why Inference Efficiency Matters

Training is usually a one-time cost, whereas inference is repeated on every model call. A public estimate from 2025 placed OpenAI's daily traffic at roughly 8.6 trillion tokens,<sup><a href="#references">[2]</a></sup> while the DeepSeek-V4 technical report states that its pre-training used 32 trillion tokens.<sup><a href="#references">[3]</a></sup> The former is continuing inference traffic and the latter is a one-time training corpus, so they are not directly comparable; the contrast nevertheless illustrates why inference cost matters.

Generation volume also differs across applications:

- Chatbot output is read by people, so reading speed and patience limit response length;
- Agents often follow a query → internal reasoning trajectory → final output process, and may generate far more internal tokens than appear in the final response;
- Generating more tokens generally consumes more compute.

For teams operating products or inference platforms, even a small reduction in per-request cost accumulates across a large request volume.

#### Inference Providers and Open-Source Frameworks

Inference providers fall broadly into two groups: those serving closed models from organizations such as OpenAI, Anthropic, and Google, and those serving open-weight models, including Together, Fireworks, Baseten, DeepInfra, Groq, and Cerebras.

Common open-source inference frameworks include:

- [vLLM](https://github.com/vllm-project/vllm), developed at UC Berkeley, introduced PagedAttention and is a widely used general-purpose inference framework;
- [SGLang](https://sgl-project.github.io/), also originating at UC Berkeley, introduced RadixAttention and is well suited to agent workloads with substantial prefix reuse;
- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/overview.html), highly optimized by NVIDIA for graphics processing units (GPUs);
- [llama.cpp](https://github.com/ggml-org/llama.cpp), implemented in C/C++ and able to run models locally and on central processing units (CPUs).

#### Three Metrics for Inference Speed

“Faster inference” must be defined through concrete metrics:

- **Time to first token (TTFT):** the interval from submitting a request until the first token appears;
- **Latency:** the generation time for an individual request, often measured in seconds per token, which determines how smoothly a response appears;
- **Throughput:** the total number of tokens generated per second across concurrent requests.

Interactive applications usually prioritize TTFT and per-request latency, while offline batch processing prioritizes aggregate throughput. Improving one metric does not necessarily improve the other two; batch size and memory use create important trade-offs among them.

#### Differences in Parallelism between Training and Inference

| Dimension | Supervised training | Autoregressive inference |
| --- | --- | --- |
| Are the tokens known? | Every token in the training sequence is given | The next token has not yet been generated |
| Sequence parallelism | Positions can be processed in parallel | Tokens must be generated sequentially |
| Computation | Well suited to large matrix multiplications | The next step waits for the current token |
| Device utilization | Usually easier to keep high | More difficult to keep high |

This sequential dependency makes large compute devices harder to utilize during inference than during training. Efficiency analysis must therefore distinguish the stage that processes prompt tokens in parallel from the stage that generates response tokens one at a time.

### 1.2 Tensor-Dimension Notation in Transformer Computation

The following sections use einops-like notation for Transformer tensors. A letter denotes both a dimension and its length:

- \(B\): batch size;
- \(T\): sequence length;
- \(D\): model dimension;
- \(H\): attention-head dimension.

For the matrix multiplication \(\mathrm{BTD}\times\mathrm{DH}\rightarrow\mathrm{BTH}\), dimensions fall into three categories:

- A **contracting dimension** appears in both operands but not in the output. Here, \(D\) is contracted;
- An **ordinary dimension** appears in only one operand and remains in the output. Here, \(B\), \(T\), and \(H\) have this role;
- A **batching dimension** appears in both operands and the output. For example, in \(\mathrm{BD}\times\mathrm{BD}\rightarrow\mathrm{B}\), \(D\) is contracted and \(B\) remains as a batching dimension.

<figure>
  <img src="../../../posts/inference/transformer-diagram.png" alt="Tensor dimensions in Transformer attention and multilayer perceptron layers">
  <figcaption>Attention and multilayer perceptron layers in a Transformer. Red marks contracting dimensions, blue marks batching dimensions, and each operation is annotated with its input and output shapes. Source: How to Scale Your Model.<sup><a href="#references">[4]</a></sup></figcaption>
</figure>

#### Dimension Conventions in the Transformer

The later calculations use these common conventions:

- \(F=4D\): the multilayer perceptron (MLP) typically projects the model dimension to an intermediate dimension about four times larger;
- \(D=NH\): the model dimension is divided among \(N\) attention heads, each of dimension \(H\);
- \(N=KG\): in grouped-query attention (GQA), \(K\) is the number of key-value heads and each key-value head is shared by \(G\) query heads;
- \(S=T\): during training, \(S\) input positions predict \(T\) output positions of the same sequence length.

### 1.3 Arithmetic Intensity of Matrix Multiplication

Arithmetic intensity measures how many floating-point operations are performed per byte transferred. High arithmetic intensity gives the compute units more opportunity to reach their peak throughput; at low intensity, execution spends much of its time waiting for data movement.

Consider one matrix multiplication in the MLP:

\[
Y=XW,\qquad X\in\mathbb{R}^{B\times D},\quad W\in\mathbb{R}^{D\times F},\quad Y\in\mathbb{R}^{B\times F}.
\]

Here, \(B\) is the batch size, \(D\) the model dimension, and \(F\) the intermediate MLP dimension. Assume that all tensors use brain floating point (BF16), with two bytes per element, and temporarily ignore cache reuse and other overhead.

#### Compute and Memory Traffic of an MLP Matrix Multiplication

The operation has four steps:

| Step | Operation | FLOPs or memory traffic |
| --- | --- | --- |
| 1 | Read \(X\) from high-bandwidth memory (HBM) | \(2BD\) bytes |
| 2 | Read \(W\) from HBM | \(2DF\) bytes |
| 3 | Compute \(Y=XW\) | \(2BDF\) FLOPs |
| 4 | Write \(Y\) to HBM | \(2BF\) bytes |

The floating-point operations and total data transfer are therefore

\[
\mathrm{FLOPs}=2BDF,\qquad \mathrm{Bytes}=2BD+2DF+2BF.
\]

The arithmetic intensity \(I\) is

\[
\begin{aligned}
I&=\frac{\mathrm{FLOPs}}{\mathrm{Bytes}}\\
&=\frac{2BDF}{2BD+2DF+2BF}\\
&=\frac{BDF}{BD+DF+BF}.
\end{aligned}
\]

#### Batch Size Determines the Approximate Arithmetic Intensity

When the batch size is much smaller than the model and intermediate dimensions, \(B\ll D,F\), the \(DF\) term for reading the weight matrix dominates. Thus,

\[
I\approx B.
\]

Intuitively, all \(B\) examples reuse the same weights. A larger batch performs more computation for each weight read and therefore increases arithmetic intensity.

#### Compute-Bound and Memory-Bound Threshold on an H100

An H100 has a theoretical BF16 throughput of roughly \(989\times10^{12}\) FLOPs/s and memory bandwidth of roughly \(3.35\times10^{12}\) bytes/s.<sup><a href="#references">[7]</a></sup> Their ratio gives the accelerator's arithmetic-intensity threshold:

\[
I_{\text{H100}}=\frac{989\times 10^{12}}{3.35\times 10^{12}}\approx 295\ \text{FLOPs/Byte}.
\]

- An operation above roughly \(295\) FLOPs/byte is more likely to be <strong>compute-bound</strong>;
- An operation below this threshold is more likely to be <strong>memory-bound</strong>.

Combining the threshold with \(I\approx B\), this simplified theoretical model suggests that the MLP multiplication can become compute-bound only when \(B>295\).

At the extreme \(B=1\), the operation becomes a matrix-vector multiplication with arithmetic intensity close to \(1\). It reads a weight matrix containing \(DF\) elements to perform roughly \(2DF\) FLOPs and is therefore clearly constrained by memory bandwidth. Single-request, token-by-token generation often resembles this workload, which explains why LLM inference is so sensitive to memory bandwidth.

### 1.4 Arithmetic Intensity of LLM Inference

#### Naive Autoregressive Inference Recomputes the Entire Prefix

The most direct form of autoregressive inference feeds the complete history through the Transformer after every newly generated token and samples the next token from the distribution at the final position.

<figure>
  <img src="../../../posts/inference/naive-inference.webp" alt="Naive autoregressive inference repeatedly feeds all historical tokens into the Transformer">
  <figcaption>Naive autoregressive inference reruns the Transformer on the entire extended sequence after every token, repeatedly computing the same prefix. Source: How to Scale Your Model.<sup><a href="#references">[5]</a></sup></figcaption>
</figure>

For a sequence of length \(t\), full attention costs \(O(t^2)\). Generating \(T\) tokens therefore requires

\[
\sum_{t=1}^{T}O(t^2)=O(T^3).
\]

Adjacent generation steps share almost the same prefix, so much of this work can be reused. The standard solution stores the keys and values produced by historical tokens at every attention layer in HBM. This is the <strong>key-value cache (KV Cache)</strong>.

#### KV Cache Separates Prefill from Generation

<figure>
  <img src="../../../posts/inference/cached-inference.webp" alt="Prefill and token-by-token generation using a KV Cache">
  <figcaption>With a KV Cache, the prompt is encoded in parallel during prefill. Generation computes only the new token and appends its keys and values to the cache. Source: How to Scale Your Model.<sup><a href="#references">[5]</a></sup></figcaption>
</figure>

For every sequence in the batch, historical token, layer, and key-value head, the cache stores an \(H\)-dimensional key vector and value vector. Ignoring constants, its size grows as \(BSLKH\), where \(B\) is batch size, \(S\) history length, \(L\) number of layers, and \(K\) number of key-value heads.

Inference now has two stages:

1. **Prefill:** encode all prompt tokens at once, parallelizing across the sequence dimension as in training;
2. **Generation:** produce one token at a time and append its key and value to the KV Cache; the time dimension remains sequential.

The following calculations compare FLOPs, memory traffic, and arithmetic intensity for the MLP and attention layers. Let \(S\) be the existing context length and \(T\) the number of new tokens processed by the current forward pass. Prefill uses \(T=S\), while generation uses \(T=1\).

#### MLP Arithmetic Intensity Depends on \(BT\)

Considering only the MLP matrix multiplications and assuming \(BT\ll D,F\),

\[
I_{\mathrm{MLP}}\approx BT.
\]

- **Prefill:** \(T=S\), so \(I_{\mathrm{MLP}}\approx BS\). Larger batches and longer prompts both increase arithmetic intensity and make the MLP more likely to become compute-bound;
- **Generation:** \(T=1\), so \(I_{\mathrm{MLP}}\approx B\). Sufficient concurrent requests are required to raise arithmetic intensity, but concurrency changes dynamically in interactive services.

<details>
  <summary>Derivation of MLP FLOPs and memory traffic</summary>

  Let the input \(X\) have shape \(B\times T\times D\), the up-projection matrix \(W_{\mathrm{up}}\) and gate matrix \(W_{\mathrm{gate}}\) have shape \(D\times F\), and the down-projection matrix \(W_{\mathrm{down}}\) have shape \(F\times D\). Count only matrix multiplications and continue to assume BF16:

  | Step | Operation | FLOPs or memory traffic |
  | --- | --- | --- |
  | 1 | Read \(X\) | \(2BTD\) bytes |
  | 2 | Read \(W_{\mathrm{up}}\), \(W_{\mathrm{gate}}\), and \(W_{\mathrm{down}}\) | \(6DF\) bytes |
  | 3 | Compute \(U=XW_{\mathrm{up}}\) | \(2BTDF\) FLOPs |
  | 4 | Write \(U\) | \(2BTF\) bytes |
  | 5 | Compute \(G=XW_{\mathrm{gate}}\) | \(2BTDF\) FLOPs |
  | 6 | Write \(G\) | \(2BTF\) bytes |
  | 7 | Compute \(Y=[\operatorname{GeLU}(G)\odot U]W_{\mathrm{down}}\) | \(2BTDF\) FLOPs |
  | 8 | Write \(Y\) | \(2BTD\) bytes |

  GeLU is the Gaussian error linear unit and \(\odot\) denotes element-wise multiplication. Ignoring the cost of the activation and element-wise product,

  \[
  \begin{aligned}
  \mathrm{FLOPs}_{\mathrm{MLP}}&=6BTDF,\\
  \mathrm{Bytes}_{\mathrm{MLP}}&=4BTD+4BTF+6DF.
  \end{aligned}
  \]

  Therefore,

  \[
  I_{\mathrm{MLP}}=\frac{6BTDF}{4BTD+4BTF+6DF}.
  \]

  When \(BT\ll D,F\), the weight-read term \(6DF\) dominates the denominator, giving \(I_{\mathrm{MLP}}\approx BT\).

</details>

#### Generation Arithmetic Intensity of Attention Is Below 1

Count only the main matrix multiplications and required memory traffic after applying FlashAttention, without materializing the full attention matrix. Let \(S\) be the number of historical tokens in the KV Cache and \(T\) the number of new output positions. The arithmetic intensity of attention is

\[
I_{\mathrm{Attention}}=\frac{ST}{S+T}.
\]

For the two stages:

- **Prefill:** \(T=S\), so \(I_{\mathrm{Attention}}=S/2\). Longer contexts increase arithmetic intensity;
- **Generation:** \(T=1\), so \(I_{\mathrm{Attention}}=S/(S+1)<1\). Increasing batch size \(B\) does not change this result.

<details>
  <summary>Derivation of attention FLOPs and memory traffic</summary>

  Query \(Q\) has shape \(B\times T\times D\), while cached keys \(K\) and values \(V\) both have shape \(B\times S\times D\):

  | Step | Operation | FLOPs or memory traffic |
  | --- | --- | --- |
  | 1 | Read \(Q\), \(K\), and \(V\) | \(2BTD+4BSD\) bytes |
  | 2 | Compute \(A=QK^\top\) | \(2BSTD\) FLOPs |
  | 3 | Compute \(Y=\operatorname{softmax}(A)V\) | \(2BSTD\) FLOPs |
  | 4 | Write \(Y\) | \(2BTD\) bytes |

  The totals are

  \[
  \begin{aligned}
  \mathrm{FLOPs}_{\mathrm{Attention}}&=4BSTD,\\
  \mathrm{Bytes}_{\mathrm{Attention}}&=4BSD+4BTD.
  \end{aligned}
  \]

  Dividing cancels \(B\) and \(D\):

  \[
  I_{\mathrm{Attention}}=\frac{4BSTD}{4BSD+4BTD}=\frac{ST}{S+T}.
  \]

</details>

#### Prefill and Generation Have Different Bottlenecks

| Inference stage | MLP arithmetic intensity | Attention arithmetic intensity | Main characteristic |
| --- | --- | --- | --- |
| Prefill | \(BS\) | \(S/2\) | Parallel across the sequence and more likely to be compute-bound |
| Generation | \(B\) | \(S/(S+1)<1\) | Token-by-token execution and usually memory-bandwidth-bound |

The central conclusion is that <strong>prefill is usually closer to compute-bound, while token generation is usually closer to memory-bound</strong>. This distinction explains why the stages require different batching strategies and motivates later optimizations for latency, throughput, and KV Cache size.

### 1.5 How Batch Size Affects Inference Latency and Throughput

During memory-bound token generation, increasing batch size \(B\) creates a direct trade-off:

- **Higher throughput:** a larger batch amortizes the cost of reading model parameters across more sequences;
- **Higher generation latency:** a larger batch also expands the KV Cache and increases the amount of data read for each token.

The following simplified memory model estimates this trade-off. Its results are theoretical upper bounds under ideal conditions rather than measurements from a real system.

#### Memory Model for Transformer Parameters and the KV Cache

Let \(V\) be vocabulary size, \(D\) hidden dimension, \(F\) MLP intermediate dimension, \(N\) number of attention heads, \(K\) number of key-value heads, \(H\) head dimension, and \(L\) number of layers. A simplified Transformer parameter count is

\[
P=2VD+3LDF+L(2DNH+2DKH).
\]

The term \(2VD\) represents input and output embeddings, \(3LDF\) represents the three MLP projection matrices in every layer, and the final term represents the query, key, value, and output projections in attention.

Assume that both parameters and the KV Cache use BF16, with two bytes per element. Model parameters occupy

\[
M_{\mathrm{param}}=2P.
\]

For one sequence of context length \(S\), every layer stores both keys and values, so its KV Cache occupies

\[
M_{\mathrm{KV,seq}}=4SKHL.
\]

The factor of four comes from two caches—keys and values—and two bytes per BF16 element. With batch size \(B\), the amount of data read during one generation step is approximately

\[
M(B)=2P+4BSKHL.
\]

If memory bandwidth is \(\beta_{\mathrm{mem}}\) and generation is entirely memory-bandwidth-bound, the theoretical latency and throughput per generated token are

\[
\begin{aligned}
\operatorname{Latency}(B)&=\frac{M(B)}{\beta_{\mathrm{mem}}},\\
\operatorname{Throughput}(B)&=\frac{B}{\operatorname{Latency}(B)}
=\frac{B\beta_{\mathrm{mem}}}{M(B)}.
\end{aligned}
\]

#### Theoretical Results for Llama 2 13B on an H100

For Llama 2 13B with context length \(S=1024\), use the following model and hardware parameters:<sup><a href="#references">[6]</a></sup><sup><a href="#references">[7]</a></sup>

| Parameter | Value |
| --- | ---: |
| Hidden dimension \(D\) | 5120 |
| MLP intermediate dimension \(F\) | 13824 |
| Attention heads \(N\) | 40 |
| Key-value heads \(K\) | 40 |
| Head dimension \(H\) | 128 |
| Transformer layers \(L\) | 40 |
| Vocabulary size \(V\) | 32000 |
| H100 memory bandwidth \(\beta_{\mathrm{mem}}\) | 3.35 TB/s |

Substitution gives approximately 13.015 billion parameters. The BF16 parameters occupy about 26.03 GB, and the KV Cache for each sequence occupies about 0.84 GB.

| Batch size \(B\) | Total memory | Theoretical latency | Theoretical throughput | One 80 GB H100 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 26.87 GB | 8.02 ms/token | 124.7 tokens/s | Fits |
| 64 | 79.72 GB | 23.80 ms/token | 2689.5 tokens/s | Near the limit |
| 256 | 240.78 GB | 71.87 ms/token | 3561.8 tokens/s | Does not fit |

These results rely on idealized assumptions: computation and memory traffic overlap perfectly; kernel launch, scheduling, communication, and other system overheads are ignored; and every step reads all parameters and the corresponding KV Cache. The numbers should therefore be understood as theoretical estimates.

#### Batch Size Creates a Latency–Throughput Trade-off

As \(B\) increases, KV Cache use grows as \(O(B)\). Each generation step must move more data, so latency increases. At the same time, one parameter read serves more sequences, raising throughput.

The throughput gains eventually diminish. Increasing batch size from 1 to 64 improves throughput by about 21.6 times. Increasing it again from 64 to 256 multiplies batch size by four but improves throughput by only about 32%, because the KV Cache has become the dominant memory cost.

The trade-off is therefore clear:

- Smaller batches reduce per-request latency;
- Larger batches increase aggregate throughput but consume more memory and increase latency.

#### Inference Parallelism and Two-Stage Batching

When one GPU can hold the complete model, two forms of parallel execution are possible:

- **Model replication:** deploy \(M\) independent replicas on \(M\) GPUs. Ideally, per-request latency stays nearly unchanged while total throughput increases by a factor of \(M\);
- **Model or KV Cache sharding:** distribute parameters or cache entries across GPUs. This requires additional communication and makes implementation and performance analysis more complex.

Prefill and generation also have different batching objectives:

- **Prefill:** TTFT is dominated by prefill time, so small batches are usually preferable for handling new requests quickly;
- **Generation:** larger batches can combine several active requests and improve aggregate throughput.

## 2. Taking Shortcuts (Lossy)

Token generation is usually memory-bandwidth-bound. A direct optimization is therefore to <strong>reduce the data read and stored during inference</strong>.

These methods change model structure, numerical precision, or parameters and may reduce accuracy. The objective is to lower inference complexity while preserving as much capability as possible.

### 2.1 Reducing the KV Cache

KV Cache size grows with sequence length, layer count, and the number of key-value heads. Reducing the number or dimension of cached vectors—or the amount of retained history—lowers generation-stage memory traffic.

#### Grouped-Query Attention Shares Keys and Values across Query Heads

In multi-head attention (MHA), every query head has its own key and value heads. Grouped-query attention (GQA) retains \(N\) query heads but uses only \(K\) key and value heads, each shared by \(N/K\) query heads.<sup><a href="#references">[8]</a></sup>

- **MHA:** \(K=N\), with independent key and value heads;
- **Multi-query attention (MQA):** \(K=1\), with all query heads sharing one set;
- **GQA:** \(1<K<N\), balancing MHA expressiveness and MQA cache efficiency.

<figure>
  <img src="../../../posts/inference/gqa-architecture.png" alt="Key-value head sharing in MHA, GQA, and MQA">
  <figcaption>MHA, GQA, and MQA. In GQA, a group of query heads shares one key head and one value head. Source: GQA.</figcaption>
</figure>

Relative to MHA, GQA reduces the KV Cache to \(K/N\) of its original size. Less cache traffic lowers theoretical latency, and freed memory can support larger batches.

<figure>
  <img src="../../../posts/inference/gqa-speed.png" alt="Single-example inference time for MHA, GQA, and MQA">
  <figcaption>Single-example inference time in the GQA experiments. More key-value groups move GQA toward MHA latency; MQA has the smallest cache and lowest latency. Source: GQA.</figcaption>
</figure>

Using the earlier simplified Llama 2 13B configuration with \(N=40\):

| Attention | \(K\) | Batch \(B\) | Total memory | Theoretical latency | Theoretical throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| MHA | 40 | 64 | 79.72 GB | 23.80 ms/token | 2689.5 tokens/s |
| GQA | 8 | 64 | 33.41 GB | 9.97 ms/token | 6416.7 tokens/s |
| GQA | 8 | 256 | 65.63 GB | 19.59 ms/token | 13068.2 tokens/s |

At equal batch size, GQA lowers theoretical latency and raises throughput. Its smaller cache also permits a batch of 256 within 80 GB. Accuracy must still be verified: GQA-8-XXL approached MHA-XXL's average task score in the paper while running substantially faster, but results depend on architecture, training, and evaluation.<sup><a href="#references">[8]</a></sup>

<figure>
  <img src="../../../posts/inference/gqa-accuracy.png" alt="Inference time and task accuracy of MHA, MQA, and GQA">
  <figcaption>GQA achieved an average score close to MHA while retaining inference efficiency close to MQA in this experiment. Source: GQA.</figcaption>
</figure>

#### Multi-Head Latent Attention Compresses Key-Value Representations

Multi-head latent attention (MLA) compresses hidden state \(h\) into a low-dimensional latent vector rather than caching complete key and value vectors:<sup><a href="#references">[9]</a></sup>

\[
c=W_ch,
\qquad
\mathbf{k}=W_Kc,
\qquad
\mathbf{v}=W_Vc.
\]

Ordinary attention caches key and value dimensions of \(NH\), while MLA mainly caches a compressed vector of dimension \(C\). The saving is substantial when \(C\ll NH\).

<figure>
  <img src="../../../posts/inference/mla-schema.png" alt="KV Cache structures of MHA, GQA, MQA, and MLA">
  <figcaption>Shaded regions are stored during inference. MLA caches compressed latent KV representations and projects them when needed. Source: DeepSeek-V2.</figcaption>
</figure>

DeepSeek-V2 compresses an \(NH=16384\)-dimensional representation to \(C=512\). Rotary position embedding (RoPE) cannot be fully absorbed into this path, so an additional 64-dimensional RoPE key is stored, for \(576\) cached dimensions per token.<sup><a href="#references">[9]</a></sup>

Its ablations provide two model-specific observations: MHA outperformed GQA and MQA overall on harder benchmarks, while MLA slightly exceeded MHA overall with a much smaller cache.<sup><a href="#references">[9]</a></sup>

<details>
  <summary>DeepSeek-V2 attention ablations</summary>
  <figure>
    <img src="../../../posts/inference/mla-accuracy.png" alt="Accuracy comparison of MHA, GQA, and MQA in DeepSeek-V2">
    <figcaption>MHA performed better overall than GQA and MQA on these harder benchmarks. Source: DeepSeek-V2.</figcaption>
  </figure>
  <figure>
    <img src="../../../posts/inference/mla-accuracy2.png" alt="KV Cache and accuracy comparison of MLA and MHA in DeepSeek-V2">
    <figcaption>MLA greatly reduced per-token KV Cache size and scored higher on most listed tasks. Source: DeepSeek-V2.</figcaption>
  </figure>
</details>

These are empirical results for a particular model and training setup, not a universal guarantee that MLA outperforms MHA.

#### Cross-Layer Attention Shares Keys and Values across Layers

Cross-layer attention (CLA) extends sharing across Transformer layers. GQA shares keys and values among query heads; CLA lets adjacent layers reuse the same keys and values instead of caching a separate set per layer.<sup><a href="#references">[10]</a></sup>

<figure>
  <img src="../../../posts/inference/cla-diagram.png" alt="Key-value projections in a conventional Transformer and cross-layer attention">
  <figcaption>A conventional Transformer computes and caches keys and values in every layer; CLA lets upper layers reuse those from lower layers. Source: Reducing Transformer Key-Value Cache Size with Cross-Layer Attention.</figcaption>
</figure>

Experiments on 1B models showed that CLA could use a smaller cache at similar validation perplexity, improving the accuracy–cache Pareto frontier.<sup><a href="#references">[10]</a></sup>

<figure>
  <img src="../../../posts/inference/cla-results.png" alt="Pareto frontier of validation perplexity and KV Cache size with and without CLA">
  <figcaption>Validation perplexity versus per-token cache size. Red points use CLA and show a better trade-off. Source: Reducing Transformer Key-Value Cache Size with Cross-Layer Attention.</figcaption>
</figure>

#### Local Attention Truncates the Retained History

Local or sliding-window attention reads only a fixed recent window rather than the full history. The cache per layer no longer grows with complete sequence length.<sup><a href="#references">[11]</a></sup><sup><a href="#references">[12]</a></sup><sup><a href="#references">[13]</a></sup>

<figure>
  <img src="../../../posts/inference/longformer-attention.png" alt="Global, sliding-window, dilated-window, and combined global-window attention patterns">
  <figcaption>Global attention and three sparse patterns. Sliding windows restrict local context, dilation expands the receptive field, and global tokens restore long-distance communication. Source: Longformer.</figcaption>
</figure>

- **The effective context expands with depth:** information can propagate across windows layer by layer;
- **Long-range modeling may degrade:** a layer cannot directly access information that has left its window.

Hybrid designs alternate local and global layers: local layers control cache and compute, while global layers periodically restore long-range interaction.

#### DeepSeek-V4 Combines Compression and Sparse Selection

DeepSeek-V4 supports contexts up to one million tokens using three mechanisms:<sup><a href="#references">[3]</a></sup>

- **Compressed sparse attention (CSA):** compress every \(m\) historical tokens into one representation;
- **DeepSeek sparse attention (DSA):** score historical positions with a lightweight indexer and retain the top \(k\) compressed KV entries;
- **Heavily compressed attention (HCA):** increase compression further for very long contexts.

<figure>
  <img src="../../../posts/inference/deepseek-v4-attention.png" alt="DeepSeek-V4 attends to sliding-window KV and selected compressed KV">
  <figcaption>DeepSeek-V4 concatenates recent sliding-window KV with compressed KV selected by an indexer, then applies shared-key-value multi-query attention. Source: DeepSeek-V4.</figcaption>
</figure>

#### Summary of KV Cache Reduction

- **Reduce cached dimensions:** GQA shares KV across query heads, MLA caches latent representations, and CLA shares KV across layers;
- **Truncate or sparsify history:** local attention keeps a fixed window, while sparse attention selects a small number of relevant positions;
- **Use other sequence models:** linear attention, state-space models such as Mamba 2 and GatedDeltaNet, and diffusion models also seek to avoid full-attention cache costs.

All impose stronger structural assumptions in exchange for lower inference cost, so their accuracy must be verified through training and evaluation.

### 2.2 Quantization: Reducing Memory Traffic with Lower Precision

Quantization represents values with fewer bits. It reduces parameter and activation memory and can improve memory-bound generation, but finite low-precision ranges introduce rounding error.

#### Quantization and Dequantization

For affine quantization with floating-point value \(x\), scale \(s\), and zero point \(z\):

\[
\begin{aligned}
q&=\operatorname{round}\left(\frac{x}{s}\right)+z,\\
\hat{x}&=(q-z)s.
\end{aligned}
\]

The stored integer is \(q\), and \(\hat{x}\) is its approximate reconstruction. Implementations also clamp \(q\) to the target integer range. For \(x=5.2342\), \(s=0.1\), and \(z=4\):

\[
q=56,
\qquad
\hat{x}=5.2.
\]

The reconstruction error is \(0.0342\). A smaller scale gives finer spacing but usually covers a narrower real-valued range.

#### Common Numerical Formats

<figure>
  <img src="../../../posts/inference/number-formats.png" alt="Bit layouts of FP32, FP16, FP8, and INT8">
  <figcaption>Floating-point formats allocate bits to sign, exponent, and mantissa; integer formats do not. Source: <a href="https://www.baseten.co/blog/fp8-efficient-model-inference-with-8-bit-floating-point-numbers/">Baseten</a>.</figcaption>
</figure>

| Format | Storage | Main characteristic |
| --- | ---: | --- |
| FP32 | 4 bytes | High precision and range; often used for master weights, gradient accumulation, or optimizer states |
| BF16 | 2 bytes | Same exponent width as FP32; common in training and inference |
| FP8 | 1 byte | E4M3 reaches \(\pm448\) on H100 and can support mixed-precision training |
| INT8 | 1 byte | Usually ranges from \(-128\) to \(127\); common in inference |
| INT4 | 0.5 bytes | Usually ranges from \(-8\) to \(7\); smaller but often less accurate |

Format choice also depends on hardware support and efficient matrix-multiplication and dequantization kernels.<sup><a href="#references">[14]</a></sup><sup><a href="#references">[15]</a></sup>

#### Quantization-Aware Training and Post-Training Quantization

| Dimension | QAT | PTQ |
| --- | --- | --- |
| Stage | During training | After training |
| Mechanism | Simulate quantization error in the forward pass | Calibrate scales and zero points using a small dataset |
| Advantage | Weights adapt to quantization error | Low cost; no full retraining |
| Limitation | Requires expensive retraining | More sensitive at low bit widths |

Quantization-aware training (QAT) usually preserves accuracy better, while post-training quantization (PTQ) is more convenient for an existing model. Both depend on granularity, calibration data, and target hardware.<sup><a href="#references">[15]</a></sup>

> **GPTQ:** a PTQ method for generative Transformers that uses approximate second-order information and compensates unquantized weights as quantization proceeds.<sup><a href="#references">[16]</a></sup>

#### Activation-Aware Weight Quantization Protects Salient Weights

Activation-aware weight quantization (AWQ) observes that weights multiplied by unusually large activation channels matter more to the output and should receive lower quantization error.<sup><a href="#references">[17]</a></sup>

1. Measure activation distributions on a small calibration set;
2. Identify roughly \(0.1\%\)–\(1\%\) salient weight channels;
3. Scale salient channels before uniformly quantizing the weights.

Keeping a few weights in FP16 creates inefficient mixed-precision computation. AWQ instead uses equivalent scaling so all weights retain one low-precision format.

<figure>
  <img src="../../../posts/inference/awq-schema.png" alt="AWQ identifies salient channels from activations and scales them before quantization">
  <figcaption>Direct INT3 rounding causes large error; retaining FP16 outliers hurts hardware efficiency; AWQ scales salient channels before uniform low-precision computation. Source: AWQ.</figcaption>
</figure>

Reported experiments reduced model memory by about four times and accelerated inference by about 3.2 times when moving from FP16 to INT3. These gains depend on model, hardware, granularity, and kernel implementation.<sup><a href="#references">[17]</a></sup>

### 2.3 Model Pruning: Removing Structure and Recovering Capability through Distillation

Structured pruning removes complete layers, attention heads, or hidden dimensions, producing regular tensor shapes that can yield real hardware speedups. NVIDIA's pruning and knowledge-distillation pipeline is:<sup><a href="#references">[18]</a></sup>

1. Estimate the importance of layers, heads, and hidden dimensions on a 1024-example calibration set;
2. Remove low-importance structures to form a smaller student;
3. Use the original model as a teacher to recover capability through distillation.

<figure>
  <img src="../../../posts/inference/pruning-kd-loop.png" alt="Iterative LLM importance estimation, ranking, pruning, and knowledge distillation">
  <figcaption>Estimate and rank embedding dimensions, attention heads, and MLP channels; remove unimportant structures; then recover capability through distillation. Source: Compact Language Models via Pruning and Knowledge Distillation.</figcaption>
</figure>

#### Pruning Reduces the Cost of Training Smaller Models

Starting from Nemotron-4 15B, the method produced 8B and 4B Minitron models through pruning and distillation. The paper reports using as little as about \(1/40\) of the tokens required for training same-size models from scratch, with improved MMLU in some comparisons.<sup><a href="#references">[18]</a></sup>

<figure>
  <img src="../../../posts/inference/pruning-kd.png" alt="Training cost and MMLU scores of Minitron models">
  <figcaption>Training cost and MMLU of Minitron 4B and 8B. The green dashed path starts from Nemotron-4 15B. Source: Compact Language Models via Pruning and Knowledge Distillation.</figcaption>
</figure>

Recovery still depends on importance estimation, pruning ratio, distillation data, and training budget.

### 2.4 Summary: Two Routes to a Faster Model

Lossy shortcuts aim to reduce inference complexity while retaining accuracy.<sup><a href="#references">[1]</a></sup>

#### Training an Efficient Architecture from Scratch

1. Design a faster architecture;
2. Train it from scratch.

This is direct but pays the full training cost.

#### Reusing Capability through Distillation

1. Design a faster architecture;
2. Initialize it with reusable weights from the original model, even if the architectures differ;
3. Use the original as a teacher and distill its capability into the efficient model.

This route reuses prior training investment, but success depends on architectural compatibility and whether distillation can recover capability lost through compression.<sup><a href="#references">[18]</a></sup>

## 3. Use Shortcuts but Double Check: Lossless Methods

The methods in Chapter 2 trade changes in architecture or numerical precision for speed and may reduce accuracy. Another approach is to guess outputs using a low-cost method and then verify them with the original model. With a mathematically correct verification procedure, generation can be accelerated while preserving the original model's output distribution.

Here, “lossless” means that the <strong>final sampling distribution is identical to that of the target model</strong>, not that every randomized run produces exactly the same token sequence.

### 3.1 Speculative Sampling: A Draft Model Proposes and the Target Model Verifies

Prefill can process a span of tokens in parallel, whereas ordinary decoding proceeds one token at a time. In other words, it is often more efficient for the target model to check several candidate tokens in one parallel pass than to generate them sequentially. Speculative sampling exploits precisely this asymmetry: <strong>parallel verification is more efficient than sequential generation</strong>.<sup><a href="#references">[19]</a></sup><sup><a href="#references">[20]</a></sup>

Speculative sampling uses two models:

- **Draft model \(p\)**: a smaller, cheaper model that first generates \(K\) candidate tokens autoregressively—for example, four at a time;
- **Target model \(q\)**: the original large model, which evaluates all candidate positions in one forward pass and applies an acceptance rule to determine how many tokens to retain.

One round of speculative sampling can be summarized as follows:

1. The draft model generates \(K\) candidate tokens in sequence.
2. The target model computes the probabilities at all candidate positions in parallel.
3. Candidates are accepted in order until the first rejection.
4. After a rejection, one token is sampled from a corrected residual distribution. If every candidate is accepted, one additional token is sampled from the target model.

For a token \(x\) proposed by the draft model, the acceptance probability is

\[
a(x)=\min\left(1,\frac{q(x)}{p(x)}\right).
\]

When a candidate is rejected, sampling directly from \(q\) again would be incorrect. The token must instead be sampled from the normalized residual distribution

\[
r(x)=\frac{\max(q(x)-p(x),0)}
{\sum_y\max(q(y)-p(y),0)}.
\]

This corrected rejection-sampling rule guarantees that every target-model call produces at least one token and, in theory, makes the final token an exact sample from the target distribution \(q\). Practical implementations remain subject to finite numerical precision.<sup><a href="#references">[20]</a></sup>

<details>
  <summary>Expand: Complete speculative-sampling algorithm</summary>

  <figure>
    <img src="../../../posts/inference/speculative-sampling-algorithm.png" alt="Complete speculative-sampling algorithm in which a draft model proposes candidates and a target model verifies them in parallel">
    <figcaption>Speculative-sampling algorithm. The draft model \(p\) proposes \(K\) consecutive tokens, the target model \(q\) computes probabilities for the candidate prefixes in parallel, and sampling is corrected through the acceptance probability and residual distribution. Source: Accelerating Large Language Model Decoding with Speculative Sampling.</figcaption>
  </figure>

</details>

> **Supplementary material**
>
> Google's [speculative decoding animation](https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4) gives an intuitive view of the “draft—parallel verification—accept or reject” process.

<details>
  <summary>Expand: Verifying distribution preservation with a binary vocabulary</summary>

  Suppose the vocabulary contains only \(A\) and \(B\), and the target and draft distributions are \([q(A),q(B)]\) and \([p(A),p(B)]\), respectively. Further suppose the draft model oversamples \(A\):

  \[
  p(A)>q(A),\qquad p(B) < q(B).
  \]

  The residual distribution then samples only \(B\), so \(r(A)=0\) and \(r(B)=1\). The final probability of sampling \(A\) is

  \[
  \begin{aligned}
  \Pr(A)
  &=p(A)\frac{q(A)}{p(A)}+p(B)\times 0\\
  &=q(A).
  \end{aligned}
  \]

  The final probability of sampling \(B\) is

  \[
  \begin{aligned}
  \Pr(B)
  &=p(B)+p(A)\left(1-\frac{q(A)}{p(A)}\right)\\
  &=p(A)+p(B)-q(A)\\
  &=1-q(A)\\
  &=q(B).
  \end{aligned}
  \]

  Therefore, whether a token comes from the draft model or the residual distribution, its final probability is exactly the target-model probability. This binary example captures why speculative sampling preserves the target distribution.

</details>

#### Speedup Depends on Candidate Acceptance Rate

In experiments with Chinchilla 70B, speculative sampling accelerates decoding by roughly \(2\)–\(2.5\times\) on XSum and HumanEval while producing essentially the same task results as ordinary autoregressive sampling.<sup><a href="#references">[20]</a></sup>

However, a larger number of draft tokens \(K\) is not always better:

1. Increasing \(K\) lets one target-model call attempt to confirm more tokens.
2. Later candidates depend on all earlier candidates being correct, so the joint acceptance rate gradually declines.
3. Draft generation and target verification both have costs, so an excessively large \(K\) can increase total latency again.

<details>
  <summary>Expand: Experimental results and the draft-length tradeoff</summary>

  <figure>
    <img src="../../../posts/inference/speculative-sampling-results.png" alt="Speed and task results of ordinary autoregressive sampling and speculative sampling on XSum and HumanEval">
    <figcaption>Comparison of ordinary autoregressive sampling and speculative sampling. Speculative sampling roughly halves average time per token while preserving similar task performance. Source: Accelerating Large Language Model Decoding with Speculative Sampling.</figcaption>
  </figure>

  <figure>
    <img src="../../../posts/inference/speculative-sampling-stats.png" alt="Effects of draft-token count on sampling time, acceptance rate, and per-round latency">
    <figcaption>Effect of draft-token count \(K\). Increasing \(K\) reduces the number of target-model calls, but also lowers candidate acceptance and raises verification time per round. The best \(K\) therefore depends on the task. Source: Accelerating Large Language Model Decoding with Speculative Sampling.</figcaption>
  </figure>

</details>

#### The Draft Model Must Be Cheap and Close to the Target Model

In practice, a much smaller model is commonly used as the draft model—for example, an approximately 8B model for a 70B target, or an approximately 1B model for an 8B target. A smaller model proposes candidates more cheaply, but if its distribution differs too much from the target model, acceptance falls.

The draft model must therefore balance two objectives:

1. **Low generation cost**: otherwise the draft stage consumes the intended speedup.
2. **A distribution close to the target model**: this raises candidate acceptance and can be encouraged through knowledge distillation.

#### Medusa and EAGLE Improve Candidate Generation

Most opportunities to improve speculative sampling lie in generating candidate tokens more quickly and accurately:

- **Medusa** adds multiple decoding heads to the target model, predicts several future tokens in parallel, and verifies multiple candidate paths at once with tree attention, eliminating the need to maintain a complete independent draft model.<sup><a href="#references">[21]</a></sup>
- **EAGLE** autoregressively predicts high-level features from the target model's penultimate layer and combines them with a one-token-shifted sequence to reduce uncertainty in feature prediction.<sup><a href="#references">[22]</a></sup>

<figure>
  <img src="../../../posts/inference/medusa-eagle.png" alt="Candidate generation in ordinary speculative sampling, Lookahead, Medusa, and EAGLE">
  <figcaption>Comparison of candidate-generation methods. Medusa uses multiple decoding heads to predict future tokens in parallel; EAGLE uses both tokens and high-level features from the target model to generate candidates. Source: Stanford CS336 Lecture 10.</figcaption>
</figure>

#### Speculative Sampling Summary

- Mathematical correction keeps the final result an exact sample from the target-model distribution, within the limits of numerical precision.
- Draft generation and target verification exploit the asymmetry that parallel checking is more efficient than sequential generation.
- Actual speedup depends on draft cost, candidate acceptance, and the number of proposed tokens; there remains substantial room to innovate in draft-model training and architecture.

## 4. Handling Dynamic Workloads

The shapes of offline training data are usually known in advance, whereas online inference traffic changes continuously. Real-time requests are difficult to assemble directly into stable, regular batches for three main reasons:

1. **Requests arrive at different times**: waiting to fill a batch increases queueing latency for requests that arrived earlier.
2. **Requests may share prefixes**: a common system prompt or multiple samples from the same prompt create duplicate prefixes. Processing them independently repeats both computation and KV Cache storage.
3. **Sequence lengths differ**: prompt and generation lengths are both variable, and padding every request to a common length wastes computation and GPU memory.

A dynamic inference system must therefore do more than raise GPU utilization. It must admit new requests promptly, remove completed requests, and process sequences of different lengths efficiently.

### 4.1 Continuous Batching: Dynamically Updating the Batch at Each Decoding Iteration

#### Online Inference Requests Form Irregular Batches

Training and online inference have different data shapes:

- **Training**: a batch can usually be represented as a regular \(B\times S\times D\) tensor, with every sequence in the batch sharing length \(S\).
- **Online inference**: requests arrive at different times and finish after generating different numbers of tokens, so active sequences resemble a ragged array.

In static batching, once a batch starts running, it normally cannot be updated until every request has completed. Positions belonging to short requests become idle after those requests finish, while newly arrived requests must wait for the current batch to end.

<figure>
  <img src="../../../posts/inference/static-batching.png" alt="Sequences of different lengths in static batching and idle positions left after early completion">
  <figcaption>Variation in sequence length under static batching. Yellow denotes prompt tokens, blue denotes generated tokens, and red denotes sequence termination. After a short request finishes, its position cannot immediately be reused by a new request. Source: Stanford CS336 Lecture 10.</figcaption>
</figure>

#### Iteration-Level Scheduling: Updating the Batch after Every Decoding Step

Continuous batching reduces scheduling granularity from an entire request to a single decoding iteration. Orca calls this mechanism iteration-level scheduling:<sup><a href="#references">[23]</a></sup>

1. The scheduler selects a group of requests to execute.
2. The model performs one decoding iteration for that group, generating one token for each request.
3. The scheduler identifies completed requests and removes them immediately.
4. Before the next iteration, waiting requests are inserted into the newly available positions.

As a result, a new request waits only for the current decoding iteration rather than for the longest request in the whole batch. A request that finishes early can also return its result immediately.

#### Selective Batching: Separate Attention, Combine Other Operators

Iteration-level scheduling creates batches whose requests have different context lengths. Ordinary batching expects matching tensor shapes, so these sequences cannot simply be stacked into one regular \(B\times S\times D\) tensor.

Orca uses selective batching to treat two classes of operations differently:<sup><a href="#references">[23]</a></sup>

1. **Attention operations**: each sequence reads a different length of KV Cache, so attention is computed separately according to each sequence's length.
2. **Non-attention operations**: MLPs, linear projections, and normalization do not require cross-token interaction, so tokens from all sequences can be concatenated along the sequence dimension and processed together.

For example, suppose three sequence tensors have shapes \([3,D]\), \([9,D]\), and \([5,D]\). Attention handles the three sequences separately, while non-attention operations can pack them as

\[
\begin{aligned}
X_{\mathrm{packed}}
&=\operatorname{concat}(X_1,X_2,X_3)\\
&\in\mathbb{R}^{(3+9+5)\times D}\\
&=\mathbb{R}^{17\times D}.
\end{aligned}
\]

This avoids padding every sequence to length 9 while still creating larger matrix operations for computations that share model parameters. The core principle is to <strong>batch only operators whose shapes are compatible and that genuinely benefit from batching</strong>.

### 4.2 PagedAttention: Managing the KV Cache with Paging

Continuous batching constantly adds and removes requests, while each sequence's KV Cache grows dynamically during generation. PagedAttention, introduced by vLLM, borrows virtual-memory paging from operating systems. It maps a logically contiguous KV Cache onto physically noncontiguous GPU-memory blocks, enabling on-demand allocation and prefix sharing between sequences.<sup><a href="#references">[24]</a></sup>

#### Contiguous Preallocation Wastes Memory through Internal and External Fragmentation

Traditional systems reserve one contiguous KV Cache region for an entire sequence when a request arrives, based on prompt length and maximum generation length. This causes two kinds of memory waste:

1. **Internal fragmentation**: generation often stops before the maximum length, leaving reserved but unused space unavailable to other requests.
2. **External fragmentation**: scattered gaps remain between request caches. Even when enough total GPU memory is free, there may be no single contiguous region large enough for a new allocation.

<figure>
  <img src="../../../posts/inference/paged-attention-fragmentation.png" alt="Internal and external fragmentation caused by contiguous KV Cache allocation">
  <figcaption>GPU-memory fragmentation caused by contiguous preallocation. Slots reserved for the maximum generation length but never used form internal fragmentation; gaps between requests that cannot be combined into a large contiguous region form external fragmentation. Source: vLLM.</figcaption>
</figure>

#### Paged Mapping: Logically Contiguous, Physically Noncontiguous

PagedAttention divides a sequence's KV Cache into fixed-size blocks. The sequence remains logically ordered by token, but each logical block can map to any free physical block. The system allocates another block only after the existing one is full.

<figure>
  <img src="../../../posts/inference/paged-attention-blocks.png" alt="Attention reads physically noncontiguous KV Cache through a block table">
  <figcaption>Attention computation under PagedAttention. A query reads historical keys and values scattered across physical blocks through a block mapping; physical storage need not be contiguous. Source: vLLM.</figcaption>
</figure>

<figure>
  <img src="../../../posts/inference/paged-attention-logical.png" alt="Mapping logical KV blocks of two requests to noncontiguous physical KV blocks">
  <figcaption>Mapping between logical and physical blocks. Each request maintains its own logical ordering, while a block table maps those logical blocks to any available physical KV blocks. Source: vLLM.</figcaption>
</figure>

This design has two immediate consequences:

1. **On-demand growth**: the cache is no longer reserved for the maximum generation length in advance. Internal fragmentation can occur only in the final partially filled block.
2. **Flexible allocation**: a new block can use any free location, so the full KV Cache no longer needs to occupy one contiguous physical region and external fragmentation no longer prevents allocation.

#### KV Cache Sharing: Store a Common Prefix Only Once

Different sequences often share the same prefix, particularly in two situations:

1. **Shared system prompts**: multiple requests use the same system prompt or few-shot examples.
2. **Multiple samples from one prompt**: tasks such as program synthesis generate several candidate answers from the same prompt.

<figure>
  <img src="../../../posts/inference/paged-attention-sharing.png" alt="Multiple requests share KV Cache for a common system prompt and few-shot examples">
  <figcaption>Different requests can share the KV Cache corresponding to a common prefix and allocate new blocks only for their distinct inputs and outputs. Source: vLLM.</figcaption>
</figure>

PagedAttention implements this sharing with block-level copy-on-write (CoW). Multiple sequences can map their logical blocks to the same read-only physical blocks. Only when one sequence must modify a shared block does the system copy that block and redirect the sequence's write to its private copy. This avoids duplicate prefix storage without allowing one sequence's writes to affect another.

<figure>
  <img src="../../../posts/inference/paged-attention-parallel.png" alt="Multiple sampled sequences share prefix blocks and use block-level copy-on-write when they diverge">
  <figcaption>Multiple samples from one prompt initially share prefix blocks. When their sequences diverge, only the final shared block that requires a write is copied, and block references are updated accordingly. Source: vLLM.</figcaption>
</figure>

#### Other vLLM Optimizations

In addition to PagedAttention, vLLM reduces inference overhead through several techniques:

1. **Fusing block reads with attention computation**: physical-block access and attention are performed in one kernel, reducing kernel-launch overhead.
2. **Efficient attention kernels**: implementations such as FlashAttention and FlashDecoding accelerate attention computation.
3. **CUDA Graphs**: previously recorded GPU workflows are replayed to reduce CPU scheduling and kernel-launch overhead.

PagedAttention does not change the model's computation. Its core contribution is to <strong>manage GPU memory for dynamic workloads using the operating-system idea of paging</strong>. Continuous batching dynamically updates the set of active requests, while PagedAttention lets their KV caches grow on demand, be reclaimed promptly, and share prefixes.

## 5. Summary

1. **Inference has broad uses**: large-language-model inference powers practical applications such as chatbots, code completion, and agents, and is also an important part of model evaluation and reinforcement learning.
2. **Inference workloads differ from training workloads**: autoregressive generation has sequential dependencies, decoding is usually memory-bandwidth-bound, and online request arrivals and sequence lengths change dynamically.
3. **Model-level optimization can lower inference cost**: new attention architectures, quantization, pruning, knowledge distillation, and speculative sampling improve efficiency by reducing cache size, numerical precision, model size, or generation cost.
4. **Ideas from systems research are equally important**: speculative sampling borrows from speculative execution, while PagedAttention borrows paging from operating systems, creating new optimization methods for large-language-model inference.
5. **There is substantial room for better model architectures**: inference efficiency is not merely a deployment concern; it can be considered during model design and training.

## References

[1] Stanford CS336. Lecture 10: Inference. [Online]. Available: https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py

[2] OpenAI Bests Google in Race for Consumer AI Token Consumption. [Online]. Available: https://www.pymnts.com/artificial-intelligence-2/2025/openai-bests-google-in-race-for-consumer-ai-token-consumption/

[3] DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence. [Online]. Available: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf

[4] How to Scale Your Model: Transformers. [Online]. Available: https://jax-ml.github.io/scaling-book/transformers/

[5] How to Scale Your Model: Inference. [Online]. Available: https://jax-ml.github.io/scaling-book/inference/

[6] Llama 2: Open Foundation and Fine-Tuned Chat Models. [Online]. Available: https://arxiv.org/abs/2307.09288

[7] NVIDIA H100 Tensor Core GPU. [Online]. Available: https://www.nvidia.com/en-us/data-center/h100/

[8] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. [Online]. Available: https://arxiv.org/abs/2305.13245

[9] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. [Online]. Available: https://arxiv.org/abs/2405.04434

[10] Reducing Transformer Key-Value Cache Size with Cross-Layer Attention. [Online]. Available: https://arxiv.org/abs/2405.12981

[11] Longformer: The Long-Document Transformer. [Online]. Available: https://arxiv.org/abs/2004.05150

[12] Generating Long Sequences with Sparse Transformers. [Online]. Available: https://arxiv.org/abs/1904.10509

[13] Mistral 7B. [Online]. Available: https://arxiv.org/abs/2310.06825

[14] FP8-LM: Training FP8 Large Language Models. [Online]. Available: https://arxiv.org/abs/2310.18313

[15] FP8 versus INT8 for Efficient Deep Learning Inference. [Online]. Available: https://arxiv.org/abs/2303.17951

[16] GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. [Online]. Available: https://arxiv.org/abs/2210.17323

[17] AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration. [Online]. Available: https://arxiv.org/abs/2306.00978

[18] Compact Language Models via Pruning and Knowledge Distillation. [Online]. Available: https://arxiv.org/abs/2407.14679

[19] Fast Inference from Transformers via Speculative Decoding. [Online]. Available: https://arxiv.org/abs/2211.17192

[20] Accelerating Large Language Model Decoding with Speculative Sampling. [Online]. Available: https://arxiv.org/abs/2302.01318

[21] Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. [Online]. Available: https://arxiv.org/abs/2401.10774

[22] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. [Online]. Available: https://arxiv.org/abs/2401.15077

[23] Orca: A Distributed Serving System for Transformer-Based Generative Models. [Online]. Available: https://www.usenix.org/system/files/osdi22-yu.pdf

[24] Efficient Memory Management for Large Language Model Serving with PagedAttention. [Online]. Available: https://arxiv.org/abs/2309.06180
