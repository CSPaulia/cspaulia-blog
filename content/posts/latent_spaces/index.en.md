---
title: "Latent Spaces and Network Architectures"
date: 2026-09-16T10:00:00+08:00
series:
    main: "Generative Models"
    subseries: "Fundamentals"
categories: ["Generative Models"]
tags: ["Latent Spaces", "Autoencoders", "Diffusion Models"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Notes for Lecture 4 of MIT's “Introduction to Flow Matching and Diffusion Models 2026”: why latent spaces are needed, standard autoencoders, variational autoencoders (VAEs), latent diffusion models (LDMs), neural network architectures for diffusion models, and large-scale case studies."
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
    image: "autoencoder_architecture.png"
    alt: "Autoencoder architecture: data space compressed by an encoder into latent space, then reconstructed by a decoder"
    caption: "The encoder–latent space–decoder structure of an autoencoder"
    relative: true
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

## 1. Standard Autoencoders

### 1.1. Why Latent Spaces Are Needed: Three Problems from High-Dimensional Data

A 600×1000 color image has 3 color channels. Flattened into a vector, its dimension is:

\[
3 \times 600 \times 1000 = 1.8 \times 10^6
\]

This is an extremely high-dimensional space, and modeling directly in it runs into three problems:

- **GPU memory blows up**: the vector field network takes and returns vectors of this dimension.
- **The learning problem becomes very hard**: the network must fit a vector field in a 1.8-million-dimensional space.
- **Redundancy**: nearby pixels are highly correlated, so the raw representation repeats a great deal of information.

High dimensionality hurts diffusion models more than supervised learning, for two reasons:

- A diffusion model learns a vector field \(u_t^{\theta}: \mathbb{R}^d \to \mathbb{R}^d\), so its <strong>output is itself high-dimensional</strong>, not a single scalar label.
- Sampling applies this vector field many times (simulating an ODE), so high-dimensional errors accumulate across iterations.

If the data can be moved into a much lower-dimensional space and the distribution learned there, both problems become easier.

### 1.2. Constructing a Latent Space: Encoder and Decoder

A latent space is constructed by an <strong>autoencoder</strong>, which consists of an encoder and a decoder. Together they compress samples from data space into latent space and reconstruct them back:

\[
\text{Data space } \mathbb{R}^d \;\xrightarrow{\ \text{Encoder}\ }\; \text{Latent space } \mathbb{R}^k \;\xrightarrow{\ \text{Decoder}\ }\; \text{Data space } \mathbb{R}^d
\]

The latent dimension \(k\) is smaller than the data dimension \(d\); this step is called <strong>compression</strong>. Because \(k < d\), the encoder is forced to discard information, and the decoder can only rely on what survives — which is exactly why an autoencoder learns a meaningful representation.

<figure>
  <img src="../../../posts/latent_spaces/autoencoder_architecture.png" alt="Autoencoder architecture: data space compressed by an encoder into latent space, then reconstructed by a decoder" width="100%" />
  <figcaption>Figure 1: The encoder–latent space–decoder structure of an autoencoder. A sample in data space \(\mathbb{R}^d\) is compressed by the encoder into a latent variable in \(\mathbb{R}^k\), then reconstructed back into data space by the decoder. The same image appears at both ends, indicating that the reconstruction target is the input itself. Figure source: MIT 6.S184 Lecture 4.</figcaption>
</figure>

### 1.3. Standard Autoencoders: Encoder, Decoder, and Reconstruction Loss

Write a sample in data space as \(x \in \mathbb{R}^d\) and a latent variable as \(z \in \mathbb{R}^k\). A standard autoencoder consists of two networks:

| Component | Notation | Mapping | Parameters |
| --- | --- | --- | --- |
| Encoder | \(\mu_\phi\) | \(\mathbb{R}^d \to \mathbb{R}^k\) | \(\phi\) |
| Decoder | \(\mu_\theta\) | \(\mathbb{R}^k \to \mathbb{R}^d\) | \(\theta\) |

The encoder maps a sample to a latent variable \(z = \mu_\phi(x)\), and the decoder maps the latent variable back to data space \(\mu_\theta(z)\). Training requires that encoding followed by decoding reconstructs the input, which gives the <strong>reconstruction loss</strong>:

\[
L_{\mathrm{recon}}(\theta, \phi) = \mathbb{E}_{x \sim p_{\mathrm{data}}}\left[\left\|\mu_\theta(\mu_\phi(x)) - x\right\|^2\right]
\]

That is, the mean squared error between the reconstruction and the original image. Once trained, a standard autoencoder succeeds at two things:

- ✓ **Compression**: the data is mapped into a low-dimensional latent space.
- ✓ **Reconstruction**: the latent variable can be decoded back into the original sample.

At the same time, the encoder also <strong>induces a distribution over latent variables</strong>: given a data distribution \(x \sim p_{\mathrm{data}}\) and setting \(z = \mu_\phi(x)\), we have

\[
z \sim p_{\mathrm{latent}}
\]

This \(p_{\mathrm{latent}}\) is the push-forward of \(p_{\mathrm{data}}\) through the encoder — not a distribution we chose in advance.

#### The Latent Distribution Is Unconstrained: Compressible but Hard to Sample

The full plan for using a latent space has two steps: transform the data into latents, then learn the distribution of latents. This raises a key question:

> What happens to the data distribution when it is transformed into latent space?

The answer is that <strong>we don't know</strong>. The reconstruction loss only constrains reconstruction quality; it says nothing about the distribution of the latent variables.

The consequence is that we may have made the learning problem (that is, training) much harder. The latent space reduces the dimension, but if \(p_{\mathrm{latent}}\) itself has a strange shape, learning it is no easier than learning the distribution in the original space. The end result is that we <strong>can compress, but cannot learn to sample the distribution</strong>.

<figure>
  <img src="../../../posts/latent_spaces/bad_latent_space.png" alt="The latent space learned by a standard autoencoder: samples scatter without clear structure" width="60%" />
  <figcaption>Figure 2: A "bad" latent space. Connecting nearby samples with lines shows that the encoder's latent distribution is disorganized, with no usable geometric structure, which makes training a generative model on top of it difficult. Figure source: MIT 6.S184 Lecture 4.</figcaption>
</figure>

Solving this requires not an autoencoder that reconstructs better, but one that produces a <strong>"nice" latent distribution</strong>: \(p_{\mathrm{latent}}\) should have a simple, regular shape so that it can be trained on and sampled from effectively. The Variational Autoencoder (VAE) is the answer in that direction.

## 2. Variational Autoencoders (VAE)

A variational autoencoder takes a different approach: the encoder no longer outputs a point, but a distribution.

### 2.1. Encoder and Decoder: From Deterministic Mapping to Probability Distributions

The VAE encoder is a conditional distribution:

\[
q_\phi(z \mid x) = \mathcal{N}\left(z;\, \mu_\phi(x),\, \sigma_\phi^2(x) I_k\right)
\]

Given a sample \(x\), the encoder outputs a Gaussian distribution: both the mean \(\mu_\phi(x)\) and the variance \(\sigma_\phi^2(x)\) are predicted by networks, with a diagonal covariance \(\sigma_\phi^2(x) I_k\). Where a standard autoencoder learns only a mean network, the VAE encoder learns an extra variance network, which describes how large a region of latent space this sample should be encoded into.

The decoder is likewise modeled as a conditional distribution:

\[
p_\theta(x \mid z) = \mathcal{N}\left(x;\, \mu_\theta(z),\, \sigma^2 I_d\right)
\]

The key difference is that the variance \(\sigma^2 I_d\) is a <strong>fixed constant</strong>, not learned by a network. With the variance fixed, maximizing the log-likelihood is equivalent to making the mean approximate \(x\), so given a latent variable \(z\) the best reconstruction is simply the mean:

\[
x \approx \mu_\theta(z)
\]

This is what "deterministic decoding" means: the decoding side never needs to draw a random sample; taking the mean of the distribution is enough.

The sampling procedures at the two ends are:

- **Encode**: sample from the encoder distribution, \(z \sim q_\phi(\cdot \mid x)\).
- **Decode**: sample from the decoder distribution, \(x \sim p_\theta(\cdot \mid z)\); since the variance is fixed, taking \(\mu_\theta(z)\) suffices in practice.

### 2.2. VAE Reconstruction Loss: Negative Log-Likelihood Equals a Weighted Mean Squared Error

A VAE's reconstruction term is not written directly as a mean squared error: it is defined as the negative log-likelihood of the decoder, and becomes a mean squared error under a Gaussian decoder:

\[
\begin{aligned}
L_{\mathrm{recon}}(\phi, \theta)
&= \mathbb{E}_{\substack{x \sim p_{\mathrm{data}},\\ z \sim q_\phi(\cdot \mid x)}} \left[ -\log p_\theta(x \mid z) \right] \\
&= \mathbb{E}_{\substack{x \sim p_{\mathrm{data}},\\ z \sim q_\phi(\cdot \mid x)}} \left[ \frac{1}{2\sigma^2}\left\|x - \mu_\theta(z)\right\|^2 \right]
\end{aligned}
\]

The expectation is taken over two distributions at once: \(x\) comes from the data distribution \(p_{\mathrm{data}}\), and \(z\) is sampled from the encoder distribution \(q_\phi(\cdot \mid x)\).

<details>
<summary>Derivation: From negative log-likelihood to mean squared error</summary>

Substituting the Gaussian decoder from Section 2.1, \(p_\theta(x \mid z) = \mathcal{N}\left(x;\, \mu_\theta(z),\, \sigma^2 I_d\right)\), and using the Gaussian log-density

\[
-\log p_\theta(x \mid z) = \frac{1}{2\sigma^2}\left\|x - \mu_\theta(z)\right\|^2 + \text{const}
\]

</details>

The constant term depends only on the fixed \(\sigma^2\) and contains no \(\phi, \theta\), so it has no bearing on optimization. The factor \(\frac{1}{2\sigma^2}\) is likewise a fixed constant: it does not change the optimum, only the scale of the loss. Under a Gaussian decoder, the reconstruction loss is therefore essentially the mean squared error of a standard autoencoder.

### 2.3. Prior Loss: Pulling the Latent Distribution Toward a Standard Normal

The reconstruction loss only constrains reconstruction quality; the shape of the latent distribution \(p_{\mathrm{latent}}\) remains entirely unconstrained. Making the latent distribution take a shape we specify requires an extra term that pulls it there. The prior loss chooses the standard normal distribution \(\mathcal{N}(0, I)\) as the target shape and directly minimizes the KL divergence between the encoder distribution and that target:

\[
L_{\mathrm{prior}}(\phi) = \mathbb{E}_{x \sim p_{\mathrm{data}}}\left[ D_{\mathrm{KL}}\left(q_\phi(z \mid x) \,\|\, \mathcal{N}(0, I)\right) \right]
\]

The Kullback-Leibler divergence (KL divergence) measures how different two distributions are: \(D_{\mathrm{KL}}(q \,\|\, p) \ge 0\), with equality if and only if \(q = p\). When both distributions are diagonal normal, it has a closed form:

\[
D_{\mathrm{KL}}(q \,\|\, p) = \frac{1}{2}\left( \underbrace{\mathcal{K}\left(\frac{\sigma_q^2}{\sigma_p^2}\right)}_{\text{Dist. of variances}} + \underbrace{\frac{\left\|\mu_q - \mu_p\right\|^2}{\sigma_p^2}}_{\text{Dist. of mean}} \right), \qquad \mathcal{K}(\alpha) = \sum_i \left( \alpha_i - \log \alpha_i - 1 \right)
\]

<details>
<summary>Derivation: KL divergence between two normal distributions</summary>

Start from the definition of KL divergence, \(D_{\mathrm{KL}}(q \,\|\, p) = \mathbb{E}_{x \sim q}\left[\log q(x) - \log p(x)\right]\) — the expectation of the difference of the log-densities. The covariance is diagonal, so the dimensions separate; consider dimension \(i\). The log-density of a normal distribution is

\[
\log q_i(x_i) = -\frac{1}{2}\log 2\pi - \frac{1}{2}\log \sigma_{q,i}^2 - \frac{\left(x_i - \mu_{q,i}\right)^2}{2\sigma_{q,i}^2}
\]

The term \(-\frac{1}{2}\log 2\pi\) is identical for \(p\) and cancels in the difference, leaving

\[
\log q_i(x_i) - \log p_i(x_i) = \frac{1}{2}\log\frac{\sigma_{p,i}^2}{\sigma_{q,i}^2} + \frac{1}{2}\left( \frac{\left(x_i - \mu_{p,i}\right)^2}{\sigma_{p,i}^2} - \frac{\left(x_i - \mu_{q,i}\right)^2}{\sigma_{q,i}^2} \right)
\]

Taking the expectation over \(x \sim q\) needs only two moments: one is the definition of the variance, and the other splits \(x_i - \mu_{p,i}\) into \(\left(x_i - \mu_{q,i}\right) + \left(\mu_{q,i} - \mu_{p,i}\right)\), whose cross term has zero expectation:

\[
\mathbb{E}_{q}\left[\left(x_i - \mu_{q,i}\right)^2\right] = \sigma_{q,i}^2, \qquad \mathbb{E}_{q}\left[\left(x_i - \mu_{p,i}\right)^2\right] = \left(\mu_{q,i} - \mu_{p,i}\right)^2 + \sigma_{q,i}^2
\]

Substituting back and collecting terms — the first expectation equals \(1\):

\[
D_{\mathrm{KL}}(q_i \,\|\, p_i) = \frac{1}{2}\left( \log\frac{\sigma_{p,i}^2}{\sigma_{q,i}^2} + \frac{\left(\mu_{q,i} - \mu_{p,i}\right)^2}{\sigma_{p,i}^2} + \frac{\sigma_{q,i}^2}{\sigma_{p,i}^2} - 1 \right)
\]

Write \(\alpha_i = \sigma_{q,i}^2 / \sigma_{p,i}^2\); then \(\log\frac{\sigma_{p,i}^2}{\sigma_{q,i}^2} = -\log\alpha_i\), which together with the remaining two terms gives exactly \(\alpha_i - \log\alpha_i - 1\). Summing over all dimensions yields the formula above:

\[
D_{\mathrm{KL}}(q \,\|\, p) = \frac{1}{2}\left( \sum_i \left( \alpha_i - \log \alpha_i - 1 \right) + \sum_i \frac{\left(\mu_{q,i} - \mu_{p,i}\right)^2}{\sigma_{p,i}^2} \right)
\]

The first sum involves only the variances and the second only the means, giving \(\mathcal{K}(\sigma_q^2/\sigma_p^2)\) and \(\|\mu_q - \mu_p\|^2/\sigma_p^2\) respectively.

</details>

<figure>
  <img src="../../../posts/latent_spaces/kl_divergence_k.png" alt="Plot of K(α) = α − log α − 1, with its minimum of 0 at α = 1" width="60%" />
  <figcaption>Figure 3: The function \(\mathcal{K}(\alpha) = \alpha - \log \alpha - 1\). Its minimum is 0 at \(\alpha = 1\), the degenerate case \(\sigma_q^2 = \sigma_p^2\) where the variance term vanishes — consistent with the KL divergence being non-negative and zero only when the two distributions coincide. Figure source: MIT 6.S184 Lecture 4.</figcaption>
</figure>

The first term involves only the variances, the second only the means. With the prior \(\mathcal{N}(0, I)\), we have \(\sigma_p^2 = 1\) and \(\mu_p = 0\), so both denominators collapse to 1 and the prior loss follows directly:

\[
L_{\mathrm{prior}}(\phi) = \mathbb{E}_{x \sim p_{\mathrm{data}}}\left[ \frac{1}{2}\left( \mathcal{K}\left(\frac{\sigma_\phi^2(x)}{1}\right) + \frac{\left\|\mu_\phi(x)\right\|^2}{1} \right) \right]
\]

Expanding \(\mathcal{K}\) into a sum over dimensions gives the form used in practice:

\[
L_{\mathrm{prior}}(\phi) = \mathbb{E}_{x \sim p_{\mathrm{data}}}\left[ \frac{1}{2} \sum_{j=1}^{k} \left( \mu_{\phi,j}^2(x) + \sigma_{\phi,j}^2(x) - \log \sigma_{\phi,j}^2(x) - 1 \right) \right]
\]

With the prior loss added:

- ✓ **Compression**
- ✓ **Training**
- ✗ **Reconstruction**

### 2.4. Training a VAE: Two Losses and β

Training combines the reconstruction loss and the prior loss into a single objective, with a coefficient \(\beta \ge 0\) setting their relative weight. The total loss is written \(L_{\mathrm{VAE}}\), and expanding it simply writes the Section 2.2 and Section 2.3 losses side by side:

\[
\begin{aligned}
L_{\mathrm{VAE}}(\phi, \theta)
&= L_{\mathrm{recon}}(\phi, \theta) + \beta\,L_{\mathrm{prior}}(\phi) \\
&= \mathbb{E}_{x \sim p_{\mathrm{data}},\, z \sim q_\phi(\cdot \mid x)}\left[ \frac{1}{2\sigma^2}\left\| x - \mu_\theta(z) \right\|^2 + \beta\,\frac{1}{2}\left( \mathcal{K}\left(\frac{\sigma_\phi^2(x)}{1}\right) + \frac{\left\|\mu_\phi(x)\right\|^2}{1} \right) \right]
\end{aligned}
\]

| **Algorithm 7** \(\beta\)-VAE Training Procedure |
| --- |
| **Input**: a dataset of samples \(x \sim p_{\mathrm{data}}\), encoder networks \((\mu_\phi(x), \log\sigma_\phi^2(x))\), decoder network \(\mu_\theta(z)\), latent dimension \(k\), constants \(\beta \ge 0\), \(\sigma^2 > 0\) |
| 1: **for** each mini-batch \(\{x_i\}_{i=1}^{B}\) **do** |
| 2: \(\quad\)Encode each \(x_i\): \(\mu_i \leftarrow \mu_\phi(x_i)\), \(\log\sigma_i^2 \leftarrow \log\sigma_\phi^2(x_i)\) |
| 3: \(\quad\)Sample noise \(\epsilon_i \sim \mathcal{N}(0, I_k)\) |
| 4: \(\quad\)Reparameterize: \(z_i \leftarrow \mu_i + \sigma_i \odot \epsilon_i\) |
| 5: \(\quad\)Decode the mean: \(\hat{x}_i \leftarrow \mu_\theta(z_i)\) |
| 6: \(\quad\)Compute the reconstruction loss \(\mathcal{L}_{\mathrm{recon}} \leftarrow \frac{1}{B}\sum_{i=1}^{B}\frac{1}{2\sigma^2}\lVert x_i - \hat{x}_i\rVert^2\) |
| 7: \(\quad\)Compute the KL loss to the prior \(p_{\mathrm{prior}}(z) = \mathcal{N}(0, I_k)\): \(\mathcal{L}_{\mathrm{KL}} \leftarrow \frac{1}{B}\sum_{i=1}^{B}\frac{1}{2}\sum_{j=1}^{k}\left(\mu_{i,j}^2 + \sigma_{i,j}^2 - \log\sigma_{i,j}^2 - 1\right)\) |
| 8: \(\quad\)Total loss \(\mathcal{L} \leftarrow \mathcal{L}_{\mathrm{recon}} + \beta\,\mathcal{L}_{\mathrm{KL}}\) |
| 9: \(\quad\)Update the model parameters \((\phi, \theta)\) via gradient descent on \(\mathcal{L}\) |
| 10: **end for** |

Here \(\mathcal{L}_{\mathrm{recon}}\) and \(\mathcal{L}_{\mathrm{KL}}\) — written \(L_{\mathrm{recon}}\) and \(L_{\mathrm{prior}}\) above — are the mini-batch averages of the Section 2.2 reconstruction loss and the Section 2.3 prior loss, respectively.

\(\beta\) sets how much weight the prior loss carries: at \(\beta = 0\) the model reduces to a standard autoencoder, and larger \(\beta\) pulls the latent distribution closer to \(\mathcal{N}(0, I)\).

With the prior loss added:

- ✓ **Compression**
- ✓ **Reconstruction**
- ✓ **Training**

### 2.5. The Reparameterization Trick: Making Sampling Differentiable

One expectation in the Section 2.4 total loss has to sample \(z\) from \(q_\phi(z \mid x)\), but that sampling step is not differentiable, so gradients cannot flow back through it. The <strong>reparameterization trick</strong> moves the randomness outside the parameters: draw a noise variable that depends on no parameter from a standard normal, then apply a deterministic transform using the encoder's output.

The encoder gives the distribution

\[
q_\phi(z \mid x) = \mathcal{N}\left(z;\, \mu_\phi(x),\, \sigma_\phi^2(x) I_k\right)
\]

Reparameterization splits the sampling into two steps — draw the noise from a standard normal that does not depend on \(\phi\), then apply the transform:

\[
\epsilon \sim \mathcal{N}(0, I_k), \qquad z = \mu_\phi(x) + \sigma_\phi(x)\,\epsilon \implies z \sim q_\phi(\cdot \mid x)
\]

The dependence of \(z\) on \(\phi\) is now a deterministic linear relation, so gradients flow normally, and the distribution of \(z\) is still \(q_\phi(\cdot \mid x)\).

Substituting this expression for \(z\) back into the total loss replaces the random variable in the expectation from \(z\) with \(\epsilon\):

\[
L_{\mathrm{VAE}}(\phi, \theta) = \mathbb{E}_{x \sim p_{\mathrm{data}},\ \epsilon \sim \mathcal{N}(0, I_k)}\left[ \frac{1}{2\sigma^2}\left\| x - \mu_\theta\left(\mu_\phi(x) + \sigma_\phi(x)\,\epsilon\right) \right\|^2 + \frac{\beta}{2}\left( \mathcal{K}\left(\frac{\sigma_\phi^2(x)}{1}\right) + \frac{\left\|\mu_\phi(x)\right\|^2}{1} \right) \right]
\]

## 3. Latent Diffusion Models (LDM)

With a VAE that compresses data into a regular latent space, a diffusion model no longer has to train on pixels — it can train in latent space instead. The full procedure is:

1. **Data**: take all training data \(x_1, \dots, x_N\) (for example, all images on the internet).
2. **Encoding**: convert all images into latents (for a VAE, simply take the mean prediction).
3. **Latent data**: this gives a dataset of latents \(z_1, \dots, z_N\), of significantly smaller size than the original high-resolution images.
4. **Latent diffusion model**: build a diffusion model on the dataset of latents — the diffusion model now generates latent vectors.
5. **Decoding**: after sampling from the diffusion model, map the generated latent back to data space.

Return the decoded image as the sample. The recipe is exactly the same as before; only the dataset has been replaced by the transformed latent dataset.

Virtually all AI-generated images or videos you see are generated in latent spaces. The reason is memory: running diffusion directly in pixel space blows up memory. Once the data is compressed into a latent space, the model can concentrate its capacity on the key content instead of spreading it evenly across every pixel.

The tensor shapes of two representative models:

| Model | Image tensor | Latent tensor | Compression |
| --- | --- | --- | --- |
| Stable Diffusion | \(3 \times 256 \times 256\) | \(4 \times 32 \times 32\) | \(48\times\) |
| FLUX 2.0 | \(3 \times 1024 \times 1024\) | \(32 \times 64 \times 64\) | \(24\times\) |

## 4. Neural Network Architectures

The previous sections treated the vector field \(u_t^\theta\) as a function that already exists. This chapter answers what network implements it: which inputs it takes, how each is encoded into vectors, and what structure processes them.

### 4.1. Parameterizing a Vector Field: The Three Inputs

The network built in this chapter outputs the vector field itself, written

\[
u_t^\theta(x \mid y)
\]

Each symbol comes from somewhere:

| Symbol | Meaning |
| --- | --- |
| \(u_t^\theta(x \mid y)\) | the vector field the network outputs |
| \(\theta\) | the network parameters |
| \(t\) | time, written as a subscript |
| \(x\) | the latent image |
| \(y\) | the prompt |

Time is a one-dimensional scalar, the latent image is a high-dimensional tensor, and the prompt is a sequence of text. The three have completely different shapes, so each has to be encoded into vectors before it enters the network.

### 4.2. Encoding Time: Sinusoidal Embedding

Time is only one-dimensional, while every other variable is high-dimensional. To make it “count” more, the network spreads it into a high-dimensional vector with the following sinusoidal embedding:

\[
\operatorname{TimeEmb}(t) = \frac{1}{\sqrt{d}}\left[\cos(2\pi w_1 t)\ \cdots\ \cos(2\pi w_{d/2}t)\ \ \sin(2\pi w_1 t)\ \cdots\ \sin(2\pi w_{d/2}t)\right]^{\top}
\]

The frequencies are chosen as a geometric sequence:

\[
w_i = w_{\min}\left(\frac{w_{\max}}{w_{\min}}\right)^{\frac{i-1}{d/2-1}}, \qquad i = 1, \dots, d/2
\]

The specific form matters less than the key property: the time embedding is a \(d\)-dimensional normalized vector,

\[
\left\| \operatorname{TimeEmb}(t) \right\| = 1
\]

### 4.3. Encoding the Language Prompt: Pre-trained Text Embeddings

The prompt is a sentence of natural language, for example “A dog running on grass in a park at sunshine in an Italian city.” To turn text into vectors, most models rely directly on pre-trained language embeddings:

- CLIP embeddings (Contrastive Language-Image Pre-training)
- T5 embeddings and similar (other pre-trained models)
- LLM embeddings also work

The result of these embeddings is that the prompt becomes a sequence of vectors of length \(S\):

\[
\operatorname{PromptEmbed}(y_{\mathrm{raw}}) \in \mathbb{R}^{S \times k}
\]

### 4.4. Patchify: Turning an Image into a Sequence of Vectors

A transformer processes sequences of vectors, while an image is a three-dimensional tensor, so the image is first cut into patches and flattened into a sequence. Write the image as

\[
x \in \mathbb{R}^{3 \times H \times W}
\]

Cutting it into patches and flattening gives

\[
\tilde{x} = \operatorname{Patchify}(x), \qquad \tilde{x} \in \mathbb{R}^{L \times k}
\]

where \(L\) is the sequence length, that is, the number of patches.

<figure>
  <img src="../../../posts/latent_spaces/patchify.png" alt="Patchify: an image cut into a regular grid of patches, then flattened into a sequence of vectors" width="100%" />
  <figcaption>Figure 4: Patchify turns an image into a sequence of vectors. On the left is the original image \(x \in \mathbb{R}^{3 \times H \times W}\); on the right the image is cut into a regular grid of patches; below, each patch is flattened into a vector and the vectors are laid out in a row, giving the sequence \(\tilde{x} \in \mathbb{R}^{L \times k}\). Source: MIT 6.S184 Lecture 4.</figcaption>
</figure>

### 4.5. Diffusion Transformer (DiT): Three Steps and the DiTBlock

The **diffusion transformer (DiT)** is a transformer aimed at diffusion models: it adapts the transformer to the specific form that diffusion models require. The network runs in three steps:

**1. Inputs**: the three inputs are each encoded into a sequence of vectors

\[
\tilde{t} = \operatorname{TimeEmb}(t) \in \mathbb{R}^k, \qquad \tilde{x}_0 = \operatorname{PatchEmb}(x) \in \mathbb{R}^{N \times k}, \qquad \tilde{y} = \operatorname{PromptEmb}(y) \in \mathbb{R}^{S \times k}
\]

**2. Attention loop**: the image sequence is passed repeatedly through the DiT block

\[
\tilde{x}_{i+1} = \operatorname{DiTBlock}(\tilde{x}_i, \tilde{t}, \tilde{y}) \in \mathbb{R}^{N \times k}, \qquad i = 0, \dots, N-1
\]

**3. Unpatchify**: the processed sequence is restored to image shape, producing the vector field

\[
u = \operatorname{Unpatchify}(\tilde{z}_N \tilde{W}) \in \mathbb{R}^{C \times H \times W}
\]

The \(\operatorname{DiTBlock}\) called repeatedly in step 2 is the core of the network. It takes all three inputs at once, each through its own mechanism:

- **The image goes through self-attention**: queries, keys, and values all come from the image, so the image processes itself.
- **The text goes through cross-attention**: queries come from the image, keys and values come from the text embeddings, so the image attends to the text.
- **Time goes through adaptive layer normalization**: the scaling and offset of the normalization are determined by the time variable.

The results of the three inputs are combined by summing them into a single stream.

The best way to understand a transformer is to implement it — Lab 03 has an implementation, and the lecture notes cover the details.

## 5. Large-Scale Diffusion Models

### 5.1. Case Study: Stable Diffusion 3

An overview of Stable Diffusion 3:

- A flow matching model with “straight line” schedulers (the CondOT path)
- Classifier-free guidance with weight 2.0 - 5.0
- Flow matching in latent space (using a pre-trained VAE)
- Number of model parameters: 8 billion
- Number of sampling steps: 50
- Dataset: LAION

<figure>
  <img src="../../../posts/latent_spaces/sd3_samples.png" alt="Samples generated by Stable Diffusion 3: three images in different styles" width="100%" />
  <figcaption>Figure 5: Samples generated by Stable Diffusion 3. Source: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis [3].</figcaption>
</figure>

Its network architecture:

- It conditions on **CLIP** (coarse-grained) and **T5-XXL** (sequence-level) text embeddings via cross-attention.
- **MM-DiT architecture**: it extends DiT from class-conditioning to text-conditioning, and processes text and images through the entire network via cross-attention.

<figure>
  <img src="../../../posts/latent_spaces/sd3_architecture.png" alt="The architecture of Stable Diffusion 3: three text encoders, the MM-DiT backbone, and the internals of a single MM-DiT block" width="100%" />
  <figcaption>Figure 6: The architecture of Stable Diffusion 3. On the left is the overall pipeline: the prompt goes through three text encoders — CLIP-G/14, CLIP-L/14, and T5-XXL — to produce text embeddings; the noisy latent is patched and positionally encoded, then passes through \(d\) MM-DiT blocks before being unpatchified back into image space, while the timestep is sinusoidally encoded and modulates those blocks together with the pooled text embedding. On the right is the internals of a single MM-DiT block: the text and image streams are each normalized and linearly projected, share a single attention layer, and then each pass through an MLP on the way back. Source: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis [3].</figcaption>
</figure>

### 5.2. Case Study: Meta MovieGen

An overview of Meta MovieGen:

- A flow matching model with “straight line” schedulers (the CondOT path)
- Classifier-free guidance
- Flow matching in latent space (using a pre-trained VAE) — especially crucial for video, because of the added time dimension
- Network architecture: DiT adapted to video (what changes?)
- Number of model parameters: 30 billion
- It used 6,144 H100 GPUs

<figure>
  <img src="../../../posts/latent_spaces/moviegen_frames.png" alt="Video frames generated by Meta MovieGen: a snow monkey in a hot spring and a close-up of a splash" width="100%" />
  <figcaption>Figure 7: Sample video frames generated by Meta MovieGen. On the left, a snow monkey soaking in a hot spring; on the right, a close-up of a splash. Source: MIT 6.S184 Lecture 4.</figcaption>
</figure>

## References

[1] P. Holderrieth and R. Shprints, "Latent Spaces, Neural network architectures," MIT 6.S184 Lecture 4 slides, 2026. [Online]. Available: https://diffusion.csail.mit.edu/2026/docs/20260128_Lecture_04_edited.pdf

[2] W. Peebles and S. Xie, "Scalable Diffusion Models with Transformers," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4195–4205.

[3] P. Esser, S. Kulal, A. Blattmann, et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," in Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.
