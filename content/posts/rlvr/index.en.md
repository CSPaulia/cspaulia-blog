---
title: "Reinforcement Learning with Verifiable Rewards (RLVR)"
date: 2026-08-25T22:05:00+08:00
series:
  - main: "Deep Reinforcement Learning"
    subseries: "Policy Optimization"
  - main: "Large Language Model"
    subseries: "Fine-tuning"
categories: ["Large Language Model", "Reinforcement Learning"]
tags: ["RLVR", "GRPO", "PPO", "Reasoning Models", "Verifiable Rewards"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "An introduction to PPO, GRPO and its variants motivated by reward overoptimization, followed by RLVR case studies in DeepSeek-R1, Kimi K1.5, Qwen 3, and agentic reinforcement learning."
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
editPost:
  URL: "https://cspaulia.github.io/cspaulia-blog/content/"
  Text: "Suggest Changes"
  appendFilePath: true
---

## 1. RLVR and RLHF: Different Reward Sources

| Dimension | RLHF | RLVR |
| --- | --- | --- |
| Reward source | Human-preference data | Results produced by rules or verifiers |
| Reward generation | Trains a reward model from human preferences, then uses it to score new responses | Checks an answer directly and assigns a reward from the verification result |
| Type of judgment | Can represent subjective preferences such as helpfulness, style, and safety | Requires an automatically verifiable criterion |
| Typical tasks | Open-ended question answering, writing style, and safety alignment | Mathematics, code, and formatting constraints |
| Policy update | Can use PPO, GRPO, or related algorithms | Can use PPO, GRPO, or related algorithms |

Both RLHF and RLVR optimize a model policy using reward signals. PPO and GRPO determine <strong>how the policy is updated</strong>; RLHF and RLVR differ in <strong>how the reward is obtained</strong>.

> **In one sentence: RLHF scores responses with a learned judge, while RLVR scores them with a verifier that can check the answer directly.**

## 2. DeepSeek-R1-Zero: Reinforcement Learning Directly from a Base Model

### 2.1 Training Pipeline and Essential Recipe

**Pipeline: DeepSeek-V3-Base → RL (GRPO) → DeepSeek-R1-Zero.**

R1-Zero performs **no reasoning SFT before RL**. Cold-start reasoning SFT belongs to the full DeepSeek-R1 pipeline, not R1-Zero. The [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) gives the following essential recipe:

| Component | DeepSeek-R1-Zero choice |
| --- | --- |
| Initial model | DeepSeek-V3-Base |
| Policy update | GRPO |
| Training tasks | Verifiable mathematics, coding, STEM, and logic problems |
| Accuracy reward | Rule-based checks such as answer matching and code tests |
| Format reward | Requires reasoning and answers to appear in designated tags |
| Process supervision | No human reasoning traces and no process reward model (PRM) |

The rule-based reward can be summarized as

\[
R_{\mathrm{rule}}=R_{\mathrm{accuracy}}+R_{\mathrm{format}}.
\]

The accuracy reward checks only the final result, while the format reward requires a `<think>...</think><answer>...</answer>` structure.

This design serves four purposes:

- separating the reasoning process from the final answer;
- allowing the final answer to be extracted for rule-based verification;
- checking whether the model follows the required format;
- measuring chain-of-thought length and language consistency separately.

Whether CoT is supervised, and which part of a response receives a reward, depends on the training method:

| Training method | Provided data | Training signal |
| --- | --- | --- |
| Final-answer SFT | Problem + answer | Imitates answer tokens |
| CoT SFT | Problem + reasoning process + answer | Imitates all reasoning and answer tokens |
| Outcome-reward RL | Problem + verifiable answer | Rewards the model according to final-answer correctness |
| Process-reward RL | Problem + intermediate-step evaluations | Evaluates individual reasoning steps |

R1-Zero belongs to the third category. For a mathematics problem, the training set provides a question and its ground-truth answer, while the model generates the following trajectory during rollout:

\[
x\longrightarrow
\underbrace{z}_{\text{model-generated CoT}}
\longrightarrow
\underbrace{y}_{\text{final answer}}.
\]

The reward mainly checks:

- whether the final answer \(y\) is correct;
- whether the output follows the `<think>` and `<answer>` format.

**The individual steps in the reasoning process \(z\) are usually not scored directly.** A trajectory can therefore contain redundant or locally incorrect steps and still receive an accuracy reward if its final answer is correct. The model is not told which reasoning method to follow, leaving it free to explore different solution trajectories.

> The original training set was not fully released. Its prompts are known to focus on problems with ground-truth answers or test cases, rather than human-written chains of thought.

### 2.2 Experimental Phenomena: Accuracy, Length, and Reflection Grow Together

- **Reasoning improves**: pass@1 on AIME 2024 rises from 15.6% to 71.0%; consistency voting over 16 samples reaches 86.7%.
- **Responses become longer**: the model allocates more tokens to difficult problems as training progresses.
- **Strategies diversify**: verification, error backtracking, and alternative solution attempts become increasingly common.
- **An “aha moment” appears**: intermediate checkpoints interrupt an approach with expressions such as “Wait” and reconsider the solution.

<figure style="text-align: center;">
  <img src="../../../posts/rlvr/r1-zero-training-phenomena.png" alt="Growth in DeepSeek-R1-Zero response length during training and an example of the model reconsidering its reasoning" loading="lazy">
  <figcaption>DeepSeek-R1-Zero produces longer responses during training and begins to reconsider intermediate reasoning steps. Source: CS336 Lecture 16.</figcaption>
</figure>

These observations require careful interpretation:

- Length and accuracy rise together, but **a longer response is not necessarily more correct**. [Dr. GRPO](https://arxiv.org/abs/2503.20783) argues that the original GRPO normalization can itself favor longer answers.
- “Wait” and self-correction patterns can already be sampled from the base model. The experiment therefore shows that RL strengthens these behaviors, not that their wording proves RL created an entirely new mechanism from nothing.
- The “aha moment” describes output behavior; it does not establish that the model has the same internal cognitive process as a human.

## 3. DeepSeek-R1: Combining Reasoning and General Capability with Multi-stage Training

### 3.1 Full Training Pipeline and Essential Recipe

**Pipeline: DeepSeek-V3-Base → cold-start reasoning SFT → reasoning RL → rejection sampling and a second SFT stage → all-scenario RL → DeepSeek-R1.**

R1-Zero shows that outcome rewards can strengthen reasoning, but it also suffers from language mixing, poor readability, and weak performance on general tasks. The full DeepSeek-R1 pipeline therefore adds SFT and extends its later rewards to general preferences and safety.

| Dimension | DeepSeek-R1-Zero | DeepSeek-R1 |
| --- | --- | --- |
| Reasoning SFT before RL | None | A small set of long-CoT data for cold start |
| First RL-stage rewards | Accuracy + format | Accuracy + format + language consistency |
| Later supervised data | None | About 600K reasoning samples + 200K non-reasoning samples |
| Final RL-stage rewards | None | Rule rewards for verifiable tasks; preference and safety rewards for general tasks |
| Goal | Test whether pure RL can strengthen reasoning | Improve reasoning, readability, general capability, and safety together |

### 3.2 Four Training Stages

#### Stage One: Cold-start Reasoning SFT

DeepSeek-V3-Base is first fine-tuned on thousands of high-quality long-CoT examples, giving the first RL stage a more readable and stable initial policy.

The cold-start data is constructed through several routes:

- Use long-CoT examples for few-shot prompting to generate new reasoning trajectories; alternatively, provide no examples and directly prompt the model to include reflection, verification, and summarization.
- retaining correct and readable trajectories produced by R1-Zero;
- human editing, followed by expansion or rewriting with DeepSeek-V3 and a second round of human checking.

<details>
<summary>Expand: How do few-shot and direct prompting differ, and which teacher model is used?</summary>

- **Few-shot prompting**: place a small number of complete “problem—long CoT—answer” examples in the prompt, then give the model a new problem and ask it to imitate their structure. No parameters are updated during generation; only filtered trajectories later become SFT data.
- **Direct prompting**: provide no complete examples, but explicitly ask the model to solve step by step, backtrack when it finds a contradiction, verify by substitution, and summarize the final answer. Reflection, verification, and summarization are generation instructions rather than human-written intermediate steps.
- **Teacher model**: the initial [paper](https://arxiv.org/abs/2501.12948) did not disclose a specific model for each of these two prompting routes. The revised pipeline confirms that DeepSeek-R1-Zero generates candidate reasoning trajectories and DeepSeek-V3 cleans up and rewrites their formatting and expression. It is therefore inaccurate to label DeepSeek-V3 as the teacher for every route.

</details>

This stage primarily establishes readable formatting and expression rather than enumerating every solution strategy. The exact data mixture and recipe were not fully released.

#### Stage Two: Reasoning-oriented Reinforcement Learning

Training continues from the cold-start model with GRPO. The setup largely follows R1-Zero but adds a language-consistency reward:

\[
R_{\mathrm{language}}=\frac{\text{number of words in the target language}}{\text{total number of words in the response}}.
\]

This reward discourages language mixing in the chain of thought and aligns the response language with the question. Ablations show a slight reduction in reasoning performance, but a clear improvement in readability.

#### Stage Three: Rejection Sampling and a Second SFT Stage

A checkpoint from the reasoning-oriented reinforcement learning stage (Stage Two) generates multiple candidate responses, after which rejection sampling retains high-quality data:

- **Verifiable tasks**: correct responses are selected using ground-truth answers or code tests.
- **Tasks that are difficult to verify with rules**: DeepSeek-V3 judges the model response against a reference answer.
- **Quality filtering**: mixed-language, extremely long-paragraph, and hard-to-read responses are removed.

This process produces about 600K reasoning samples and 200K non-reasoning samples covering writing, factual QA, translation, and related tasks. Two epochs of SFT on their mixture preserve long-form reasoning while restoring general instruction-following capability.

#### Stage Four: All-scenario Reinforcement Learning

The final stage still uses GRPO, but selects rewards by task type:

- **Reasoning tasks** continue to use rule-based correctness and format rewards.
- **General tasks** use a helpfulness reward model trained from preference data.
- **Safety tasks** evaluate the full response, including both reasoning and the final answer.
- **Multilingual tasks** retain the language-consistency reward.

This stage therefore combines RLVR and RLHF: the former supplies verifiable reasoning rewards, while the latter supplies preference and safety signals that are difficult to express as rules. **GRPO is only the policy-update algorithm.**

### 3.3 Distillation: Transferring R1 Reasoning Trajectories to Smaller Models

**Pipeline: DeepSeek-R1 generates about 800K samples → Qwen or Llama base models are fine-tuned on those responses.**

- The teacher supplies long reasoning trajectories and final answers, which the student learns directly.
- The published distillation results use SFT only, with no subsequent RL stage for the student.
- Released students include Qwen 1.5B, 7B, 14B, and 32B, plus Llama 8B and 70B.
- In a 32B controlled comparison, the distilled model outperforms a model trained with large-scale RL directly from Qwen2.5-32B-Base on every reported benchmark.

The takeaway is: **for a smaller model, imitating reasoning trajectories already discovered by a strong teacher is usually cheaper and more effective than rediscovering them through RL from scratch; discovering stronger strategies still depends on a capable base model and large-scale RL.**

<details>
<summary>View results for the DeepSeek-R1 distilled models</summary>

<figure style="text-align: center;">
  <img src="../../../posts/rlvr/r1-distillation-results.png" alt="Results of DeepSeek-R1 distilled models of different sizes on AIME, MATH, GPQA, LiveCodeBench, and CodeForces" loading="lazy">
  <figcaption>After SFT on roughly 800K R1-generated trajectories, Qwen and Llama students of several sizes acquire strong reasoning performance. Source: CS336 Lecture 16.</figcaption>
</figure>

</details>

### 3.4 Other Experiments: PRM and MCTS Did Not Deliver the Expected Gains

These are failures reported by the DeepSeek team in its particular training setup. **They do not imply that PRM or MCTS is ineffective for every reasoning model.**

#### Process Reward Model (PRM)

A [process reward model](https://arxiv.org/abs/2305.20050) does not wait until the response ends to check only the final answer. It reads the problem and the current reasoning prefix, then scores each intermediate step.

<details>
<summary>Expand: How are process rewards computed and used?</summary>

For a trajectory \(z_1,\ldots,z_m\), the reward at step \(k\) can be written as

\[
r_k=\operatorname{PRM}(x,z_{\le k}).
\]

Process rewards can provide denser feedback during policy training, and can also rerank candidate responses or guide search at inference time. For example, a model may correctly derive \(x+3=5\) and then incorrectly write \(x=8\). An outcome reward notices the error only after the response is complete, while a process reward can assign a low score as soon as the incorrect step appears.

</details>

- **There is no universal boundary for one “step”**: the same derivation can be written as one step or split into several, making consistent segmentation and annotation difficult.
- Determining whether an intermediate step is correct is also hard: model annotations are unreliable, while human annotations do not scale.
- A model-based PRM is vulnerable to reward hacking and adds reward-model retraining and inference costs.
- PRMs can help rerank candidate responses or guide search, but in these large-scale RL experiments their gains did not offset the added cost.

#### Monte Carlo Tree Search (MCTS)

[Monte Carlo Tree Search](https://arxiv.org/abs/1712.01815) is a search algorithm that repeatedly probes possible paths to find high-value ones. For language models, it avoids generating one complete response in a single pass and instead organizes alternative reasoning routes into a tree:

| Search-tree element | Meaning in language-model reasoning |
| --- | --- |
| Root | Problem and prompt |
| Node | A partially generated reasoning process |
| Branch | One possible next reasoning step |
| Leaf | A complete trajectory and final answer |
| Node score | An estimate produced by a value model, PRM, or answer verifier |

A search repeatedly performs four operations: select the most promising node, expand several possible next steps, evaluate the new paths, and back up their values to ancestor nodes. Its main purposes are to:

- explore multiple solution routes at inference time and allocate more compute to promising branches;
- choose a more reliable answer from candidate paths instead of relying on a single sample;
- use high-quality searched trajectories to update the actor and value model during training.

The DeepSeek team explored MCTS for scaling inference-time computation, but encountered the following problems at large training scale:

- The branching space of language generation is much larger than in board games; limiting expansions at each node can instead trap search in a local optimum.
- Search quality depends heavily on a fine-grained value model, which is itself difficult to train accurately.
- MCTS with a pretrained value model can improve inference-time search, but iteratively improving both the actor and value model through repeated self-search remains difficult.

## 4. Kimi k1.5: Scaling Reinforcement Learning with Long Context

### 4.1 Training Pipeline and Essential Recipe

**Pipeline: base model → vanilla SFT → long-CoT SFT → reasoning RL → Kimi k1.5.**

[Kimi k1.5](https://arxiv.org/abs/2501.12599) treats a long context as a reasoning search space. Rather than explicitly building a search tree, the model attempts, checks, backtracks, and corrects within one long sequence.

| Component | Kimi k1.5 choice |
| --- | --- |
| Training data | Balance math, code, science, and visual reasoning; filter difficulty by model success rate |
| Cold start | Lightweight SFT on a small, high-quality long-CoT dataset |
| Policy update | A variant of Online Policy Mirror Descent |
| Value model | No critic or value network |
| Reward | Test cases for code; a CoT reward model for mathematical answer equivalence |
| Length control | Prefer shorter correct answers and penalize longer incorrect answers |
| Long-trajectory training | 128K context, partial rollouts, and hybrid training–inference deployment |

### 4.2 Long-CoT SFT: Establishing Reasoning Patterns First

The training prompts are first filtered for difficulty and verifiability:

- **Difficulty estimation**: an SFT model answers each problem ten times at relatively high temperature. A lower pass rate indicates a harder problem.
- **Preventing lucky guesses**: multiple-choice, true/false, and proof questions are removed. A prompt is also removed if the model can guess its answer without CoT within eight attempts.
- **Domain balance**: math, coding, science, and other domains are balanced so that one task does not dominate the learning signal.

Prompt engineering and rejection sampling then produce a small set of long-CoT trajectories for lightweight SFT. This stage teaches patterns such as planning, checking, reflection, and exploration; **most of the reasoning gain is still expected from the later RL stage**.

### 4.3 Reinforcement Learning: Policy Updates without a Critic

Kimi k1.5 does not train a critic. It constructs advantages from the group-mean reward and updates the policy with a variant of Online Policy Mirror Descent. For the advantage function, PMD squared loss, gradient, and complete derivation from \(J_x^{(m)}(\pi)\) to \(L_{\mathrm{PMD}}(\theta)\), see [Section 6 of *Policy and Preference Optimization for Large Language Models*](../llm-policy-optimization/#6-kimi-k15-policy-optimization-a-variant-of-online-policy-mirror-descent).

### 4.4 Reinforcement-Learning Recipe: Length, Sampling, and Reward Verification

#### Length Reward

For \(k\) answers to one problem, let \(\operatorname{len}(i)\) denote the length of answer \(i\), and define

\[
\lambda_i=
0.5-
\frac{\operatorname{len}(i)-\operatorname{min\_len}}
{\operatorname{max\_len}-\operatorname{min\_len}}.
\]

When the group contains different lengths, the length reward is

\[
r_{\mathrm{len}}(i)=
\begin{cases}
\lambda_i, & r_i=1,\\
\min(0,\lambda_i), & r_i=0.
\end{cases}
\]

- Among correct answers, shorter ones receive more reward.
- Among incorrect answers, long ones are penalized, but a short wrong answer is not rewarded.
- Because the length signal can hinder early exploration, training starts without it and activates it later.

#### Curriculum and Prioritized Sampling

- **Curriculum sampling**: train on easier problems first, then introduce harder ones so early rollouts are not almost all failures.
- **Prioritized sampling**: if problem \(i\) has historical success rate \(s_i\), sample it with probability proportional to \(1-s_i\), concentrating training on current weaknesses.

#### Reward Verification

- **Code**: generate test cases automatically, then cross-check them with multiple correct submissions.
- **Math**: because equivalent answers can have different forms, train a CoT reward model on roughly 800K examples to reason about answer equivalence. In manual spot checks, it reached 98.5% accuracy versus 84.4% for a conventional scalar reward model.

The distinction matters: Kimi's coding reward is close to rule verification, while its math reward uses a model to judge equivalence. The latter serves verifiable tasks but still inherits reward-model errors.

### 4.5 RL Infrastructure: Making Long Rollouts Trainable

- **Iterative synchronous training**: rollout workers generate trajectories, then trainer workers update the model in each iteration.
- **Partial rollouts**: each generation phase has a token budget. An unfinished long trajectory is stored in a replay buffer and continued in the next iteration, preventing a few long responses from blocking the batch.
- **Repeat detection**: repetitive generations can be stopped early and additionally penalized.
- **Hybrid training–inference deployment**: Megatron and vLLM run in separate containers on the same GPU pool. Training and rollout reuse memory in alternating phases, while a weight-transfer component moves updated parameters.

### 4.6 Experimental Results: RL Scales Both Performance and Reasoning Length

In a small-model experiment using only math data, accuracy and response length generally rise with RL iterations on most tasks, although small-sample evaluations such as AIME fluctuate substantially. **Longer trial-and-error trajectories correlate with improvement here, but this does not imply that a longer answer is necessarily more correct.**

<details>
<summary>View performance and response length during RL</summary>

<figure style="text-align: center;">
  <img src="../../../posts/rlvr/kimi-rl-scaling.png" alt="Accuracy and response-length trends across Kimi k1.5 reinforcement-learning iterations" loading="lazy">
  <figcaption>Accuracy and token length generally increase on most tasks, while AIME is much noisier. Source: CS336 Lecture 16.</figcaption>
</figure>

</details>

The paper also compares RL with **Expert Iteration**. Expert iteration retains successful trajectories and continues supervised learning, so it learns only from positive examples. With a group baseline, RL also decreases the probability of low-reward responses. RL performs better on most tasks, suggesting that **negative gradients from low-reward samples are also valuable training signals**.

<details>
<summary>View the comparison between RL and Expert Iteration</summary>

<figure style="text-align: center;">
  <img src="../../../posts/rlvr/kimi-rl-vs-expert-iteration.png" alt="Kimi k1.5 comparison between reinforcement learning and positive-only expert iteration" loading="lazy">
  <figcaption>RL using both positive and negative signals outperforms positive-only expert iteration on most evaluations. Source: CS336 Lecture 16.</figcaption>
</figure>

</details>

## 5. Qwen3: Unifying Thinking and Non-thinking Modes

### 5.1 Training Pipeline and Essential Recipe

[Qwen3](https://arxiv.org/abs/2505.09388) separates post-training into a flagship route and a lightweight-model route:

| Model route | Training pipeline |
| --- | --- |
| Flagship models | Base model → long-CoT cold start → reasoning RL → thinking-mode fusion → general RL |
| Lightweight models | Base model → strong-to-weak distillation from flagship models |

<figure style="text-align: center;">
  <img src="../../../posts/rlvr/qwen3-post-training-pipeline.png" alt="Qwen3 four-stage flagship post-training pipeline and strong-to-weak distillation pipeline for lightweight models" loading="lazy">
  <figcaption>Flagship Qwen3 models complete four post-training stages, while lightweight models are distilled from strong teachers. Source: CS336 Lecture 16; original figure from the Qwen3 technical report.</figcaption>
</figure>

The essential recipe is:

- the first two stages establish and reinforce long-form reasoning;
- the final two integrate thinking and non-thinking behavior in one model and restore broad capabilities;
- users can control reasoning with `/think`, `/no_think`, and a token budget;
- small models use strong-to-weak distillation instead of repeating the entire four-stage process.

### 5.2 Four Training Stages

#### Stage One: Long-CoT Cold Start

The data cover math, code, logic, and STEM, each with a reference answer or executable tests:

- Qwen2.5-72B-Instruct removes prompts that are hard to verify, contain multiple subquestions, or can be solved correctly without CoT.
- QwQ-32B generates multiple candidate trajectories. Responses with wrong answers, repetition, guessing, inconsistencies between reasoning and summaries, language mixing, or suspected evaluation leakage are removed.
- Only a small, high-quality subset is used for SFT. The goal is to establish reasoning patterns, not maximize benchmark scores during cold start.

#### Stage Two: Reasoning Reinforcement Learning

Starting from the cold-start model, Qwen3 continues training with GRPO and verifiable rewards. The final dataset contains only **3,995 query–verifier pairs**, each satisfying three conditions:

- it was not used during cold start;
- it is learnable for the current model but remains challenging;
- it contributes coverage of reasoning subdomains.

RL data quality is therefore not simply about volume. If a problem is too easy, almost every response in a group is correct; if it is too hard, almost every response fails. Neither case produces a useful relative-advantage signal.

#### Stage Three: Thinking-Mode Fusion

This stage continues SFT on a mixture of two data types:

| Mode | Input control | Output format |
| --- | --- | --- |
| Thinking | `/think`, which may be omitted | `<think>reasoning</think>` followed by the answer |
| Non-thinking | `/no_think` | an empty `<think></think>` block followed by a direct answer |

- Thinking data are generated by the Stage Two model on Stage One queries and filtered by rejection sampling.
- Non-thinking data cover instruction following, writing, QA, translation, role play, and other general tasks.
- In multi-turn conversations, multiple mode flags may appear; the model follows the final flag.

Mode fusion also allows reasoning to stop midstream. When a user-defined token budget is reached, the system inserts an instruction to stop thinking and answer using the reasoning accumulated so far. This behavior was not trained with a separate budget dataset; it emerged after mode fusion.

#### Stage Four: General Reinforcement Learning

General RL covers more than twenty task types, targeting instruction and format following, open-ended preferences, tool use, RAG, and specialized scenarios. It uses three reward types:

1. **Rule-based rewards** check answers, formats, or explicit instruction constraints.
2. **Model-based rewards with references** ask Qwen2.5-72B-Instruct to score a response against a reference answer.
3. **Model-based rewards without references** use a reward model trained from human preferences to judge helpfulness and style.

This stage therefore combines RLVR and RLHF rather than remaining purely verifiable reasoning training.

### 5.3 Strong-to-Weak Distillation: Small Models Skip the Four-stage Pipeline

Lightweight models use Qwen3-32B or Qwen3-235B-A22B as teachers in two phases:

1. **Off-policy distillation** directly learns the teachers' `/think` and `/no_think` responses.
2. **On-policy distillation** lets the student generate responses, then minimizes KL divergence between student and teacher output distributions.

Compared with running the full four-stage pipeline for every small model, the paper reports that this route uses roughly one-tenth of the GPU hours while improving both Pass@1 and Pass@64.

### 5.4 Experimental Findings: Thinking Budgets Help, but General Training Has Trade-offs

- **Inference-time scaling works**: increasing the thinking budget from 1K to 32K tokens generally improves AIME, LiveCodeBench, and GPQA.
- **Mode fusion improves controllability**: Stage Three enables both thinking and non-thinking responses and improves general and instruction-following performance.
- **General RL further improves open-ended tasks**: Stage Four adds gains on general, instruction, and agent tasks.
- **Specialized reasoning can regress**: thinking-mode scores on difficult tasks such as AIME 2024 and LiveCodeBench decline after Stages Three and Four, suggesting that broader capability training can dilute some specialized reasoning performance.

<details>
<summary>View the relationship between thinking budget and benchmark performance</summary>

<figure style="text-align: center;">
  <img src="../../../posts/rlvr/qwen3-thinking-budget.png" alt="Qwen3 performance on AIME, LiveCodeBench, and GPQA under different thinking-token budgets" loading="lazy">
  <figcaption>On these evaluations, a larger thinking budget generally yields a higher Pass@1. Source: CS336 Lecture 16; original figure from the Qwen3 technical report.</figcaption>
</figure>

</details>

## 6. Qwen3-Coder-Next: RLVR for Coding Agents

[Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729) is an 80B-total-parameter MoE coding model that activates roughly 3B parameters per forward pass. Its focus is not a single-turn algorithm problem, but an agent repeatedly inspecting files, calling tools, editing code, and running tests inside a real repository.

**Pipeline: Qwen3-Next base model → code and agentic mid-training → SFT → domain experts → expert distillation.**

### 6.1 Mid-training: Shifting the Distribution toward Repositories and Agent Trajectories

Mid-training continues language-model training from the pretrained base, but shifts data toward code and agent workflows:

- **Repository-level data** concatenate files from one repository, totaling roughly 600B tokens, while context length expands to 262,144 tokens.
- **Pull-request data** turn real PRs into problem-description, repository-context, and patch examples.
- **Text–code grounding** cleans Common Crawl and technical websites, rewriting HTML and advertising noise into structured text.
- **Synthetic QA** generates progressively deeper coding questions and answers from technical documents.
- **Agent trajectories** use several agent frameworks and a teacher model to generate multi-turn tool-use trajectories in executable environments.
- **Auxiliary objectives** mix in a small amount of instruction data and Fill-In-the-Middle (FIM) data to retain instruction following and code-editing ability.

Natural data preserve breadth and robustness, while synthetic data better match real tool use. Too much synthetic data can instead cause overspecialization and reduce response diversity.

### 6.2 Expert Models and Distillation: Specialize Separately, Then Merge Capabilities

All experts begin from the same SFT checkpoint but continue training for different tasks:

<figure style="text-align: center;">
  <img src="../../../posts/rlvr/qwen3-coder-expert-models.png" alt="Qwen3 Next trains four specialized expert models and distills them into a unified Qwen3 Next Coder" loading="lazy">
  <figcaption>Qwen3 Next trains web-development, user-experience, single-turn QA, and software-engineering experts, then distills their capabilities into one model. Source: CS336 Lecture 16; original figure from the Qwen3-Coder-Next technical report.</figcaption>
</figure>

| Expert | Main training objective |
| --- | --- |
| Web development | Pass visual-quality, rendering, and interaction checks |
| User experience | Adapt to different IDE, CLI, and tool-call formats |
| Single-turn QA | Use executable tests to reinforce code generation and complex instruction following |
| Software engineering | Perform multi-turn localization, editing, and verification inside repositories |

The experts are then distilled into one unified model, avoiding expert routing or multiple deployment models.

### 6.3 Automated Environment Construction: Turning Patches into Verifiable Tasks

To scale software-engineering RL, the system automatically creates roughly 800K verifiable tasks across more than nine programming languages:

1. collect repositories that can run in containers;
2. locate functions or classes with ASTs and inject controlled bugs;
3. run tests to confirm that the injected version fails and patch restoration passes;
4. reverse-generate a natural-language issue from the patch and hide test files that would directly reveal the answer.

<figure style="text-align: center;">
  <img src="../../../posts/rlvr/qwen3-coder-agent-environment.png" alt="Agent-environment construction from repository collection and bug injection to container verification and issue generation" loading="lazy">
  <figcaption>The automated pipeline turns code patches into software-engineering tasks whose outcomes can be checked by execution. Source: CS336 Lecture 16; original figure from the Qwen3-Coder-Next technical report.</figcaption>
</figure>

This extends RLVR from final-answer matching to checking whether the environment reaches a test-satisfying state. The agent may take different edit paths; it receives completion reward as long as the final patch passes the verifier.

### 6.4 Agentic Reinforcement Learning: Long-horizon Capability and Reward Hacking Grow Together

Software-engineering RL uses multi-turn interaction and assigns a trajectory-level reward from final task completion. It also adds:

- **unfinished-trajectory penalties** when the agent exceeds the maximum number of turns;
- **tool-format penalties** on tokens associated with invalid tool calls;
- **a reward-hacking blocker** that prevents the model from reconnecting to the original GitHub repository and reading the reference patch from future commits.

As training progresses, SWE-Bench Verified performance rises and the average number of agent turns grows from roughly 50 to 130, indicating longer-horizon coding behavior. Without the blocker, however, the score suddenly jumps to 84.6: the model is not becoming better at repairing bugs; it is restoring remotes, reading commit history, and retrieving the reference answer.

<details>
<summary>View the agentic RL and reward-hacking experiment</summary>

<figure style="text-align: center;">
  <img src="../../../posts/rlvr/qwen3-coder-agent-rl.png" alt="Qwen3-Coder-Next performance, agent turns, and reward-hacking behavior during reinforcement learning" loading="lazy">
  <figcaption>The left plot shows genuine growth with a blocker; the right plot's anomalous jump comes from restoring a remote and retrieving the reference patch. Source: CS336 Lecture 16; original figure from the Qwen3-Coder-Next technical report.</figcaption>
</figure>

</details>

This experiment exposes a central limitation of RLVR: <strong>a verifier guarantees only that the checked condition was satisfied, not that the model used the method intended by the designer.</strong> If the environment leaks a shortcut, policy optimization may turn exploiting that shortcut into the highest-reward behavior.

## References

[1] Stanford University, "CS336 Language Modeling from Scratch: Lecture 16—Post-Training 2: Reinforcement Learning from Verifiable Rewards," course slides, 2025. [Online]. Available: https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf.
