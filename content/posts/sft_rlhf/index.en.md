---
title: "SFT and RLHF"
date: 2025-09-29T11:30:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "Large Language Model"
    subseries: "Fine-tuning"
categories: ["Deep Learning Skills", "Large Language Model", "Reinforcement Learning"]
tags: ["SFT", "Reinforcement Learning", "RLHF"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "Introduction of SFT and RLHF."
# canonicalURL: "https://canonical.url/to/page"
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
    image: "cover.jpg" # image path/url
    alt: "sft_rlhf" # alt text
    caption: "SFT and RLHF" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

- **Pre-training**: gives a language model general capabilities such as text continuation and knowledge use, but does not guarantee reliable instruction following.
- **Post-training**: uses data that better represents the desired behaviors, making the model more controllable in instruction following, response style, and safety boundaries.
- **Disclosure**:
  - **Early research**: described annotation guidelines and training workflows in greater detail, such as [Stiennon et al.'s early RLHF work](https://arxiv.org/abs/2009.01325) and [Bai et al.'s safety-alignment work](https://arxiv.org/abs/2204.05862).
  - **Open-source models**: often include distilled data, but their release notes may not disclose the data sources or exact recipe.
  - **Closed-source models**: post-training data and workflows are usually core competitive assets, so less information is public.
- **Scope**: public papers and open-source implementations explain the basic mechanisms, but do not necessarily reproduce the complete post-training pipeline of a production model.

## 1. Post-Training: Three Stages

> Diagram source: [InstructGPT](https://arxiv.org/abs/2203.02155).

![stages](stage.png)

1. Collect data and train a **supervised** policy.
  - Sample a prompt from the prompt dataset.
  - Annotators label the desired output.
  - Use the labeled data to perform supervised fine-tuning of the LLM.
2. Collect **comparison data** and train a **reward model**.
  - Sample a prompt and multiple model outputs.
  - Annotators rank these outputs from “best” to “worst”.
  - Use the ranking data to train the reward model.
3. With the **trained reward model**, optimize the policy using **reinforcement learning**.
  - Sample a new prompt from the dataset.
  - Generate an output with the current policy.
  - The reward model scores the output (Reward).
  - Update the policy using PPO (or other RL methods) based on the reward.

---

## 2. Building an SFT Dataset

SFT has two components: training data and a training method. This section first looks at what instruction data contains and which data properties affect model behavior.

### 2.1. The two components of SFT

- **Training data**: defines the inputs, outputs, and behaviors that the model should imitate.
- **Training method**: updates the model parameters with this data so that the model learns the target behaviors.

### 2.2. Open instruction datasets: evolution, examples, and differences

Open SFT datasets have broadly evolved from task-oriented instructions, synthetic instructions, and multi-turn conversations to tool calling and agentic tasks:

![The evolution of open SFT datasets from task fine-tuning and synthetic instructions to conversations and tool use](sft-data-progression.png)

FLAN → Self-Instruct → Alpaca → ShareGPT/Vicuna → OpenAssistant → WizardLM → Tulu3 → Nemotron → tool use, and more.

**Representative examples**

**FLAN** includes traditional NLP tasks such as email subject generation, text classification, long-form summarization, and structured data-to-text generation.

<figure>
  <img src="flan-examples.png" alt="Complete FLAN examples of email subject generation, text classification, summarization, and structured data-to-text generation">
  <figcaption>Complete FLAN examples of email subject generation, text classification, summarization, and structured data-to-text generation. Source: CS336 Lecture 15.</figcaption>
</figure>

**Alpaca** uses short, single-turn instructions covering common-sense advice, concept explanations, and code generation.

<figure>
  <img src="alpaca-examples.png" alt="Complete Alpaca examples of health advice, algorithm explanation, and code generation for computing a list average">
  <figcaption>Complete Alpaca examples of health advice, algorithm explanation, and code generation for computing a list average. Source: CS336 Lecture 15.</figcaption>
</figure>

**OpenAssistant** responses are typically longer and more detailed, and may involve complex knowledge and references.

<figure>
  <img src="openassistant-examples.png" alt="Complete OpenAssistant examples of an economics explanation and science project suggestions for children">
  <figcaption>Complete OpenAssistant examples of an economics explanation and science project suggestions for children. Source: CS336 Lecture 15.</figcaption>
</figure>

**Nemotron-SFT-OpenCode-v1** extends instruction data to task planning, structured messages, and tool calling.

<figure>
  <img src="nemotron-tool-use-examples.png" alt="Complete Nemotron-SFT-OpenCode-v1 examples of task planning and tool calling">
  <figcaption>Complete Nemotron-SFT-OpenCode-v1 examples of task planning and tool calling. Source: CS336 Lecture 15.</figcaption>
</figure>

**Key differences among instruction datasets**

- **Conversational style**: early datasets such as FLAN resemble traditional NLP tasks; later datasets increasingly emphasize natural conversation.
- **Response format**: datasets make different choices about answer length, bullet points, and writing style. Models imitate these choices, while long responses are also harder for humans to annotate consistently.
- **Knowledge and references**: detailed facts, complex knowledge, and citations can deepen answers, but inaccurate citations or superficial imitation of citation formats may increase hallucinations.
- **Task scope**: instruction data has expanded from text-only question answering to tool calling and agentic tasks.
- **Scale and safety**: dataset size, long-tail coverage, and the proportion of safety examples are not visible from a few samples, but can substantially affect model behavior.

<figure>
  <img src="instrction_dataset.png" alt="Comparison of dataset size, conversation turns, and input-output lengths across instruction datasets">
  <figcaption>Instruction datasets differ substantially in size, average conversation turns, and input-output lengths. Source: Wang et al., 2023.</figcaption>
</figure>

**Response style: preference scores do not equal capability gains**

- **At the data level**: response lengths vary widely across instruction datasets, and models imitate these stylistic features during SFT.
- **In preference evaluation**: both human and GPT evaluators tend to prefer lists and longer answers, so preference scores are sensitive to presentation style. [Experiments by Dubois et al.](https://arxiv.org/abs/2305.14387) show a strong length effect.
- **In capability evaluation**: longer and more detailed answers do not necessarily improve factuality, reasoning, coding, or other benchmark results. A [systematic evaluation by Wang et al.](https://arxiv.org/abs/2306.04751) also found that preference-based evaluations did not fully reflect the capability differences exposed by benchmarks.

<figure>
  <img src="preference-length-bias.png" alt="Human and GPT evaluator preferences for lists and longer responses">
  <figcaption>Human and GPT evaluators commonly prefer lists and longer responses, so preference scores can contain a strong style component. Source: Dubois et al., 2023.</figcaption>
</figure>

> <strong>Higher preference does not imply stronger capability.</strong> SFT models should be evaluated with both preference judgments and capability benchmarks for factuality, reasoning, coding, and related skills.

<details>
  <summary>View the benchmark comparison across instruction datasets</summary>

  <figure>
    <img src="instruction-benchmark-comparison.png" alt="Performance of instruction datasets on factuality, reasoning, multilinguality, coding, and open-ended evaluation">
    <figcaption>Different instruction datasets excel at different capabilities; a high open-ended preference score does not guarantee corresponding gains on other benchmarks. Source: Wang et al., 2023.</figcaption>
  </figure>
</details>

---

### 2.3. Knowledge extraction and alignment: SFT works best for eliciting existing knowledge

SFT examples containing complex knowledge or references teach the model two things at once:

1. **Content**: the association between a question and relevant facts.
2. **Behavior**: when to provide a detailed explanation or citations.

- **Citation risk**: a model may learn the surface format of citations without learning to verify them.
- **Unknown-fact risk**: [experiments by Gekhman et al.](https://arxiv.org/abs/2405.05904) found that models learn facts unknown during pre-training more slowly, while continued fitting can reduce development-set performance.

> <strong>Practical conclusion: SFT is better suited to eliciting and organizing existing capabilities than serving as a reliable knowledge store.</strong>

“Fine-tuning on unknown facts increases hallucination” is an empirical observation under specific experimental conditions, not an unconditional theorem. In principle, correctness-based feedback may be more suitable than imitating a single reference answer.

<details>
  <summary>View the explanation and experiment on unknown-fact fine-tuning</summary>

  <figure>
    <img src="knowledge-extraction-hallucination.png" alt="Behavior-cloning explanation and experimental results for hallucination caused by fine-tuning on unknown facts">
    <figcaption>The left panel illustrates how behavior cloning may teach a model to guess unknown facts; the right panel shows slower fitting of unknown facts and declining development accuracy after overfitting. Sources: <a href="https://news.berkeley.edu/2023/04/24/berkeley-talks-transcript-chatgpt-developer-john-schulman/">Schulman, 2023</a>; <a href="https://arxiv.org/abs/2405.05904">Gekhman et al., 2024</a>.</figcaption>
  </figure>
</details>

---

### 2.4. Safety supervised fine-tuning: a small amount of targeted data can substantially change behavior

Widely deployed models must be useful while reducing misinformation, scams and spam, and direct compliance with harmful instructions.

- **Limited public information**: [Llama 2](https://arxiv.org/abs/2307.09288) switched to RLHF after collecting only a few thousand safety demonstrations; modern models rarely disclose their full safety-SFT data and pipeline.
- **Open practice**: [Tülu 3](https://arxiv.org/abs/2411.15124) provides a comparatively detailed open pipeline that includes CoCoNot (10,983 examples), WildJailbreak (50,000), and WildGuardMix (50,000) for safety and non-compliance.
- **Scenario sources**: risk scenarios can be extracted from real user interactions and paired with suitable safe responses. [WildChat](https://arxiv.org/abs/2405.01470) contains one million real ChatGPT conversations spanning multiple languages, potentially harmful uses, and jailbreak behavior.

<details>
  <summary>View examples of open safety data and real user scenarios</summary>

  <figure>
    <img src="tulu3-safety-data.png" alt="Composition and scale of the Tülu 3 safety and non-compliance datasets">
    <figcaption>Tülu 3's safety and non-compliance data includes CoCoNot, WildJailbreak, and WildGuardMix.</figcaption>
  </figure>

  <figure>
    <img src="safety-scenarios-from-users.png" alt="Examples of extracting safety scenarios and jailbreak strategies from real WildChat interactions">
    <figcaption>Real user logs expose refusal boundaries and provide concrete harmful-query and jailbreak scenarios. Sources: WildChat and Tülu 3.</figcaption>
  </figure>
</details>

[Experiments in Safety-Tuned LLaMAs](https://arxiv.org/abs/2309.07875) further showed that, under their training setup, adding about 500 Alpaca-style safety examples substantially improved results across four safety evaluations.

<figure>
  <img src="safety-small-data-effect.png" alt="Effect of different amounts of safety data on scores across four safety evaluation datasets">
  <figcaption>In this experiment, a small amount of safety data sharply reduced harmful-output scores, after which the gains gradually tapered off.</figcaption>
</figure>

> <strong>Practical conclusion: safety SFT depends more on targeted scenarios and coverage than on sample count alone.</strong> However, too much homogeneous safety data can cause exaggerated safety behavior, making the model refuse benign prompts that only superficially resemble harmful ones.

---

### 2.5. Summary of practical tips for constructing SFT data

1. SFT works best when the base model already has certain capabilities, and the data helps “extract” them. If you try to use SFT to “add” behaviors the model fundamentally lacks, results are often poor.
2. Not all factually correct data improves performance. Even high-quality factual data can disrupt the model’s existing distribution/alignment and degrade performance.
3. Some data types (e.g., safety, instruction-following, style) can yield large gains even in small amounts. However, improving long-tail behaviors (broad coverage, sparse scenarios) typically requires much more data.

---

### 2.6. SFT training: from basic gradient descent to integration across training stages

#### The basic training loop

SFT still uses ordinary gradient-descent training. A standard training loop is sufficient for many academic experiments; training efficiency and stability become the main concerns only as data and compute scale up.

#### Instruction tuning during pretraining

1. Pretrain on web data or a pretraining corpus.
2. Mix instruction-tuning data into pretraining.
3. Do an additional short instruction-tuning stage.

#### Midtraining and two-phase training

![minicpm](./minicpm.png)

[MiniCPM](https://arxiv.org/abs/2404.06395) adopts this recipe. Similar approaches also appear to be common among LLM companies, although public details remain limited:
- In the Stable stage, train on a pure pretraining dataset (left in the figure).
- In the Decay stage, train on a mixture of pretraining + instruction-tuning data (right in the figure).

---

## 3. RLHF (Reinforcement Learning with Human Feedback)

### 3.1. From imitation to optimization

The central difference between SFT and RLHF is a shift in the training objective: from **imitating reference answers** to **maximizing a measurable reward**.

#### Imitation: SFT fits the reference-answer distribution

Given an input \(x\), SFT adjusts the model distribution \(\hat{p}(y\mid x)\) to approximate the reference-answer distribution \(p^*(y\mid x)\):

\[
\hat{p}(y\mid x) \approx p^*(y\mid x)
\]

SFT therefore requires answer samples from a reference policy, such as human-written target responses.

#### Optimization: RLHF searches for a higher-reward policy

Rather than approximating a reference-answer distribution, RLHF searches for an output distribution that receives higher reward:

\[
\hat{p}=\arg\max_p\mathbb{E}_{y\sim p(\cdot\mid x)}[R(y,x)]
\]

Here, \(R(y,x)\) is a measurable reward. From this perspective, the language model is a **policy** to be optimized.

| Dimension | SFT: imitation | RLHF: optimization |
| --- | --- | --- |
| Training objective | Fit a reference-answer distribution | Maximize measurable reward |
| Required signal | Reference-answer samples | Reward signals for output quality |
| Role of the model | Generative model | Policy |

> <strong>Core shift: SFT learns how reference answers are generated, while RLHF pushes the model toward answers with higher reward.</strong>

---

### 3.2. Why RLHF is needed: human preferences are not the same as human demonstrations

SFT relies on human-written reference answers, but this form of supervision has two limitations:

1. **Demonstrations are costly**: annotators must write a complete, high-quality answer from scratch; preference annotation usually requires only comparing candidate answers and selecting the better one.
2. **Generation–Value Gap (G–V Gap)**: what people write is not necessarily what they prefer when acting as evaluators.

#### Generation–Value Gap: writing behavior and evaluation criteria can differ

- **Generation (G)**: the answer an annotator writes in response to a prompt.
- **Value (V)**: the answer the annotator prefers when comparing alternatives.

When these differ, SFT can imitate the annotator's writing behavior without directly learning their actual preferences.

<details>
  <summary>View the news-summarization experiment by Zhang et al.</summary>

  [The news-summarization experiment by Zhang et al.](https://aclanthology.org/2024.tacl-1.3/) asked six annotators to compare the same 100 pairs of summaries:

  - **The aggregate preference was nearly even**: 50.4% for freelance-writer summaries and 49.6% for Instruct Davinci summaries.
  - **Individual preferences varied substantially**: annotators disagreed on the preferred style, with overall agreement of only \(\alpha=0.07\).
  - **Writing and preference could point in opposite directions**: one annotator wrote abstractive summaries but preferred the more extractive Instruct Davinci summaries 57% of the time.

  <figure>
    <img src="generation-value-gap.png" alt="Preferences of six annotators between freelance-writer and Instruct Davinci summaries">
    <figcaption>Aggregate preference was almost evenly split between human and model summaries, while individual annotators differed substantially. Source: Zhang et al., 2024.</figcaption>
  </figure>
</details>

> <strong>RLHF directly uses feedback about which answer is better instead of treating one human demonstration as the only correct target.</strong>

---

### 3.3. Preference data: annotation format, guidelines, and feedback sources jointly determine quality

#### Pairwise preference annotation: select the better of two responses

The standard setup generates two responses to the same prompt and asks an annotator to compare them under a shared rubric:

1. Show the prompt, response A, and response B together.
2. Select the better response, optionally distinguishing “better” from “slightly better.”
3. Store the result as \((x,y^+,y^-)\), where \(y^+\) is preferred and \(y^-\) is dispreferred.

<figure>
  <img src="pairwise-feedback-interface.png" alt="Interface showing two model responses and asking an annotator to select the preferred one">
  <figcaption>The annotator compares two responses under the same guidelines and selects response 1 or response 2 together with preference strength.</figcaption>
</figure>

#### Annotation guidelines: define goals, priorities, and boundaries

[The InstructGPT annotation guidelines](https://arxiv.org/abs/2203.02155) summarize response quality using three goals:

- **Helpful**: relevant, clear, and sufficiently informative for the task.
- **Truthful**: factually correct and non-misleading, with uncertainty stated when appropriate.
- **Harmless**: avoids physical, psychological, and social harm.

When these goals conflict, truthfulness and harmlessness generally take priority over simply satisfying the request. Guidelines should also explain when to skip an example, how to handle ambiguous cases, and how to prioritize different quality dimensions.

> [An early Bard annotation rubric](https://assets.bwbx.io/documents/users/iqjWHBFdfxIU/rqKqEqbXBnDI/v0) asks annotators to judge correctness, relevance, writing quality, and multiple rating levels at once. Too many overlapping dimensions increase cognitive load and make judgments less consistent.

#### Feedback sources: human annotation and AI feedback

**Option 1: human annotation**. Ask the model to generate \(N\) candidate responses, then have annotators compare or rank them.

- **Annotation quality**: judgments may be low-quality or incorrect, and some labels may themselves be generated with another language model.
- **Annotator distribution**: annotators' regional, cultural, and professional backgrounds affect their criteria and can therefore shape model behavior.
- **Different priorities**: some annotators focus on formatting, while others emphasize factuality, reasoning, or completeness.

**Option 2: AI feedback**. Use a stronger language model to compare or rank the \(N\) candidates. This is commonly called **AI Feedback**.

AI feedback is now part of several open post-training pipelines:

| Example | How AI feedback is used |
| --- | --- |
| [UltraFeedback](https://arxiv.org/abs/2310.01377) | Uses GPT-4 to provide more than one million multidimensional feedback records for 250,000 conversations |
| [Zephyr-7B](https://arxiv.org/abs/2310.16944) | Applies Distilled Direct Preference Optimization (dDPO) to teacher-ranked preference data without additional human annotation |
| [Tülu 3](https://arxiv.org/abs/2411.15124) | Generates candidates with multiple models, then uses a stronger model to select preferred answers along dimensions such as instruction following, truthfulness, and honesty |

**Self-training: the model both generates data and participates in supervision.** [Constitutional AI (CAI)](https://arxiv.org/abs/2212.08073) divides this process into two stages:

1. **Supervised-learning stage**: the model answers red-team prompts, then generates critiques and revisions under a set of principles; the revised answers are used for fine-tuning.
2. **AI-feedback reinforcement-learning stage**: the model generates pairs of answers, AI selects preferences under the principles, and those preferences train a preference model used to optimize the policy. This is called **Reinforcement Learning from AI Feedback (RLAIF)**.

<figure>
  <img src="constitutional-ai-self-training.png" alt="Two-stage Constitutional AI process from self-critique and revision to RLAIF training">
  <figcaption>Constitutional AI first constructs supervised data through self-critique and revision, then uses AI preferences to train a preference model and the final policy. Source: Bai et al., 2022.</figcaption>
</figure>

The preference-data format can remain unchanged whether feedback comes from a human or a model; what changes is who produces the judgment and which biases it may contain.

> <strong>Preference data is not objective ground truth: it encodes the annotation rubric, feedback source, and associated judgment biases.</strong>

---

### 3.4. From PPO to Direct Preference Optimization: Is Online RL Necessary?

PPO-based RLHF requires the current policy to generate responses, a reward model to score them, a critic to estimate advantages, and PPO to update the policy. Several simpler alternatives remove parts of this online reinforcement-learning pipeline:

- **Control tokens**: prepend `[GOOD]` to preferred responses and `[BAD]` to dispreferred responses, then perform SFT on the response pairs.
- **Preferred responses only**: discard dispreferred responses and use preferred responses directly as SFT targets.
- **Reward-model filtering**: generate candidate responses, let the reward model select strong responses, and use those responses for supervised training.
- **Best-of-N**: generate many candidates—for example, \(N=1024\)—and keep only the response with the highest reward.

These approaches simplify training, but they may discard useful preference information or write reward-model errors back into the training data.

#### PPO and DPO: Two Main Routes

- **PPO-based RLHF**: a reward model scores new responses from the current policy, and PPO performs online updates from rewards and advantages.
- **DPO**: preferred–dispreferred response pairs update the policy directly, without an explicit reward model or online rollouts.

| Dimension | PPO-based RLHF | DPO |
| --- | --- | --- |
| How feedback is used | A reward model scores new responses | The policy learns directly from pairs |
| Online generation during training | Required | Not required |
| Additional models | Reward model and critic | Usually none |
| Main characteristic | Can continue exploring responses from the current policy | Pipeline resembles supervised fine-tuning |

Both approaches constrain the policy relative to a reference model. PPO applies an explicit reference KL during policy optimization, whereas DPO incorporates reference-relative probability ratios into its preference loss.

> For the core objectives of TRPO, PPO, DPO, SimPO, and length-normalized DPO, see [Policy and Preference Optimization for Large Language Models](../llm-policy-optimization/). For PPO's general training procedure, see [Proximal Policy Optimization (PPO)](../ppo/).

#### DPO and Expert Iteration: Refreshing Training Data with New Models

DPO can train on a fixed preference dataset or participate in multiple rounds of expert iteration:

1. Generate \(K\) candidate responses for each collected prompt.
2. Use rejection sampling with a reward model and convert high-scoring responses into a new round of SFT data.
3. Use pairwise preferences to train the reward model and DPO model, while task-specific SFT data supplements different capabilities.
4. Select the strongest model from previous rounds and use it to regenerate candidates for the next round.

<figure>
  <img src="dpo-expert-iteration.png" alt="A multi-round post-training pipeline combining DPO with expert iteration">
  <figcaption>DPO can be combined with rejection sampling, task-specific SFT, and repeated model iteration rather than being limited to one fixed dataset. Source: Tülu 3.</figcaption>
</figure>

Thus, “DPO requires no online rollouts” describes the objective within one DPO round. A post-training system that continually refreshes its data with newer models remains iterative as a whole.

### 3.5. Comparing Algorithms: Conclusions Depend on the Experimental Setup

Neither “PPO always outperforms DPO” nor “DPO always outperforms PPO” is a general conclusion.

<details>
<summary>Expand: which experimental factors can change the algorithm ranking?</summary>

- [Ivison et al.](https://arxiv.org/abs/2406.09279) separate preference data, learning algorithm, reward model, and policy-training prompts in controlled experiments. Data quality has the largest effect; the algorithm is only one factor.
- [Tülu 3](https://arxiv.org/abs/2411.15124) also shows that outcomes can change with learning rate, DPO normalization, margin coefficient, number of epochs, and batch size.
- Data sources, candidate-response distributions, evaluators, and length preferences can all change the ranking between algorithms.

</details>

> An algorithm comparison should report the data, reward or preference source, sampling strategy, hyperparameters, and evaluation method. Comparing algorithm names alone has little explanatory power.

### 3.6. RLHF Risks: Reward Overoptimization and Mode Collapse

#### Reward Overoptimization: The Model Learns to Please the Scorer

Reward overoptimization does not mean that “a high reward is bad.” It means <strong>optimizing an imperfect reward model too aggressively</strong>:

- **Proxy reward**: the score produced by the reward model used for training.
- **Independent quality**: response quality judged by a separate group of humans or an independent evaluator.

Early in training, both usually improve. With further optimization, the policy may discover biases in the reward model and produce responses that score well but are actually worse. Proxy reward then keeps rising while independent quality declines.

In the figure, the horizontal axis is proxy reward and the vertical axis is independent evaluation win rate. Movement toward the upper right represents genuine improvement; continued movement to the right while the curve turns downward indicates reward overoptimization.

<figure>
  <img src="reward-overoptimization.png" alt="The relationship between proxy reward and evaluation win rate under different preference sources">
  <figcaption>With human preferences and noisy simulated preferences, evaluation win rate falls after reaching a peak; a single low-noise GPT-4 simulator does not reproduce this pattern. Source: AlpacaFarm.</figcaption>
</figure>

[AlpacaFarm](https://arxiv.org/abs/2305.14387) compares expert iteration, Best-of-N, and PPO:

- **Human preferences**: as proxy reward keeps increasing, evaluation win rate first rises and then falls.
- **Noisy model preferences**: a similar overoptimization curve appears.
- **A single low-noise model preference source**: no clear decline appears in this experiment, which may lead researchers to underestimate overoptimization under real feedback.

Therefore, the reward model's training score is not sufficient; response quality must also be checked on an independent evaluation set. This risk is not unique to PPO: direct preference optimization can also degrade when trained for too long.

#### Mode Collapse: Reward Optimization Can Reduce Diversity and Calibration

Repeatedly pursuing high-reward modes can make the output distribution overly concentrated:

- **Lower entropy**: the model produces a small set of high-reward patterns more often, reducing response diversity.
- **Miscalibration**: token probabilities after post-training need not match empirical correctness, so high confidence does not necessarily imply greater reliability.
- **Incomplete evaluation**: a single win rate or reward score may miss degradation in diversity, entropy, and calibration.

Practical training should jointly monitor reward, independent quality evaluations, KL divergence, policy entropy, response diversity, and calibration error. KL or entropy regularization can mitigate these problems but cannot guarantee that mode collapse is avoided.

### 3.7. Practical RLHF Takeaways

- **Feedback data are difficult too**: annotation rules, human or AI evaluators, and candidate-response distributions all act as confounding factors.
- **Training is more complex than SFT**: PPO in particular must coordinate the policy, reference model, reward model, and critic.
- **Stronger optimization is not always better**: monitor overoptimization, mode collapse, and actual task quality in addition to proxy reward.
---

## References

[1] Stanford University, "CS336 Language Modeling from Scratch: Lecture 15—RLHF & Alignment," course slides, 2025. [Online]. Available: https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf.
