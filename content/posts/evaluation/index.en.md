---
title: "Large Language Model Evaluation"
date: 2026-08-22T11:12:03+08:00
series:
  main: "Large Language Model"
  subseries: "Evaluation"
categories: ["Large Language Model", "Evaluation"]
tags: ["Evaluation", "Benchmark"]
author: "CSPaulia"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "Study notes for CS336 Lecture 12 on large language model evaluation."
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
    image: "artificial-analysis.png"
    alt: "Artificial Analysis model intelligence leaderboard"
    caption: "Artificial Analysis model intelligence leaderboard."
    relative: true
    hidden: false
    hiddenInList: false
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes"
    appendFilePath: true
---

Evaluation asks a simple question: given a fixed model, how good is it?

The answer is not simple. Inputs, prompts, tools, metrics, and costs can all change the result.

Perplexity, knowledge, reasoning, instruction-following, and safety benchmarks reveal only different aspects of a model.

There is no single correct evaluation. Start from the goal, define the rules, and inspect individual examples and predictions.

## 1. The Core of Evaluation: Turning “Good” into a Concrete Metric

### 1.1 The Procedure Is Simple; Choosing the Metric Is Not

Evaluation appears to have only three steps:

1. Define a set of prompts.
2. Send them to a model and collect its responses.
3. Compute accuracy.

The difficult part is deciding what a “good model” means. Evaluation criteria guide model development, so the core challenge is:

> <strong>Abstract construct → concrete metric</strong>

### 1.2 Benchmark Performance: Measuring Preselected Capabilities

One definition is that a model is better when it scores higher on benchmarks. Artificial Analysis aggregates several evaluations into an intelligence index, making models easier to compare.

<figure>
  <img src="../../../posts/evaluation/artificial-analysis.png" alt="Artificial Analysis Intelligence Index model leaderboard" loading="lazy">
  <figcaption>An aggregate index compresses several benchmarks into one score. Source: <a href="https://artificialanalysis.ai/">Artificial Analysis</a>.</figcaption>
</figure>


This metric is clear and reproducible, but its conclusion depends on the selected tasks and aggregation method. It cannot capture all of a model's value.

### 1.3 Capability and Cost: Deployment Value Depends on Efficiency

When two models have similar capabilities, the cheaper model is often more practical. “Good” can therefore mean achieving stronger capabilities under a given operating cost.

<figure>
  <img src="../../../posts/evaluation/artificial-analysis-cost.png" alt="Relationship between model intelligence index and operating cost" loading="lazy">
  <figcaption>The upper-left region contains models with higher capability and lower cost. Source: <a href="https://artificialanalysis.ai/">Artificial Analysis</a>.</figcaption>
</figure>

### 1.4 Human Preference: Comparing the Experience of Responses

Another definition is that a model is better when users prefer its responses. Arena AI builds rankings from human preferences, capturing qualities such as style and helpfulness that are difficult to score automatically.

<figure>
  <img src="../../../posts/evaluation/lmarena-leaderboard.png" alt="Arena AI model leaderboard based on human preferences" loading="lazy">
  <figcaption>Human preference provides a quality signal distinct from fixed benchmarks. Source: <a href="https://arena.ai/leaderboard">Arena AI</a>.</figcaption>
</figure>

However, preference results depend on the participating users, prompt distribution, and response style.

### 1.5 Usage and Payment: Measuring Practical Value Through Real Choices

Continued use and willingness to pay also suggest that a model provides practical value. OpenRouter reports adoption through model token usage.

<figure>
  <img src="../../../posts/evaluation/openrouter.png" alt="OpenRouter model ranking by token usage" loading="lazy">
  <figcaption>Real usage reflects model adoption. Source: <a href="https://openrouter.ai/rankings">OpenRouter Rankings</a>.</figcaption>
</figure>

Usage is also affected by price, free access, availability, and platform recommendations, so it is not equivalent to capability.

## 2. Evaluation Methods

### 2.1 Perplexity: Measuring the Probability Assigned to Data

#### Definition of Perplexity: Higher Data Probability Produces a Lower Score

A language model is a probability distribution \(p(x)\) over token sequences. For a dataset \(D\) containing \(N\) tokens, perplexity (PPL) is:

\[
\begin{aligned}
\operatorname{PPL}(D)
&= \left(\frac{1}{p(D)}\right)^{1/N} \\
&= \exp\left(-\frac{1}{N}\log p(D)\right).
\end{aligned}
\]

The more probability the model assigns to the data, the lower its PPL. Pretraining reduces PPL on the training set, while traditional language modeling research also reports test-set PPL.

For the complete computation procedure, valid-token handling, and comparability conditions, see [Language Model Evaluation Metrics: Perplexity (PPL)](../metric/#perplexity-ppl). This section focuses on how PPL is used in model evaluation and where it falls short.

#### In-Distribution Evaluation: Training and Test Data Share a Source

Classic datasets include:

| Dataset | Text source |
| --- | --- |
| Penn Treebank (PTB) | The Wall Street Journal |
| WikiText-103 | Wikipedia |
| One Billion Word Benchmark (1BW) | EuroParl, United Nations, and news text from WMT11 |

The classic paradigm trains on a dataset's training split and evaluates on its test split. This is <strong>in-distribution evaluation</strong>.

> Convolutional neural networks (CNNs) and long short-term memory (LSTM) networks reduced PPL on 1BW from 51.3 to 30.0. [Jozefowicz et al., 2016](https://arxiv.org/abs/1602.02410)

#### GPT-2 Zero-Shot Evaluation: Distribution Shift Affects Perplexity

GPT-2 was trained on the 40 GB WebText corpus, collected from web pages linked by Reddit posts, and then evaluated zero-shot on standard datasets. This is <strong>out-of-distribution evaluation</strong>.

<figure>
  <img src="../../../posts/evaluation/gpt2-perplexity.png" alt="Zero-shot results of different GPT-2 sizes across language modeling datasets" loading="lazy">
  <figcaption>GPT-2 zero-shot language modeling results; bold numbers outperform the previous state of the art. Source: <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">GPT-2 technical report</a>.</figcaption>
</figure>

Transfer helps more on small datasets such as PTB, but GPT-2 falls behind specially trained models on a large dataset such as 1BW. Out-of-distribution PPL is therefore also affected by the gap between the training and test distributions.

#### Perplexity and General Capability: An Idealized Belief Is Not a Practical Theorem

Let \(t\) be the true data distribution and \(p\) the model distribution. Cross-entropy satisfies:

\[
H(t,p) = H(t) + D_{\mathrm{KL}}(t\Vert p).
\]

Cross-entropy reaches \(H(t)\) if and only if \(p=t\), giving a PPL of \(\exp(H(t))\). If the model truly recovered the data distribution, it would also represent \(p(\text{solution}\mid\text{problem})\).

This motivates the intuition that continually reducing PPL will eventually produce general intelligence.

#### Conditional Perplexity: Evaluate Only the Response

Ordinary PPL evaluates every token. In “Stanford was founded in 1885,” for example, a task may care only about the year, yet the prediction of “founded” also affects the score.

For a prompt \(x\) and response \(y=(y_1,\ldots,y_m)\), we can compute perplexity only over the response:

\[
\begin{aligned}
\operatorname{PPL}(y\mid x)
&= \exp\left(-\frac{1}{m}\sum_{i=1}^{m}\log p(y_i\mid x,y_{\lt i})\right) \\
&= p(y\mid x)^{-1/m}.
\end{aligned}
\]

#### Cloze and Sentence Completion: Perplexity in Benchmark Form

LAMBADA asks a model to use a long context to predict the final word. Although it reports accuracy, the answer is still selected through conditional probability. [Paperno et al., 2016](https://arxiv.org/abs/1606.06031)

<figure>
  <img src="../../../posts/evaluation/lambada.png" alt="Examples from the LAMBADA cloze benchmark" loading="lazy">
  <figcaption>LAMBADA target words usually require the full passage context. Source: <a href="https://arxiv.org/abs/1606.06031">LAMBADA paper</a>.</figcaption>
</figure>

HellaSwag presents several sentence completions. The model assigns each candidate a probability under the context and selects the most plausible one, making the task another comparison of conditional probabilities. [Zellers et al., 2019](https://arxiv.org/abs/1905.07830)

<figure>
  <img src="../../../posts/evaluation/hellaswag.png" alt="Examples from the HellaSwag multiple-choice sentence completion benchmark" loading="lazy">
  <figcaption>HellaSwag uses adversarial filtering to construct incorrect options that are difficult to reject through superficial patterns alone. Source: <a href="https://arxiv.org/abs/1905.07830">HellaSwag paper</a>.</figcaption>
</figure>

#### Perplexity Leaderboards: Submitted Probabilities Must Be Valid

- Participants submit a language model, and the leaderboard computes `log_prob = LM(test_data)`.
- The evaluator must trust that these scores come from a normalized probability distribution whose probabilities sum to one.
- Downstream tasks usually generate `response = LM(prompt)` and directly compute response accuracy, without trusting probabilities reported by the model.

#### Perplexity Summary

- PPL remains widely used in model development because it follows smooth scaling behavior.
- PPL cannot replace evaluations grounded in realistic use cases.

### 2.2 Exam Benchmarks

Exams are also useful for evaluating language models:

- The subject and difficulty can be controlled.
- Questions can have unambiguous answers that are easy to grade automatically.

#### MMLU: Broad Knowledge Rather Than Language Understanding

[Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) has the following properties:

- It covers 57 subjects, including mathematics, U.S. history, law, and morality.
- It uses multiple-choice questions collected by students from public online sources.
- Despite its name, it primarily tests knowledge rather than language understanding.
- GPT-3 was originally evaluated with few-shot prompting.

<figure>
  <img src="mmlu.png" alt="MMLU few-shot prompt examples and GPT-3 performance across task categories" loading="lazy">
  <figcaption>MMLU few-shot prompting and GPT-3 test results. Source: <a href="https://arxiv.org/abs/2009.03300">MMLU paper</a>.</figcaption>
</figure>

The [HELM MMLU prediction viewer](https://crfm.stanford.edu/helm/mmlu/latest/) exposes individual prompts, model predictions, and scores.

#### MMLU-Pro: Raising Difficulty Through Cleaning and More Choices

[MMLU-Pro](https://arxiv.org/abs/2406.01574) addresses saturation in MMLU through several changes:

- It removes noisy or trivial questions.
- It expands each question from 4 choices to 10.
- It evaluates with chain-of-thought (CoT) prompting, giving models a fuller opportunity to reason.

> Relative to MMLU, model accuracy drops by 16%–33% on MMLU-Pro, indicating that the benchmark is less saturated. [Wang et al., 2024](https://arxiv.org/abs/2406.01574)

<figure>
  <img src="mmlu-pro.png" alt="Comparison of MMLU and MMLU-Pro accuracy, choice distributions, and prompting methods" loading="lazy">
  <figcaption>MMLU-Pro improves discrimination through cleaning, more choices, and reasoning prompts. Source: <a href="https://arxiv.org/abs/2406.01574">MMLU-Pro paper</a>.</figcaption>
</figure>

#### GPQA: Expert-Level Questions That Resist Search

[Graduate-Level Google-Proof Q&A (GPQA)](https://arxiv.org/abs/2311.12022) was written by 61 PhD-level contractors recruited through Upwork and filtered through multiple validation stages.

<figure>
  <img src="gpqa.png" alt="GPQA question writing, expert validation, and non-expert validation pipeline" loading="lazy">
  <figcaption>GPQA retains questions on which experts agree but non-experts still struggle even with search access. Source: <a href="https://arxiv.org/abs/2311.12022">GPQA paper</a>.</figcaption>
</figure>

> - Domain experts achieve 65% accuracy.
> - Non-experts achieve 34% accuracy with Google access and about 30 minutes per question.
> - GPT-4 achieves 39% accuracy. [Rein et al., 2023](https://arxiv.org/abs/2311.12022)

#### HLE: Extending the Frontier With Cross-Disciplinary Questions

[Humanity's Last Exam (HLE)](https://arxiv.org/abs/2501.14249) targets knowledge benchmarks that frontier models are beginning to saturate:

- It contains 2,500 cross-disciplinary questions spanning multimodal, multiple-choice, and short-answer formats.
- A US\$500,000 prize pool and paper co-authorship incentivized experts to submit questions.
- Frontier language models filtered the submissions before several rounds of expert review.

<figure>
  <img src="hle-examples.png" alt="HLE examples from classics, ecology, mathematics, and computer science" loading="lazy">
  <figcaption>HLE covers highly specialized fields and uses both text and images. Source: <a href="https://arxiv.org/abs/2501.14249">HLE paper</a>.</figcaption>
</figure>

<figure>
  <img src="hle-pipeline.png" alt="HLE construction pipeline from submissions through model difficulty checks and expert review" loading="lazy">
  <figcaption>HLE selected 2,500 public questions from roughly 70,000 submission attempts. Source: <a href="https://arxiv.org/abs/2501.14249">HLE paper</a>.</figcaption>
</figure>

<figure>
  <img src="hle-results.png" alt="Accuracy of several language models on HLE, GPQA, MATH, and MMLU" loading="lazy">
  <figcaption>Frontier models at the time remained far from saturating HLE. Source: <a href="https://arxiv.org/abs/2501.14249">HLE paper</a>.</figcaption>
</figure>

#### Evaluation Methods for the Four Exam Benchmarks

- <strong>MMLU:</strong> the prompt ends with `Answer:`. The evaluator reads the next-token log probabilities of A–D, selects the most probable letter, compares it with the gold option, and reports accuracy. [Original MMLU evaluator](https://github.com/hendrycks/test/blob/master/evaluate.py)
- <strong>MMLU-Pro:</strong> the model may first generate a chain of thought, but must finish with `The answer is (X)`. The evaluator extracts the final option from A–J and compares it with the gold option. [Official MMLU-Pro evaluator](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py)
- <strong>GPQA:</strong> each question has four options. Zero-shot, few-shot, chain-of-thought, and web-assisted protocols change how the answer is produced, but accuracy still depends on whether the final option matches the gold option. [Rein et al., 2023](https://arxiv.org/abs/2311.12022)
- <strong>HLE:</strong> the model returns an answer, explanation, and confidence. A separate judge model extracts the final answer and compares it with the reference, allowing a small tolerance for numerical answers. The benchmark reports accuracy and calibration error. [Official HLE evaluator](https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_judge_results.py)

All four benchmarks evaluate whether the final answer is correct, but they obtain that answer differently. Model comparisons must therefore fix the prompt template, answer extraction rules, and scoring implementation.

#### Exam Benchmark Summary

- Questions trend harder as models improve and saturate older benchmarks.
- Multiple-choice questions can be made arbitrarily difficult while remaining easy to grade.
- Exams do not fully capture open-ended real-world use, where a single correct answer may not exist.

### 2.3 Chat Benchmarks

Exam benchmarks have explicit answers, but users rarely ask AI assistants multiple-choice questions. Real requests produce open-ended responses whose correctness, helpfulness, and style are difficult to reduce to a single reference answer.

<figure>
  <img src="arena-beets.png" alt="Comparison of two Arena AI responses to a question about herbs for a beet and goat cheese salad" loading="lazy">
  <figcaption>The same open-ended question can produce two reasonable responses with different styles, requiring a rater to decide which is better. Source: <a href="https://arena.ai/">Arena AI</a>.</figcaption>
</figure>

#### Chatbot Arena: Anonymous Pairwise Comparison by Humans

[Chatbot Arena](https://arxiv.org/abs/2403.04132) crowdsources human preferences:

- A user submits a real prompt.
- The system obtains responses from two randomly selected anonymous models.
- The user chooses response A, response B, both are good, or both are bad.

Pairwise comparisons can be used to fit Elo ratings. If models A and B have ratings \(R_A\) and \(R_B\), the probability that A wins is:

\[
P(A\succ B)=\frac{1}{1+10^{(R_B-R_A)/400}}.
\]

The ratings are fitted by maximizing the probability of the observed comparisons, producing the [Arena AI leaderboard](https://arena.ai/leaderboard). This design has several properties:

- Prompts come from real users, while new prompts and models can be added continuously.
- Every model does not need to answer exactly the same prompts.
- The user population is uncontrolled and may contain biases, vote manipulation, or spam.
- Binary preference conflates response style with factual correctness.
- Users may be unable to verify an answer, and model sycophancy may influence their choices.

#### AlpacaEval: Reducing Cost With Model Judges

[AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) replaces per-example human comparisons with automatic judging:

- It contains 805 instructions gathered from several sources.
- GPT-4 Preview compares the evaluated model's response with a baseline response.
- The metric is win rate against the baseline.

Language-model judges favor longer responses, allowing models to improve their ranking through verbosity. AlpacaEval 2.0 uses regression to control for length differences and reports a length-controlled win rate. [Dubois et al., 2024](https://arxiv.org/abs/2404.04475)

<figure>
  <img src="alpacaeval-chat-correlations.png" alt="Spearman correlations between automatic evaluation metrics and Chatbot Arena rankings" loading="lazy">
  <figcaption>Automatic metrics are often validated by correlation with Chatbot Arena's human-preference rankings; length-controlled AlpacaEval 2.0 has a stronger correlation. Source: <a href="https://github.com/tatsu-lab/alpaca_eval">AlpacaEval</a>.</figcaption>
</figure>

<figure>
  <img src="alpacaeval-leaderboard.png" alt="AlpacaEval 2.0 leaderboard with length-controlled win rates" loading="lazy">
  <figcaption>Raw and length-controlled win rates can produce different rankings. Source: <a href="https://tatsu-lab.github.io/alpaca_eval/">AlpacaEval leaderboard</a>.</figcaption>
</figure>

#### WildBench: Improving Reliability With Real Conversations and Checklists

[WildBench](https://arxiv.org/abs/2406.04770) builds an automatic evaluation from real user requests:

- It selects 1,024 challenging examples from roughly one million human–chatbot conversations.
- It creates a task-specific checklist of capabilities and errors to inspect.
- GPT-4 Turbo judges each response against that checklist.
- It provides both pairwise WB-Reward and single-response WB-Score metrics.
- Its results correlate strongly with Chatbot Arena rankings.

<figure>
  <img src="wildbench.png" alt="WildBench pipeline for checklist-guided pairwise and single-response evaluation" loading="lazy">
  <figcaption>Task-specific checklists structure the judging process and produce interpretable rationales. Source: <a href="https://arxiv.org/abs/2406.04770">WildBench paper</a>.</figcaption>
</figure>

The [HELM WildBench prediction viewer](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench) exposes individual examples, model responses, and scores.

#### Chat Benchmark Summary

- Open-ended responses lack a unique correct answer and are harder to evaluate than multiple-choice questions.
- Pairwise comparison between similar responses often provides a clearer preference signal.
- Both human and language-model judges introduce biases.
- Explicit checklists or rubrics improve judging reliability.

### 2.4 Agentic Benchmarks

Chat benchmarks evaluate what a language model <strong>says</strong>; agentic benchmarks evaluate what it <strong>does</strong>.

An agent combines a language model with an agent scaffold. The scaffold decides when to invoke the model, which tools to use, and how to continue from environmental feedback. Agentic benchmarks therefore tend to require tool use and multi-step iteration.

#### SWE-bench: Verifying Code Repairs With Unit Tests

[SWE-bench](https://arxiv.org/abs/2310.06770) turns real software issues into executable tasks:

- It contains 2,294 tasks from 12 Python repositories.
- The input is a codebase and a GitHub issue description.
- The agent must modify the code and produce a patch.
- The primary metric is whether the unit tests pass.

<figure>
  <img src="swebench.png" alt="SWE-bench issue description, gold patch, generated patch, and unit-test results" loading="lazy">
  <figcaption>SWE-bench does not require textual identity with the reference patch; tests determine whether the repair works. Source: <a href="https://arxiv.org/abs/2310.06770">SWE-bench paper</a>.</figcaption>
</figure>

Current results are available on the [LLM Stats SWE-bench Verified page](https://llm-stats.com/benchmarks/swe-bench-verified).

#### Terminal-Bench: Long-Horizon Tasks in a General Terminal Environment

[Terminal-Bench](https://arxiv.org/abs/2601.11868) uses the computer terminal as a common environment:

- The terminal is simple and general enough for programming, data processing, and system operations.
- The agent reads the task, executes commands, and changes state inside an isolated Docker container.
- Hidden tests inspect the final environment after the task.
- Contributors crowdsourced 229 tasks, with 89 forming Terminal-Bench 2.0.

<figure>
  <img src="terminal-bench.png" alt="Terminal-Bench task input, Docker execution environment, and hidden-test workflow" loading="lazy">
  <figcaption>The agent receives the task description and execution environment, but not the test files or reference solution. Source: <a href="https://www.tbench.ai/">Terminal-Bench</a>.</figcaption>
</figure>

<details>
<summary>View Terminal-Bench task difficulty and results</summary>

<figure>
  <img src="terminal-bench-human-time.png" alt="Distribution of time required by expert and junior engineers for Terminal-Bench tasks" loading="lazy">
  <figcaption>Experts usually finish within one day, while junior engineers more often require several hours or days. Source: <a href="https://arxiv.org/abs/2601.11868">Terminal-Bench paper</a>.</figcaption>
</figure>

<figure>
  <img src="terminal-bench-results.png" alt="Terminal-Bench 2.0 agent leaderboard snapshot" loading="lazy">
  <figcaption>The leaderboard identifies both the agent scaffold and the underlying model, reflecting their combined performance. Source: <a href="https://www.tbench.ai/">Terminal-Bench leaderboard</a>.</figcaption>
</figure>

</details>

#### CyBench: Evaluating Cybersecurity Agents With Capture-the-Flag Tasks

[CyBench](https://arxiv.org/abs/2408.08926) contains 40 Capture the Flag (CTF) tasks:

- The agent interacts with an isolated cybersecurity environment through Bash.
- Tasks require inspecting files, analyzing services, exploiting vulnerabilities, and submitting a flag.
- Subtask questions can expose intermediate progress.
- Human first-solve time measures task difficulty.

<figure>
  <img src="cybench.png" alt="CyBench task description, agent interaction, environment, and answer evaluation workflow" loading="lazy">
  <figcaption>CyBench records both the final flag and subtask answers to reveal progress on complex security tasks. Source: <a href="https://arxiv.org/abs/2408.08926">CyBench paper</a>.</figcaption>
</figure>

<figure>
  <img src="cybench-agent.png" alt="CyBench agent loop across acting, execution, observation, and memory updates" loading="lazy">
  <figcaption>The agent repeatedly chooses commands, observes the environment, and updates memory before submitting an answer. Source: <a href="https://arxiv.org/abs/2408.08926">CyBench paper</a>.</figcaption>
</figure>

<details>
<summary>View CyBench results</summary>

<figure>
  <img src="cybench-results.png" alt="CyBench task solve rate, subtask completion, and hardest solved task" loading="lazy">
  <figcaption>CyBench reports full-task solve rate, subtask completion, and the hardest solved task. Source: <a href="https://llm-stats.com/benchmarks/cybench">LLM Stats CyBench</a>.</figcaption>
</figure>

</details>

Current results are available on the [LLM Stats CyBench page](https://llm-stats.com/benchmarks/cybench).

#### MLE-bench: Completing Machine Learning Engineering in Kaggle Competitions

[MLE-bench](https://arxiv.org/abs/2410.07095) turns 75 Kaggle competitions into agent tasks. Instead of merely answering a question, an agent must complete an end-to-end machine learning workflow:

- read the competition description and process the data;
- train, test, and debug models;
- produce a valid `submission.csv`;
- receive a score under the original competition metric.

<figure>
  <img src="mlebench.png" alt="MLE-bench workflow from Kaggle competition materials to agent submission and grading" loading="lazy">
  <figcaption>MLE-bench evaluates whether an agent can complete a machine learning engineering task, not merely answer a question. Source: <a href="https://arxiv.org/abs/2410.07095">MLE-bench paper</a>.</figcaption>
</figure>

<details>
<summary>View MLE-bench results</summary>

<figure>
  <img src="mlebench-results.png" alt="Performance of different agents across MLE-bench difficulty levels" loading="lazy">
  <figcaption>The leaderboard must identify the agent, underlying language model, and runtime together. Source: <a href="https://github.com/openai/mle-bench">MLE-bench</a>.</figcaption>
</figure>

</details>

#### Agent Scaffolds: Execution Frameworks Change Model Capabilities

An [agent scaffold](https://www.philschmid.de/agents-2.0-deep-agents) organizes interactions among the model, tools, and environment. Common designs include:

- **explicit planning**: maintain a todo list and check off completed steps;
- **hierarchical delegation**: let a primary agent call sub-agents with cleaner contexts;
- **persistent memory**: preserve information across steps by reading and writing files;
- **context engineering**: constrain execution with more explicit process instructions.

<figure>
  <img src="agent-scaffolds.png" alt="An agent completing tasks through planning, sub-agents, and persistent memory" loading="lazy">
  <figcaption>An agent scaffold can combine planning, orchestration, sub-agents, and persistent memory. Source: <a href="https://www.philschmid.de/agents-2.0-deep-agents">Agent 2.0: Deep Agents</a>.</figcaption>
</figure>

#### Agentic Benchmark Summary

- Agents dramatically expand the range of tasks a language model can complete.
- Agent scaffolds strongly affect the resulting capability.
- Evaluating an agent means evaluating both its scaffold and its language model.

### 2.5 Pure Reasoning Benchmarks

The preceding tasks all depend on linguistic or world knowledge. Pure reasoning benchmarks try to reduce the advantage of memorized facts and test whether a model can infer the rules of a new task.

#### ARC-AGI: Novel Tasks Reduce the Value of Memorization

[ARC-AGI](https://arcprize.org/arc-agi) uses visual tasks designed to be solvable by humans yet challenging for AI. Each task follows a different rule, so directly memorizing training examples does not solve new instances.

- **ARC-AGI-1 (2019)**: infer a grid-transformation rule from a few input-output examples;
- **ARC-AGI-2 (March 2025)**: introduce more multi-step reasoning tasks;
- **ARC-AGI-3 (March 2026)**: extend static puzzles into interactive environments.

<figure>
  <img src="arc-task-grids.jpg" alt="ARC-AGI task requiring inference of a colored-grid transformation from examples" loading="lazy">
  <figcaption>ARC-AGI requires a model to infer a rule from examples and apply it to a new input. Source: <a href="https://arcprize.org/arc-agi">ARC Prize</a>.</figcaption>
</figure>

<figure>
  <img src="arc-agi-2-unsolved.png" alt="ARC-AGI-2 visual grid tasks requiring multi-step reasoning" loading="lazy">
  <figcaption>ARC-AGI-2 adds more complex compositional and multi-step transformations. Source: <a href="https://arcprize.org/arc-agi">ARC Prize</a>.</figcaption>
</figure>

Pretrained language models initially made almost no progress on ARC-AGI. Scores began to rise substantially only after reasoning models such as o1 and o3 appeared.

<details>
<summary>View ARC-AGI-1 and ARC-AGI-2 score trends</summary>

<figure>
  <img src="arc-agi-results.png" alt="ARC-AGI-1 and ARC-AGI-2 scores by model release date" loading="lazy">
  <figcaption>ARC-AGI-1 scores rose rapidly with reasoning models and coding agents, while ARC-AGI-2 remains harder. Source: <a href="https://arcprize.org/arc-agi">ARC Prize</a>.</figcaption>
</figure>

</details>

[ARC-AGI-3](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) further requires an agent to observe an environment, act, and adapt its strategy from feedback.

<figure>
  <img src="arc-agi-3.png" alt="Interactive visual environment in ARC-AGI-3" loading="lazy">
  <figcaption>ARC-AGI-3 extends abstract reasoning into environments that require continued interaction. Source: <a href="https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf">ARC-AGI-3 technical report</a>.</figcaption>
</figure>

<details>
<summary>View ARC-AGI-3 results</summary>

<figure>
  <img src="arc-agi-3-results.png" alt="Scores of different models on ARC-AGI-3" loading="lazy">
  <figcaption>Current models still score very poorly on ARC-AGI-3. Source: <a href="https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf">ARC-AGI-3 technical report</a>.</figcaption>
</figure>

</details>

#### Pure Reasoning Benchmark Summary

- The goal is to disentangle reasoning from prior knowledge, though the two are difficult to separate completely.
- The tasks remain bounded by human reasoning rather than superhuman reasoning.
- These benchmarks clearly expose capability gaps in current models.

### 2.6 Safety Benchmarks

Safety evaluation resembles crash testing: define the behaviors that should not occur, then test whether the system produces them under pressure.

[HarmBench](https://arxiv.org/abs/2402.04249) is built around 510 harmful behaviors that violate laws or social norms and tests whether a model or agent will carry out the corresponding requests.

The [HELM HarmBench leaderboard](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench) reports aggregate results, while the [safety-failure example page](https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4) exposes individual cases.

#### AIR-Bench: Organizing Risks From Policies and Regulations

[AIR-Bench](https://arxiv.org/abs/2407.17436) derives risks from regulatory frameworks and company policies, organizing them into 314 fine-grained categories and constructing 5,694 test prompts.

<figure>
  <img src="air-bench-overview.png" alt="AIR-Bench taxonomy covering system safety, content safety, societal risks, and legal rights" loading="lazy">
  <figcaption>AIR-Bench uses a four-level risk taxonomy to unify safety policies from different organizations. Source: <a href="https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard">HELM AIR-Bench</a>.</figcaption>
</figure>

Results are available on the [HELM AIR-Bench leaderboard](https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard).

#### Jailbreaking: Bypassing Model Refusal Mechanisms

- Language models are commonly trained to refuse harmful instructions.
- [Greedy Coordinate Gradient (GCG)](https://arxiv.org/abs/2307.15043) automatically optimizes adversarial prompt suffixes to bypass safety restrictions.
- Prompts optimized against open-weight models can also transfer to closed models such as GPT-4.

<details>
<summary>View GCG jailbreak examples</summary>

<figure>
  <img src="gcg-examples.png" alt="GCG adversarial suffixes causing several language models to bypass refusal mechanisms" loading="lazy">
  <figcaption>Nonsensical adversarial suffixes can cause different models to answer requests that they should refuse. Source: <a href="https://arxiv.org/abs/2307.15043">Universal and Transferable Adversarial Attacks on Aligned Language Models</a>.</figcaption>
</figure>

</details>

#### The Boundary of Safety Depends on Context

- Politics, law, and social norms vary across countries and settings, making many safety judgments highly contextual.
- Risks take many forms, including hallucination, sycophancy, assistance with crime, inequality, and erosion of critical thinking.
- Capability and propensity should be separated: a system may possess a capability while refusing to exercise it.

Cybersecurity agents have a <strong>dual-use</strong> character: capable agents such as Mythos can be used either to break into systems or to conduct legitimate penetration testing.

## 3. Evaluation Realism, Validity, and Purpose

### 3.1 Realism: Does the Evaluation Represent Real Use?

Ecological validity measures how closely an evaluation reflects practical use:

- exam benchmarks such as GPQA are far removed from real work;
- Chatbot Arena uses real user prompts, but their distribution is uncontrolled;
- more realistic evaluations draw directly from professional tasks or actual usage.

#### GDPVal: Measuring Practical Work With Occupational Tasks

[GDPVal](https://arxiv.org/abs/2510.04374) covers 44 occupations from the nine largest sectors of the US economy by GDP. Its tasks come from professionals with roughly 14 years of experience on average.

<figure>
  <img src="gdpval.png" alt="GDPVal tasks in manufacturing engineering, financial analysis, nursing, video editing, and customer service" loading="lazy">
  <figcaption>GDPVal asks models to produce documents, spreadsheets, designs, or multimedia artifacts resembling professional deliverables. Source: <a href="https://arxiv.org/abs/2510.04374">GDPVal paper</a>.</figcaption>
</figure>

#### MedHELM: Building Tasks From Clinical Work Rather Than Medical Exams

[MedHELM](https://arxiv.org/abs/2505.23802) moves beyond standardized medical exams. Twenty-nine clinicians contributed 121 clinical tasks spanning both public and private datasets.

<figure>
  <img src="medhelm-overview.png" alt="MedHELM workflow from clinical task categorization and datasets to model evaluation and community resources" loading="lazy">
  <figcaption>MedHELM covers clinical decisions, note generation, patient communication, medical research, and administrative workflows. Source: <a href="https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard">MedHELM</a>.</figcaption>
</figure>

#### Clio: Extracting Usage Patterns From Real Conversations

[Clio](https://arxiv.org/abs/2412.13678) uses language models to analyze real user data and publishes aggregate patterns in user requests.

<details>
<summary>View Clio topic-classification results</summary>

<figure>
  <img src="clio-table4.png" alt="Comparison between Clio-predicted user conversation categories and human annotations" loading="lazy">
  <figcaption>Clio's counts for common topics such as software development, homework help, and technical troubleshooting are close to human annotations. Source: <a href="https://arxiv.org/abs/2412.13678">Clio paper</a>.</figcaption>
</figure>

</details>

Real data can improve ecological validity but is more likely to expose private information. **Evaluation realism and privacy protection are in tension.**

### 3.2 Validity: Are the Evaluation Results Trustworthy?

#### Train-Test Overlap: The Model May Have Seen the Test Questions

A basic rule of machine learning is not to train on the test set. Earlier datasets usually had explicit training and test splits. Models are now trained on Internet-scale data, and outside evaluators often cannot know whether test questions appeared in the training data.

There are four routes for addressing train-test overlap:

1. **Infer overlap from model behavior.** [Oren et al., 2023](https://arxiv.org/abs/2310.17623) exploit the exchangeability of data points by comparing model probabilities for canonical and shuffled orders. A consistent preference for the canonical order may indicate prior exposure.
2. **Establish reporting norms.** Model providers should disclose overlap-detection methods and statistics. [Zhang et al., 2024](https://arxiv.org/abs/2410.08385)
3. **Continuously build fresh evaluations.** [LiveCodeBench](https://arxiv.org/abs/2403.07974) and [UncheatableEval](https://github.com/Jellyfish042/uncheatable_eval) collect tasks from new webpages or competitions, although copying can make timestamps unreliable.
4. **Use private evaluations.** Internal company codebases or personal writings are less likely to overlap with Internet training data; such data are especially convenient for perplexity evaluation.

<figure>
  <img src="contamination-exchangeability.png" alt="Detecting training-data contamination by comparing canonical and shuffled dataset orders" loading="lazy">
  <figcaption>An unusually high log-probability for the canonical order can provide evidence of training-data contamination. Source: <a href="https://arxiv.org/abs/2310.17623">Proving Test Set Contamination in Black-Box Language Models</a>.</figcaption>
</figure>

#### Dataset Quality: Correct Answers and Test Cases Can Also Be Wrong

- [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) uses human review to repair unsolvable or insufficiently tested SWE-bench tasks.
- [Platinum benchmarks](https://arxiv.org/abs/2502.03461) recheck questions, answers, and ambiguity to reduce labeling errors.
- Agentic benchmarks can have insufficient test cases, allowing trivial agents to pass tasks. [Kirova et al., 2025](https://arxiv.org/abs/2507.02825)
- [Docent](https://transluce.org/introducing-docent) uses language models to inspect agent traces and identify problems in the evaluation itself.

<details>
<summary>View benchmark errors and the effect of cleaning</summary>

<figure>
  <img src="benchmark-platinum-1.jpg" alt="Examples of mislabeled questions, logical contradictions, ambiguity, and missing conditions in benchmarks" loading="lazy">
  <figcaption>Benchmark data can contain incorrect labels, logical contradictions, ambiguity, or missing conditions. Source: <a href="https://arxiv.org/abs/2502.03461">Platinum benchmark paper</a>.</figcaption>
</figure>

<figure>
  <img src="benchmark-platinum-2.jpg" alt="Average number of errors in several benchmarks before and after cleaning" loading="lazy">
  <figcaption>Error rates vary substantially across benchmarks and usually fall sharply after cleaning. Source: <a href="https://arxiv.org/abs/2502.03461">Platinum benchmark paper</a>.</figcaption>
</figure>

</details>

### 3.3 Purpose: First Decide What the Evaluation Should Answer

There is no single correct evaluation; its form depends on the question being asked:

1. users or companies need to choose a model for a specific use case;
2. researchers want to measure a model's raw capabilities;
3. companies and policymakers need to understand benefits and harms;
4. model developers need feedback for improving the model.

#### Evaluating Methods or Evaluating Models and Systems

- Before foundation models, standardized training and test splits were mainly used to compare **methods**.
- Today's leaderboards usually compare **models or systems**, whose training data, tools, and inference strategies may all differ.
- [nanoGPT speedrun](https://x.com/karpathy/status/1846790537262571739) is an exception that evaluates methods: it fixes the data and target validation loss, then compares the compute time needed to reach that target.

<details>
<summary>View the nanoGPT speedrun example</summary>

<figure>
  <img src="karpathy-nanogpt-speedrun.png" alt="nanoGPT speedrun comparing training efficiency at a fixed target validation loss" loading="lazy">
  <figcaption>With data and target loss fixed, score changes more directly reflect improvements in the training method. Source: <a href="https://x.com/karpathy/status/1846790537262571739">Andrej Karpathy</a>.</figcaption>
</figure>

</details>

Evaluating methods encourages algorithmic innovation, while evaluating models or systems better serves downstream users. **Either way, the rules of the game must be explicit.**

---

## References

[1] Stanford CS336, "Lecture 12 - Evaluation," Executable Lecture, Stanford University, 2026. [Online]. Available: https://cs336.stanford.edu/lectures?trace=lecture_12&step=1.
