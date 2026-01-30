---
title: "100 Evaluation Metrics (Work in Progress)"
date: 2025-07-09T14:05:03+08:00
# weight: 1
# aliases: ["/first"]
categories: ["Deep Learning Skills"]
tags: ["Metrics", "Evaluation"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "[Epoch  10/100] Updating..."
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
    image: "metric_cover.png" # image path/url
    alt: "Evaluation metrics overview" # alt text
    caption: "Evaluation metrics" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Classification metrics

> Core concepts: TP / TN / FP / FN
>
> Suppose the task is “detect whether someone is sick”. The model prediction vs. reality can be summarized as:
>
> | Reality \ Prediction | Predicted positive (sick) | Predicted negative (not sick) |
> |---|---|---|
> | Actually positive (sick) | ✅ TP (true positive) | ❌ FN (false negative) |
> | Actually negative (not sick) | ❌ FP (false positive) | ✅ TN (true negative) |
>
> - **TP (True Positive)**: predict “sick”, actually “sick” → correct
> - **TN (True Negative)**: predict “not sick”, actually “not sick” → correct
> - **FP (False Positive)**: predict “sick”, actually “not sick” → false alarm
> - **FN (False Negative)**: predict “not sick”, actually “sick” → miss (often more severe)

---

### Accuracy

- Overall fraction of correct predictions

$$
Accuracy = \frac{(TP + TN)}{(TP + TN + FP + FN)}
$$

---

### Recall

- Among all true positives, the fraction that the model successfully identifies (don’t miss)

$$
Recall = \frac{TP}{(TP + FN)}
$$

---

### Precision

- Among all predicted positives, the fraction that are truly positive (don’t accuse)

$$
Precision = \frac{TP}{(TP + FP)}
$$

---

### F1-score

- Harmonic mean of precision and recall

$$
F1 = \frac{2 \times (Precision \times Recall)}{(Precision + Recall)}
$$

#### Multi-class variants

- **Macro-F1**: arithmetic mean of per-class F1.
- **Micro-F1**: compute global TP/FP/FN then F1.
- **Weighted-F1**: weighted average of per-class F1 by class frequency.

---

## Ranking quality metrics

### Average Precision (AP)

**One sentence:** AP is the average precision for a single class (commonly explained via object detection).

#### 1) Sort predicted boxes by confidence (high → low)

Each prediction has:

- bounding box
- confidence score
- predicted label

Sort by confidence descending.

#### 2) Mark each prediction as TP or FP

Iterate predictions in that order:

- If it matches an **unmatched** ground-truth box with IoU ≥ a threshold (e.g., 0.5), mark as TP.
- Otherwise mark as FP.

#### 3) Compute cumulative precision & recall

As you traverse predictions, keep cumulative counts:

- cumulative TP count: $TP_i$
- cumulative FP count: $FP_i$

Then:

```python
precision_i = TP_i / (TP_i + FP_i)
recall_i = TP_i / GT_total  # GT_total = number of ground-truth boxes
```

#### 4) Plot the PR curve & compute area

Approximate the integral:

```python
# Recall, Precision are sorted by recall ascending
AP = 0
for i in range(1, len(recall)):
    AP += precision[i] * (recall[i] - recall[i-1])
```

---

### mAP (mean Average Precision)

Mean of AP over classes:

$$
mAP = \sum_i AP_i
$$

---

### mAP\@IoU=0.5 (mAP\@0.5)

- A prediction is counted as TP only when IoU ≥ 0.5.
- Compute AP per class, then average → mAP\@0.5.

---

### mAP\@0.5:0.95

- Sweep IoU thresholds from 0.5 to 0.95 with step 0.05 (10 thresholds).
- Average the AP across these thresholds.

---

### mAP\@k

Common in **image retrieval / recommendation / multi-label ranking**:

> For each query, compute AP using only the top-$k$ retrieved results, then average across queries.

#### Example

If you search “dog” and the system returns top 10 images:

- 6 are dogs, 4 are not, and dogs appear at ranks 1, 2, 3, 6, 7, 9
- Compute precision at those dog positions and average → AP@10
- Average AP@10 across all queries → mAP@10

---

## Metrics for image captioning

### SPICE

#### Why SPICE?

Traditional metrics:

- **BLEU**: focuses on n-gram overlap (like machine translation)
- **ROUGE**: emphasizes recall (often used for summarization)
- **CIDEr**: TF-IDF weighted n-gram similarity

They largely measure word/phrase overlap, but may miss whether the **semantic structure** is correct.

SPICE is motivated by:

> “When humans judge captions, they care whether you mention the right objects, attributes, and relations.”

#### How SPICE works

> **Key idea:** parse sentences into a semantic graph (scene graph): objects + attributes + relations, then compare candidate vs. references.

##### Scene graph structure

A sentence is represented as a set of triples:

```
G = {object, attribute, relation}
```

Example:

```
Sentence: "A red car is parked beside a white house"
G = {
  (object: car),
  (object: house),
  (attribute: car, red),
  (attribute: house, white),
  (relation: car, beside, house)
}
```

##### Mathematical definition

**Inputs:**

- $G$: semantic triples from the generated sentence (candidate graph)
- $R$: semantic triples from all reference sentences (reference graph)

**Goal:**

- Compute the $F_1$ score between $G$ and $R$

**Set matching:**

- Precision:

$$
P = \frac{|\mathcal{G} \cap \mathcal{R}|}{|\mathcal{G}|}
$$

- Recall:

$$
R = \frac{|\mathcal{G} \cap \mathcal{R}|}{|\mathcal{R}|}
$$

- F1-score:

$$
F_1 = \frac{2PR}{P + R}
$$

So: $\text{SPICE} = F_1(G, R)$

SPICE can be computed for sub-categories, such as:

- object matching (object F1)
- attribute matching (attribute F1)
- relation matching (relation F1)
- finer types like color, count, size, etc.

The final SPICE score is a weighted average:

$$
	ext{SPICE}=\sum_i w_i \cdot F_1^i
$$

where $F_1^i$ is the F1 for a semantic category and $w_i$ is its weight.

---