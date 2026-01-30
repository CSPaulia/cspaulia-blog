---
title: "CLIP and Its Follow-up Works"
date: 2025-06-03T10:46:03+08:00
# weight: 1
# aliases: ["/first"]
categories: ["Base Model"]
tags: ["CLIP"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "CLIP and its follow-up works"
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
    image: "clip_cover.png" # image path/url
    alt: "clip with advanced methods" # alt text
    caption: "clip with advanced methods" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
    hiddenInList: false # hide on list pages and home
    # class: "post-cover"
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## I. CLIP

<p align="center">
  {{< img src="clip.png" alt="clip" >}}
</p>

1. Contrastive pre-training:
    
  - Collect $N$ image–text pairs from the internet (OpenAI used 400M image–text pairs) as positive pairs.
        
  - For each image, pair it with the other $N-1$ texts to form negative pairs (text does not match the image).
        
  - Feed the $N$ images into the image encoder and the corresponding texts into the text encoder. Compute dot products between image and text features to obtain a similarity matrix.
        
  - Treat each row as an $N$-way classification problem. For row $i$, the correct label is $i$ (make the $(i,i)$ entry the largest in that row). Apply cross-entropy between row $i$ and label $i$.
        
  - Similarly, treat each column as an $N$-way classification problem and apply cross-entropy so that the $(i,i)$ entry is the largest in column $i$.
        
2. Build a classifier from labels:
    
  - Turn text labels into prompts/sentences and feed them into the text encoder to get text features.
        
3. Zero-shot prediction:
    
  - Match the label text features with image features by similarity to make predictions.

## II. Using CLIP for Semantic Segmentation

### 2.1 LSeg

<p align="center">
  {{< img src="lseg.png" alt="lseg" >}}
</p>

1. Relationship to CLIP:
    
  - Use CLIP’s aligned feature space: map semantic labels and pixel features into the same space and predict segmentation by similarity.
    
  - Text encoder: the same as CLIP, and frozen during training :snowflake:.

2. Differences from CLIP:

| Method       | Training        | Text encoder params | Similarity computation |
|--------------|------------|---------------|------------|
| CLIP         | Contrastive     | Trainable          | Similarity between image–text pairs |
| LSeg         | Supervised      | Frozen             | Similarity between image features and text features |

- CLIP input: multiple image–text pairs
- LSeg input: one image + labels (can be viewed as descriptive text for the image)

### 2.2 GroupViT

<p align="center">
  {{< img src="groupvit_overview.png" alt="groupvit_overview" width="50%" >}}
</p>

1. Relationship to CLIP and LSeg:

| Method   | Training      | Similarity computation |
|--------------|------------|---------------|------------|
| CLIP     | Contrastive   | Similarity between image–text pairs |
| LSeg     | Supervised    | Similarity between image features and text features |
| GroupViT | Contrastive   | Similarity between image–text pairs |

- LSeg is trained with supervision on top of CLIP’s aligned feature space.
- GroupViT adjusts the vision encoder architecture in CLIP to better fit segmentation, while keeping CLIP-style contrastive learning.

2. Architecture details:

<p align="center">
  {{< img src="groupvit.png" alt="groupvit" >}}
</p>

- Model input: `image patches` + `learnable group tokens`
- Segmentation: assign `patch features` to `learnable group tokens` via `Group Block`
- `learnable group tokens`: similar to cluster centroids

## III. Using CLIP for Object Detection

### 3.1 ViLD

<p align="center">
  {{< img src="vild_compare.png" alt="vild_compare" >}}
</p>

- `Vanilla Detector` = `Head` + `Classifier` + supervised cross-entropy
- `ViLD-text` = `Head` + `similarity matching` + supervised cross-entropy
  - Similarity-matching pipeline (CLIP-style):
    - Feed $n$ labels (via prompt engineering) into the `text encoder` to get $n$ `text embeddings`.
    - To cover regions that do not match any label, introduce a `background embedding`.
    - Match `region embeddings` with `text embeddings` and the `background embedding` to obtain $n$ class scores plus one background score. This replaces the `Classifier` and is frozen during training :snowflake:.
- `ViLD-image` = `teacher network` + `student network` + L1 distillation
  - `teacher network`: CLIP image encoder
  - `student network`: `Vanilla Detector`
  - To reduce training cost, pre-extract $m$ `region embeddings` using a pretrained detector.
- `ViLD` = `ViLD-text` + `ViLD-image`

## IV. Using CLIP for Visual Grounding

### 4.1 GLIP

<p align="center">
  {{< img src="glip.png" alt="glip" >}}
</p>

- Essentially supervised training.
- Compute similarity between `Regions` and `Words` to classify/caption `Regions`.
- Training requires knowing the alignment between `Regions` and `Words` in captions. To obtain it:
  - Detection datasets: build captions from bounding-box annotations (e.g., Banana → “There is a banana.”)
  - Caption datasets: use a GLIP model trained on detection data to find `Regions`–`Words` alignments and construct pseudo labels.

## V. Using CLIP for Image Generation

### 5.1 CLIPasso

<p align="center">
  {{< img src="clipasso.png" alt="clipasso" >}}
</p>

#### Motivation

Observation: prior sketch generation methods often work only for a specific category.

Goal: leverage CLIP’s strong generalization to generate sketches for arbitrary categories.

#### Pipeline

1. Generate a sketch:
  - Use the `image encoder` to obtain a heatmap.
  - Sample points based on the heatmap.
  - Aggregate points with `learnable parameters` to form Bézier curves, producing a sketch.
2. Constrain generation with CLIP:
  - Feed the generated sketch and the original image into two different CLIP image encoders.
  - $L_g$ constrains geometric consistency (closer is better).
  - $L_s$ constrains semantic consistency (closer is better).

## VI. Using CLIP for Video Retrieval

### 6.1 CLIP4Clip

#### Motivation

<p align="center">
  {{< img src="clip4clip.png" alt="clip4clip" >}}
</p>

- CLIP is designed for image–text pairs. For video retrieval, the task is matching one text query against multiple frames and finding the most relevant frames.
- CLIP4Clip explores three matching strategies:
  - Parameter-free type: no extra parameters (e.g., mean pooling; ignores temporal order)
  - Sequential type: temporal modules such as LSTM and Transformer
  - Tight type: a Transformer Encoder jointly learns text and video features and outputs similarity

### 6.2 ActionCLIP

<p align="center">
  {{< img src="actionclip_overview.png" alt="actionclip_overview" width="50%" >}}
</p>

<p align="center">
  {{< img src="actionclip.png" alt="actionclip" >}}
</p>

Similar to CLIP4Clip.

## VII. Using CLIP for Speech Recognition

### 7.1 AudioCLIP

<p align="center">
  {{< img src="audioclip.png" alt="audioclip" >}}
</p>

Add an audio encoder and follow CLIP-style objectives: `audio–image` contrastive learning and `audio–text` contrastive learning.

## VIII. Using CLIP for 3D Understanding

### 8.1 PointCLIP

<p align="center">
  {{< img src="pointclip.png" alt="pointclip" >}}
</p>

1. Project the point cloud into a 2D space.
2. Build prompts: `Point Cloud Depth Map of a [CLASS]`

## IX. Using CLIP for Depth Estimation

### 9.1 DepthCLIP

<p align="center">
  {{< img src="depthclip.png" alt="depthclip" >}}
</p>

- Build 7 text prompts: "This object is [distance class]", where `[distance class]` is:
  - 'giant'
  - 'extremely close'
  - 'close'
  - 'not in distance'
  - 'a little remote'
  - 'far'
  - 'unseen'
- The network is similar to LSeg and performs classification with `Softmax`.

---

<div class="zhihu-ref">
  <div class="zhihu-ref-title">References</div>
  <ol>
    <li><a href="https://www.bilibili.com/video/BV1FV4y1p7Lm?spm_id_from=333.788.videopod.sections&vd_source=9e4f1724ef60547fa31e3c8270245ff8" target="_blank">CLIP 改进工作串讲（上）【论文精读·42】</a></li>
    <li><a href="https://www.bilibili.com/video/BV1gg411U7n4?spm_id_from=333.788.videopod.sections&vd_source=9e4f1724ef60547fa31e3c8270245ff8" target="_blank">CLIP 改进工作串讲（下）【论文精读·42】</a></li>
  </ol>
</div>