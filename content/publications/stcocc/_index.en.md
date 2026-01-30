---
title: "STCOcc: Sparse Spatial-Temporal Cascade Renovation for 3D Occupancy and Scene Flow Prediction"
date: 2025-06-01
venue: CVPR 2025
authors:
    - name: Zhimin Liao
      home: https://lzzzzzm.github.io/
    - name: Ping Wei
      home: https://gr.xjtu.edu.cn/en/web/pingwei
    - name: Shuaijia Chen
      home: https://cspaulia.github.io/cspaulia-blog/
    - name: Haoxuan Wang
    - name: Ziyang Ren
selected: true
Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_STCOcc_Sparse_Spatial-Temporal_Cascade_Renovation_for_3D_Occupancy_and_Scene_CVPR_2025_paper.pdf
Code: https://github.com/lzzzzzm/STCOcc
summary: A sparse spatial-temporal cascade renovation method for 3D occupancy and scene flow prediction with only 8.7 GB GPU memory
cover:
    image: cover.png
---

3D occupancy and scene flow offer a detailed and dynamic representation of 3D scene. Recognizing the sparsity and complexity of 3D space, previous vision-centric methods have employed implicit learning-based approaches to model spatial and temporal information. However, these approaches struggle to capture local details and diminish the model's spatial discriminative ability. To address these challenges, we propose a novel explicit state-based modeling method designed to leverage the occupied state to renovate the 3D features. Specifically, we propose a sparse occlusion-aware attention mechanism, integrated with a cascade refinement strategy, which accurately renovates 3D features with the guidance of occupied state information. Additionally, we introduce a novel method for modeling long-term dynamic interactions, which reduces computational costs and preserves spatial information.

```
