---
title: "Cylinderfusion: Self-Adaptive Cylindrical 3+1D Radar-Camera Fusion for Waterway Point Cloud Segmentation"
date: 2026-05-03
venue: ICASSP 2026
authors:
    - name: "<span class='author-highlight'>Shuaijia Chen</span>"
      home: https://cspaulia.github.io/cspaulia-blog/
    - name: Ping Wei
      home: https://gr.xjtu.edu.cn/en/web/pingwei
    - name: Linyu Zhang
    - name: Zhimin Liao
      home: https://lzzzzzm.github.io/
    - name: Bole Wang

selected: true
paper: https://ieeexplore.ieee.org/document/11461037
code: https://github.com/CSPaulia/cylinderfusion
summary: A robust radar–camera fusion network for waterway point cloud segmentation
cover:
    image: cover.png
---

Point cloud segmentation is crucial for unmanned vehicle perception on water, and radar–camera fusion further improves its performance. Most existing BEV methods fuse features within cubic voxel space, ignoring the non-uniform distribution of outdoor point clouds and image frustum points. Their performance also degrades severely under adverse weather or sensor malfunctions. To address these, we propose CylinderFusion, a robust radar–camera fusion network. It introduces a novel paradigm for multimodal fusion within cylindrical voxel space and incorporates a specially designed scatter module. To improve robustness, we introduce a Dynamic Feature Selection (DFS) mechanism that adaptively weights features during fusion. Our method achieves state-of-the-art results on the large-scale WaterScenes dataset and demonstrates strong performance on the VoD dataset. Extensive ablations validate its effectiveness.
