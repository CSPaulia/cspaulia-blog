---
title: "I²-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting"
date: 2025-10-01
venue: ICCV 2025
authors:
    - name: Zhimin Liao
      home: https://lzzzzzm.github.io/
    - name: Ping Wei
      home: https://gr.xjtu.edu.cn/en/web/pingwei
    - name: Ruijie Zhang
    - name: Shuaijia Chen
      home: https://cspaulia.github.io/cspaulia-blog/
    - name: Haoxuan Wang
    - name: Ziyang Ren
selected: true
paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Liao_I2-World_Intra-Inter_Tokenization_for_Efficient_Dynamic_4D_Scene_Forecasting_ICCV_2025_paper.pdf
arxiv: https://arxiv.org/abs/2507.09144
code: https://github.com/lzzzzzm/II-World
summary: An efficient 4D occupancy forecasting framework with dual tokenization achieving 94.8 FPS and 41.8% improvement over SOTA
cover:
    image: cover.png
---

Forecasting the evolution of 3D scenes and generating unseen scenarios through occupancy-based world models offers substantial potential to enhance the safety of autonomous driving systems. While tokenization has revolutionized image and video generation, efficiently tokenizing complex 3D scenes remains a critical challenge for 3D world models. To address this, we propose I²-World, an efficient framework for 4D occupancy forecasting. Our method decouples scene tokenization into intra-scene and inter-scene tokenizers. The intra-scene tokenizer employs a multi-scale residual quantization strategy to hierarchically compress 3D scenes while preserving spatial details. The inter-scene tokenizer residually aggregates temporal dependencies across timesteps. This dual design retains the compactness of 3D tokenizers while capturing the dynamic expressiveness of 4D approaches.

```
