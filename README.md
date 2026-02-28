<h1 align="center">Xspark Data Format</h1>

> This repository documents the data formats and dataloaders used in Xspark AI projects. If you have any questions, feel free to open an Issue.<br> 
> 此仓库用于介绍 Xspark AI 相关数据格式与 dataloader；如有任何问题，欢迎提 Issue 交流。

---

## Overview

This repo focuses on **standardizing multimodal robotics / ego data** and providing **reference dataloaders** so different embodiments and tasks can share a unified pipeline.

Key goals:
- Unified schema across embodiments (single-arm / dual-arm, grippers / dexterous hands, etc.)
- Multimodal data organization (vision, actions, states, annotations, etc.)
- Practical storage choices for scale (e.g., JPEG-compressed images in HDF5, MP4 for videos)

---

## Supported Formats

| Format | Version | Storage | Best for | Document | Loader |
|------:|:-------:|:--------|:---------|:---------|:------|
| Xspark Data Format | v1.0 | Single **HDF5** (all modalities) | unified robotics datasets / cross-embodiment training | [Doc](./format/xspark_data_format_v1.0/README.md) | [dataloader](./format/xspark_data_format_v1.0/dataloader.py) |
| X-Ego Data Format | v0.1 | Multi-file (**MP4** + **JSON**) | video-based X-Ego data & hand predictions | [Doc](./format/x-ego_v0.1/README.md) | [dataloader](./format/x-ego_v0.1/dataloader.py) |

---

## Xspark Data Format v1.0

> Xspark Data Format v1.0 merges all modalities into a single HDF5 file to unify data across tasks and embodiments. Images are stored with JPEG compression, and the format supports single/dual-arm setups, two-finger grippers, dexterous hands, and MANO-based hand estimation (X-Ego), enabling cross-embodiment consistency.<br>
> Xspark Data Format v1.0 将所有模态数据合并到单个 HDF5 文件中，实现跨任务、跨本体的统一数据表达；图像采用 JPEG 压缩存储，并支持单双臂、二指夹爪、灵巧手与基于 MANO 的手部估计（X-Ego），从而实现跨本体一致的数据组织。

- 📄 **Data Format Document**: [data format document](./format/xspark_data_format_v1.0/README.md)
- 🧩 **Data Loader Code**: [dataloader](./format/xspark_data_format_v1.0/dataloader.py)

---

## X-Ego Data Format v0.1

> X-Ego Data Format v0.1 is designed for video-based X-Ego data stored in multiple files: visual observations are saved as MP4 videos at different resolutions, and additional JSON files store hand predictions and captions.<br> 
> X-Ego Data Format v0.1 面向 Video-based 的 X-Ego 数据，采用多文件存储：不同分辨率的视觉观测以 MP4 文件保存，并使用额外的 JSON 文件存储手部预测与 caption 等信息。

- 📄 **Data Format Document**: [data format document](./format/x-ego_v0.1/README.md)
- 🧩 **Data Loader Code**: [dataloader](./format/x-ego_v0.1/dataloader.py)
