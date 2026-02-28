<h1 align="center">X-Ego Data Format v0.1</h1>

> X-Ego Format v0.1 is designed for video-based X-Ego data stored in multiple files: visual observations are saved as MP4 videos at different resolutions, and additional JSON files store hand predictions and captions.<br> 
> X-Ego Format v0.1 面向 Video-based 的 X-Ego 数据，采用多文件存储：不同分辨率的视觉观测以 MP4 文件保存，并使用额外的 JSON 文件存储手部预测与 caption 等信息。

This directory contains multimodal data for a single Episode, including multi-view videos, action annotations, hand pose recognition, and demo videos. Below is a detailed description of the main files.

### 1. Multi-view Camera Videos

Contains raw videos from 3 different camera views, each with two downsampled resolution versions.

| Filename                      | Camera View        | Resolution | Description                                       |
| :---------------------------- | :----------------- | :--------- | :------------------------------------------------ |
| `cam_head.mp4`                | Head Camera        | 1920x1080  | Raw video stream from head-mounted camera         |
| `cam_head_640_360.mp4`        | Head Camera        | 640x360    | Downsampled video from head-mounted camera        |
| `cam_head_320_240.mp4`        | Head Camera        | 320x240    | Downsampled video from head-mounted camera        |
| `cam_left_wrist.mp4`          | Left Wrist Camera  | 1920x1080  | Raw video stream from left wrist-mounted camera   |
| `cam_left_wrist_640_360.mp4`  | Left Wrist Camera  | 640x360    | Downsampled video from left wrist-mounted camera  |
| `cam_left_wrist_320_240.mp4`  | Left Wrist Camera  | 320x240    | Downsampled video from left wrist-mounted camera  |
| `cam_right_wrist.mp4`         | Right Wrist Camera | 1920x1080  | Raw video stream from right wrist-mounted camera  |
| `cam_right_wrist_640_360.mp4` | Right Wrist Camera | 640x360    | Downsampled video from right wrist-mounted camera |
| `cam_right_wrist_320_240.mp4` | Right Wrist Camera | 320x240    | Downsampled video from right wrist-mounted camera |

### 2. Annotation Files

| Filename          | Description                                                  |
| :---------------- | :----------------------------------------------------------- |
| `caption.json`    | **Atomic action segments and language annotations**. Contains timestamp-based segmentation of video clips with corresponding natural language descriptions. |
| `hand_poses.json` | **Hand pose recognition data**. Records hand keypoint detection results for each frame. **Coordinates are in camera coordinate system** |

### 3. Visualization & Demo

| Filename              | Resolution | Description                                                  |
| :-------------------- | :--------- | :----------------------------------------------------------- |
| `hand_poses.mp4`      | 640x360    | **Hand pose visualization video**. Visualizes hand pose recognition results from `hand_poses.json` overlaid on the original video. |
| `preview_640_360.mp4` | 640x540    | **Data preview video (Demo)**. A stitched preview video composed of: (1) a 640x360 head camera view, and (2) two 320x180 wrist camera views (left and right) arranged side-by-side. Designed for quick multi-view preview of the episode content. |
