<h1 align="center">Xspark Data Format v1.0</h1>

> Xspark Data Format v1.0 merges all modalities into a single HDF5 file to unify data across tasks and embodiments. Images are stored with JPEG compression, and the format supports single/dual-arm setups, two-finger grippers, dexterous hands, and MANO-based hand estimation (X-Ego), enabling cross-embodiment consistency.<br>
> Xspark Data Format v1.0 将所有模态数据合并到单个 HDF5 文件中，实现跨任务、跨本体的统一数据表达；图像采用 JPEG 压缩存储，并支持单双臂、二指夹爪、灵巧手与基于 MANO 的手部估计（X-Ego），从而实现跨本体一致的数据组织。

This format use a single HDF5 file per episode to encapsulate multi-view visuals, robot proprioception, and high-level semantics.

---

### 1. Structure Overview

The data is organized into three main functional segments: `vision`, `state`, and semantic metadata.

| Category | Key/Path | Type | Description |
| :--- | :--- | :--- | :--- |
| **Vision** | `vision/` | Group | Synchronized camera streams (RGB + shapes). |
| **State** | `state/` | Group | Proprioception like joint angles and EE poses. |
| **Semantics**| `instructions` | Dataset | Task instruction list (JSON-serialized string). |
| **Semantics**| `subtasks` | Dataset | Atomic subtask sequences (JSON-serialized string). |
| **Meta** | `additional_info/`| Group | Contextual metadata (e.g., control frequency). |
| **Meta** | `data_format_version`| Dataset| Format version identifier (e.g., "v1.0"). |

---

### 2. Vision Data (`vision/`)

Visual streams are stored as arrays of JPEG-compressed byte buffers.

| Dataset Path | Shape | Description |
| :--- | :--- | :--- |
| `cam_head/colors` | (N, byte_array) | Head-mounted RGB stream (List of JPEG bytes). |
| `cam_head/shape` | (3,) | Decoded resolution of the head camera [H, W, C]. |
| `cam_left_wrist/colors` | (N, byte_array) | Left wrist RGB stream (List of JPEG bytes). |
| `cam_left_wrist/shape` | (3,) | Decoded resolution of the left wrist camera. |
| `cam_right_wrist/colors`| (N, byte_array) | Right wrist RGB stream (List of JPEG bytes). |
| `cam_right_wrist/shape` | (3,) | Decoded resolution of the right wrist camera. |

---

### 3. State Data (`state/`)

Records synchronized joint positions and end-effector cartesian poses.

| Dataset Path | Shape | Description |
| :--- | :--- | :--- |
| `left_arm_joint_states` | (N, DOF) | Left arm joint angles (Radians). |
| `left_ee_joint_states` | (N, 1) | Left gripper/actuator opening state. |
| `left_ee_poses` | (N, 7) | Left EE pose in world frame (XYZ + Quat). |
| `right_arm_joint_states`| (N, DOF) | Right arm joint angles (Radians). |
| `right_ee_joint_states` | (N, 1) | Right gripper/actuator opening state. |
| `right_ee_poses` | (N, 7) | Right EE pose in world frame (XYZ + Quat). |

---

### 4. Global Metadata

| Path | Value Example | Description |
| :--- | :--- | :--- |
| `additional_info/frequency` | `30` | Recording and control loop frequency (Hz). |
| `instructions` | `["Fold the white towel."]` | Human language instruction for the episode. |
| `subtasks` | `[[[0, 600], "Flatten the clothes."], [[601, 1000], "Fold the clothes."]]` | List of discrete sub-steps for long-horizon tasks. |
| `data_format_version` | `"v1.0"` | Current schema version. |
