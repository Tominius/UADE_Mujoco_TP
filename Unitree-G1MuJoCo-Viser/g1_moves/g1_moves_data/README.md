---
license: cc-by-4.0
task_categories:
  - robotics
  - reinforcement-learning
tags:
  - motion-capture
  - humanoid-robot
  - unitree-g1
  - bvh
  - sim-to-real
size_categories:
  - n<1K
---

<p align="center">
  <img src="logo.png" alt="Experiential Technologies" width="500">
</p>

# G1 Moves

> **Dataset** (you are here) · **[Showcase](https://huggingface.co/spaces/exptech/g1-moves)** (interactive gallery) · **[Code](https://github.com/experientialtech/g1-moves)** (scripts & docs)

60 motion capture clips for the Unitree G1 humanoid robot (edition EDU, 29 DOF), captured from real performers in Austin, TX using [MOVIN TRACIN](https://movin3d.com/) markerless motion capture and [video2robot](https://github.com/experientialtech/video2robot) monocular video extraction. Each clip is provided at multiple pipeline stages: raw mocap (BVH/FBX), retargeted robot joint trajectories (PKL), processed RL training data (NPZ), and trained ONNX policies with baked-in observation normalization. All 60 clips have trained RL policies and ONNX exports ready for deployment.

## Dataset Summary

| | |
|---|---|
| **Total clips** | 60 |
| **Total duration** | 29.6 minutes (106,434 frames at 60 FPS) |
| **Categories** | Dance (28), Karate (27), Bonus (5) |
| **Clip duration** | 6.5s - 119.5s |
| **Robot** | Unitree G1, edition EDU, 29 DOF |
| **Capture system** | MOVIN TRACIN (markerless, LiDAR + vision) |
| **Retarget SDK** | [movin_sdk_python](https://github.com/MOVIN3D/movin_sdk_python) |

## Supported Tasks

- **Motion imitation RL**: Train policies to track reference motions on the G1 robot using the NPZ training data
- **Sim-to-real transfer**: Deploy trained policies from MuJoCo simulation to physical G1 hardware
- **Motion retargeting research**: Study human-to-robot motion transfer using the BVH-to-PKL pipeline
- **Animation / visualization**: Import FBX files into Blender, Maya, Unreal Engine, or Unity

## Dataset Structure

Each clip lives in its own subfolder organized by pipeline stage:

```
<category>/<clip>/
  capture/                Original motion capture data
    <clip>.bvh            BVH motion capture (51-joint humanoid skeleton)
    <clip>.gif            Preview animation
    <clip>.mp4            Preview video
    <clip>_bl.fbx         FBX for Blender
    <clip>_mb.fbx         FBX for Maya
    <clip>_ue.fbx         FBX for Unreal Engine
    <clip>_un.fbx         FBX for Unity
  retarget/               Retargeted G1 joint trajectories
    <clip>.pkl            Retargeted joint angles (29 DOF)
    <clip>.csv            Joint angles as CSV (no header)
    <clip>_retarget.gif   Retarget preview animation
    <clip>_retarget.mp4   Retarget preview video
  training/               Processed RL training data
    <clip>.npz            Training data with forward kinematics
    <clip>_training.gif   Training visualization
    <clip>_training.mp4   Training visualization video
  policy/                 Trained RL policy
    <clip>_policy.pt      PyTorch checkpoint
    <clip>_policy.onnx    ONNX export (baked-in observation normalization)
    <clip>_policy.gif     Policy rollout animation
    <clip>_policy.mp4     Policy rollout video
    agent.yaml            PPO hyperparameters
    env.yaml              Full environment configuration
    training_log.csv      Training metrics (rewards, losses, errors)
```

## File Format Reference

### BVH (capture)

Standard BVH motion capture format with a 51-joint humanoid skeleton.

- **Root**: Hips (6 DOF: XYZ position + YXZ Euler rotation)
- **Joints**: 3 DOF each (YXZ Euler rotation order)
- **Frame rate**: 60 FPS
- **Coordinate system**: Y-up

### PKL (retarget)

Python pickle containing a dict with retargeted G1 joint trajectories:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `fps` | scalar | int | Frame rate (60) |
| `root_pos` | (N, 3) | float64 | Root position in world frame (meters) |
| `root_rot` | (N, 4) | float64 | Root orientation as quaternion (xyzw) |
| `dof_pos` | (N, 29) | float64 | Joint angles in radians |

Joint order (29 DOF):

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 0 | left_hip_pitch | 15 | left_shoulder_pitch |
| 1 | left_hip_roll | 16 | left_shoulder_roll |
| 2 | left_hip_yaw | 17 | left_shoulder_yaw |
| 3 | left_knee | 18 | left_elbow |
| 4 | left_ankle_pitch | 19 | left_wrist_roll |
| 5 | left_ankle_roll | 20 | left_wrist_pitch |
| 6 | right_hip_pitch | 21 | left_wrist_yaw |
| 7 | right_hip_roll | 22 | right_shoulder_pitch |
| 8 | right_hip_yaw | 23 | right_shoulder_roll |
| 9 | right_knee | 24 | right_shoulder_yaw |
| 10 | right_ankle_pitch | 25 | right_elbow |
| 11 | right_ankle_roll | 26 | right_wrist_roll |
| 12 | waist_yaw | 27 | right_wrist_pitch |
| 13 | waist_roll | 28 | right_wrist_yaw |
| 14 | waist_pitch | | |

### CSV (retarget)

Same data as PKL in plain CSV format (no header row). 36 columns:

| Columns | Content |
|---------|---------|
| 0-2 | Root position (x, y, z) |
| 3-6 | Root quaternion (x, y, z, w) |
| 7-35 | Joint angles (29 DOF, same order as PKL) |

### NPZ (training)

NumPy compressed archive with forward kinematics computed from the retargeted motion. Used directly as RL training reference.

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `fps` | (1,) | float64 | Frame rate (60) |
| `joint_pos` | (N, 29) | float32 | Joint positions (radians) |
| `joint_vel` | (N, 29) | float32 | Joint velocities (rad/s) |
| `body_pos_w` | (N, 30, 3) | float32 | Body positions in world frame (meters) |
| `body_quat_w` | (N, 30, 4) | float32 | Body orientations as quaternions |
| `body_lin_vel_w` | (N, 30, 3) | float32 | Body linear velocities (m/s) |
| `body_ang_vel_w` | (N, 30, 3) | float32 | Body angular velocities (rad/s) |

N = BVH frames - 1 (velocity requires finite differences).

### FBX (capture)

Four platform-optimized FBX variants per clip:

| Suffix | Target |
|--------|--------|
| `_bl.fbx` | Blender |
| `_mb.fbx` | Maya |
| `_ue.fbx` | Unreal Engine |
| `_un.fbx` | Unity |

## Usage Examples

### Load retargeted motion (PKL)

```python
import pickle
import numpy as np

with open("dance/B_DadDance/retarget/B_DadDance.pkl", "rb") as f:
    motion = pickle.load(f)

print(f"FPS: {motion['fps']}")
print(f"Duration: {motion['dof_pos'].shape[0] / motion['fps']:.1f}s")
print(f"Root position at frame 0: {motion['root_pos'][0]}")
print(f"Joint angles shape: {motion['dof_pos'].shape}")  # (2509, 29)
```

### Load training data (NPZ)

```python
import numpy as np

data = np.load("dance/B_DadDance/training/B_DadDance.npz")

joint_pos = data["joint_pos"]    # (2508, 29)
joint_vel = data["joint_vel"]    # (2508, 29)
body_pos = data["body_pos_w"]    # (2508, 30, 3)
body_quat = data["body_quat_w"]  # (2508, 30, 4)

# Get pelvis height over time
pelvis_z = body_pos[:, 0, 2]
print(f"Pelvis height: {pelvis_z.min():.3f} - {pelvis_z.max():.3f} m")
```

### Filter clips by duration or difficulty

```python
import json

with open("manifest.json") as f:
    manifest = json.load(f)

# Find clips longer than 30 seconds
long_clips = {
    name: clip for name, clip in manifest["clips"].items()
    if clip["duration_s"] > 30
}
print(f"{len(long_clips)} clips > 30s")

# Sort by motion energy (difficulty proxy)
by_energy = sorted(
    manifest["clips"].items(),
    key=lambda x: x[1]["motion_stats"]["mean_joint_velocity"],
    reverse=True,
)
print("Most energetic:", by_energy[0][0])
print("Least energetic:", by_energy[-1][0])
```

## Data Collection

59 clips were captured using the [MOVIN TRACIN](https://movin3d.com/) markerless motion capture system, with 1 additional clip extracted from monocular video via video2robot. MOVIN TRACIN uses on-device AI to fuse LiDAR point clouds and vision into motion data without markers, suits, or multi-camera rigs. Performances were recorded and exported using [MOVIN Studio](https://www.movin3d.com/studio).

### Performers

| Prefix | Performer | Clips |
|--------|-----------|-------|
| `B_` | [Mitch Chaiet](https://mitchchaiet.com/) | Bonus clips + some dance |
| `J_` | [Jasmine Coro](https://jasminecoro.com/) | Dance choreography |
| `M_` | [Mike Gassaway](https://www.backstage.com/u/mike-gassaway/) | Karate / martial arts |
| `V_` | [Maya Coro](https://www.backstage.com/u/maya-coro/) | Fencing (via video2robot) |

### Processing Pipeline

1. **Capture**: MOVIN TRACIN records performer motion as BVH + FBX
2. **Retarget**: [movin_sdk_python](https://github.com/MOVIN3D/movin_sdk_python) maps human skeleton to G1 joint limits (1.75m human height)
3. **Ground calibration**: MuJoCo forward kinematics finds minimum foot Z, shifts root for ground contact
4. **Training data**: MuJoCo computes full-body forward kinematics (positions, orientations, velocities)
5. **RL training**: PPO with motion imitation rewards in MuJoCo-Warp (8,192 parallel envs)

## Metadata Files

| File | Description |
|------|-------------|
| `manifest.json` | Machine-readable index of all 60 clips with per-clip metadata |
| `quality_report.json` | Automated validation (joint limits, ground penetration, frame consistency) |
| `generate_metadata.py` | Script to regenerate all metadata from source data |

## Training Quality

All 60 clips have trained RL policies and ONNX exports. Aggregate metrics across all trained policies:

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Reward | 40.1 | 20.6 | 45.6 |
| Episode length | 493 / 500 | 419 | 500 |
| Body position error | 0.044 m | 0.030 m | 0.108 m |
| Joint position error | 0.54 rad | 0.39 rad | 0.93 rad |

Each policy directory contains a `training_log.csv` with per-iteration metrics (rewards, losses, tracking errors, termination rates). ONNX exports include baked-in observation normalization (160 obs → [512, 256, 128] → 29 actions) for direct deployment without the training framework.

## Standalone Inference

Run any trained policy in MuJoCo simulation without the full training framework:

```bash
pip install mujoco onnxruntime numpy
python run_policy.py dance/B_DadDance --xml /path/to/g1_mode15_square.xml
```

See `run_policy.py` for the complete observation vector construction and control loop.

## Citation

```bibtex
@misc{g1moves2026,
  title     = {G1 Moves: Motion Capture Dataset for the Unitree G1 Humanoid Robot},
  author    = {Chaiet, Mitch},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/exptech/g1-moves},
  note      = {60 motion capture clips with retargeted joint trajectories, RL training data, and trained policies}
}
```

## Equipment

### Motion Capture

59 clips were captured using the [MOVIN TRACIN](https://movin3d.com/) markerless motion capture system from [MOVIN3D](https://movin3d.com/), with 1 additional clip (V_Rocamena) extracted from monocular video via [video2robot](https://github.com/experientialtech/video2robot). MOVIN TRACIN uses on-device AI to fuse LiDAR point clouds and vision into production-ready motion data — no markers, no suit, no multi-camera rig. Captured performances were recorded and exported using [MOVIN Studio](https://www.movin3d.com/studio), which provides real-time skeleton visualization, recording management, and export to BVH and FBX formats. Retargeting from human skeleton to G1 robot joint space was performed using [movin_sdk_python](https://github.com/MOVIN3D/movin_sdk_python).

Thank you to [MOVIN3D](https://movin3d.com/) for building an incredible motion capture platform that makes professional-grade mocap accessible to robotics researchers.

### Workstation

All data was captured and policies were trained on two machines from [Dell Technologies](https://www.dell.com/):

#### [Dell Pro Max Tower T2](https://www.dell.com/en-us/shop/desktop-computers/dell-pro-max-tower-t2/spd/dell-pro-max-tower-t2-desktop/usepmtbts2_t2)

| Component | Spec |
|-----------|------|
| CPU | Intel Core Ultra 9 285K (24 cores, up to 7.2 GHz) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB GDDR7) |
| RAM | 128 GB DDR5 |
| Storage | 2x 4 TB WD SN8000S NVMe SSD (8 TB total) |
| OS | Ubuntu 24.04 LTS |

The RTX PRO 6000 Blackwell with 96 GB of VRAM enables running 8,192 parallel MuJoCo-Warp simulation environments on a single GPU for reinforcement learning training, while the 24-core Ultra 9 285K handles motion retargeting and data processing.

#### [Dell Pro Max with GB10](https://www.dell.com/en-us/shop/desktop-computers/dell-pro-max-desktop-with-nvidia-gb10/spd/dell-pro-max-gb10-desktop/usepmgb10bts)

| Component | Spec |
|-----------|------|
| SoC | NVIDIA GB10 Grace Blackwell Superchip |
| CPU | NVIDIA Grace (20x ARM Cortex-X925) |
| GPU | NVIDIA Blackwell GPU (1,024 CUDA cores, 120 GB unified memory) |
| RAM | 120 GB LPDDR5X unified (shared CPU/GPU, 273 GB/s) |
| Storage | 4 TB NVMe SSD |
| AI Performance | Up to 1,000 TOPS (INT4) |
| OS | Ubuntu 24.04 LTS (NVIDIA DGX OS 7.3.1) |

The Dell Pro Max with GB10 is a compact desktop AI supercomputer powered by the NVIDIA GB10 Grace Blackwell Superchip. Its unified memory architecture allows the GPU to access the full 120 GB memory pool without PCIe bottlenecks, running 4,096 parallel MuJoCo-Warp environments for secondary training workloads. Both machines train policies simultaneously from opposite ends of the clip queue.

Thank you to [Dell Technologies](https://www.dell.com/) for providing the compute power behind this project.

## License

CC-BY-4.0

## Acknowledgements

- [MOVIN3D](https://movin3d.com/) for the MOVIN TRACIN capture system and movin_sdk_python retargeting SDK
- [Dell Technologies](https://www.dell.com/) for the Pro Max Tower T2 and Pro Max GB10 workstations used for capture and training
- [Unitree Robotics](https://www.unitree.com/) for the G1 humanoid robot platform
