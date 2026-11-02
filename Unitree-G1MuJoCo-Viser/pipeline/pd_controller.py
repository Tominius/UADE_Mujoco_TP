from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

# Joint order (29): L leg×6, R leg×6, waist×3, L arm×7, R arm×7
NUM_JOINTS = 29


# Official per-joint PD from g1-moves run_policy.py (matches mjlab training).
OFFICIAL_KP = np.array(
    [
        40.2, 99.1, 40.2, 99.1, 28.6, 28.6,
        40.2, 99.1, 40.2, 99.1, 28.6, 28.6,
        40.2, 28.6, 28.6,
        14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
        14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
    ],
    dtype=np.float32,
)

OFFICIAL_KD = np.array(
    [
        2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
        2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
        2.6, 1.8, 1.8,
        0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
        0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
    ],
    dtype=np.float32,
)


def build_v2_pd_gains() -> tuple[np.ndarray, np.ndarray]:
    """Per-joint PD gains for Policy Stability Fix v2."""
    kp = np.zeros(NUM_JOINTS, dtype=np.float32)
    kd = np.zeros(NUM_JOINTS, dtype=np.float32)
    kp[:12] = 80.0
    kd[:12] = 8.0
    kp[12:15] = 60.0
    kd[12:15] = 6.0
    kp[15:] = 40.0
    kd[15:] = 4.0
    return kp, kd


def build_official_pd_gains() -> tuple[np.ndarray, np.ndarray]:
    return OFFICIAL_KP.copy(), OFFICIAL_KD.copy()


def compute_pd_torques(
    target_pos: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    *,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    model: mujoco.MjModel | None = None,
    use_official_gains: bool = False,
) -> np.ndarray:
    if kp is None or kd is None:
        if use_official_gains:
            kp, kd = build_official_pd_gains()
        else:
            kp, kd = build_v2_pd_gains()
    jp = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
    jv = np.asarray(joint_vel, dtype=np.float32).reshape(-1)
    tp = np.asarray(target_pos, dtype=np.float32).reshape(-1)
    torques = kp * (tp - jp) - kd * jv
    if model is not None and model.nu >= NUM_JOINTS:
        for i in range(NUM_JOINTS):
            lo, hi = model.actuator_ctrlrange[i]
            torques[i] = np.clip(torques[i], lo, hi)
    return torques


def apply_pd_targets(
    data,
    target_pos: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    *,
    use_position_actuators: bool,
    model: mujoco.MjModel | None = None,
    use_official_gains: bool = False,
) -> None:
    tp = np.asarray(target_pos, dtype=np.float32)
    if use_position_actuators:
        data.ctrl[:NUM_JOINTS] = tp
    else:
        data.ctrl[:NUM_JOINTS] = compute_pd_torques(
            tp, joint_pos, joint_vel, model=model, use_official_gains=use_official_gains
        )
