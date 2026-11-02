from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pipeline.pd_controller import apply_pd_targets, build_v2_pd_gains, compute_pd_torques

if TYPE_CHECKING:
    import mujoco

# G1 mode-15 default joint positions (29 DOF) — training uses all zeros.
DEFAULT_JOINT_POS = np.zeros(29, dtype=np.float32)

# Per-joint action scale from mjlab G1 tracking env (B_DadDance/policy/env.yaml).
G1_ACTION_SCALE = np.array(
    [
        0.5475464629911068,
        0.35066146637882434,
        0.5475464629911068,
        0.35066146637882434,
        0.43857731392336724,
        0.43857731392336724,
        0.5475464629911068,
        0.35066146637882434,
        0.5475464629911068,
        0.35066146637882434,
        0.43857731392336724,
        0.43857731392336724,
        0.5475464629911068,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.07450087032950714,
        0.07450087032950714,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.07450087032950714,
        0.07450087032950714,
    ],
    dtype=np.float32,
)


def actions_to_joint_targets(actions: np.ndarray, *, use_mjlab_scale: bool = False) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32).reshape(-1)
    if a.shape[0] != 29:
        raise ValueError(f"expected 29 actions, got {a.shape[0]}")
    if use_mjlab_scale:
        return DEFAULT_JOINT_POS + a * G1_ACTION_SCALE
    return DEFAULT_JOINT_POS + a


def apply_joint_targets(
    data,
    target_pos: np.ndarray,
    *,
    use_position_actuators: bool,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    model: mujoco.MjModel | None = None,
) -> None:
    apply_pd_targets(
        data,
        target_pos,
        joint_pos,
        joint_vel,
        use_position_actuators=use_position_actuators,
        model=model,
    )


__all__ = [
    "DEFAULT_JOINT_POS",
    "G1_ACTION_SCALE",
    "actions_to_joint_targets",
    "apply_joint_targets",
    "apply_pd_targets",
    "build_v2_pd_gains",
    "compute_pd_torques",
]
