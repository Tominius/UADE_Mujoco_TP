from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.policy_discovery import PolicyBundle


def npz_frame_to_qpos(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    joint_pos: np.ndarray,
    frame: int,
    nq: int,
    *,
    anchor_index: int = 0,
) -> np.ndarray:
    """Build MuJoCo qpos from G1 Moves NPZ (pelvis anchor + 29 joints)."""
    q = np.zeros(nq, dtype=np.float64)
    q[:3] = body_pos_w[frame, anchor_index]
    q[3:7] = body_quat_w[frame, anchor_index]
    n_j = min(29, nq - 7)
    q[7 : 7 + n_j] = joint_pos[frame, :n_j]
    return q


def load_qpos_from_npz(npz_path: str | Path, nq: int) -> tuple[np.ndarray, float]:
    """Load full qpos trajectory and fps from policy training NPZ."""
    motion = np.load(npz_path)
    joint_pos = motion["joint_pos"]
    body_pos_w = motion["body_pos_w"]
    body_quat_w = motion["body_quat_w"]
    fps = float(motion["fps"])
    num_frames = int(joint_pos.shape[0])
    qpos = np.stack(
        [
            npz_frame_to_qpos(body_pos_w, body_quat_w, joint_pos, i, nq)
            for i in range(num_frames)
        ],
        axis=0,
    )
    return qpos, fps


def load_qpos_from_bundle(bundle: PolicyBundle, nq: int) -> tuple[np.ndarray, float]:
    return load_qpos_from_npz(bundle.npz_path, nq)
