from __future__ import annotations

from pathlib import Path

import numpy as np

from gvhmr_gmr.pipeline.obs_builder import ReferenceFrame

ANCHOR_BODY_INDEX = 0


def load_reference_frames_from_npz(npz_path: str | Path, target_fps: float = 50.0) -> list[ReferenceFrame]:
    motion = np.load(npz_path)
    joint_pos = motion["joint_pos"].astype(np.float32)
    joint_vel = motion["joint_vel"].astype(np.float32)
    body_pos = motion["body_pos_w"].astype(np.float32)
    body_quat = motion["body_quat_w"].astype(np.float32)
    src_fps = float(motion["fps"])

    frames: list[ReferenceFrame] = []
    for i in range(len(joint_pos)):
        frames.append(
            ReferenceFrame(
                ref_joint_pos=joint_pos[i],
                ref_joint_vel=joint_vel[i],
                anchor_pos_w=body_pos[i, ANCHOR_BODY_INDEX],
                anchor_quat_wxyz=body_quat[i, ANCHOR_BODY_INDEX],
                fps=src_fps,
                source="npz",
            )
        )

    if abs(src_fps - target_fps) < 1e-6:
        return frames

    # Resample via time interpolation (reuse GMRMotionAdapter logic inline).
    from gvhmr_gmr.pipeline.gmr_adapter import GMRMotion, GMRMotionAdapter

    adapter = GMRMotionAdapter.__new__(GMRMotionAdapter)
    adapter.target_fps = target_fps
    adapter.motion = GMRMotion(
        fps=src_fps,
        root_pos=body_pos[:, ANCHOR_BODY_INDEX],
        root_rot_xyzw=_wxyz_batch_to_xyzw(body_quat[:, ANCHOR_BODY_INDEX]),
        dof_pos=joint_pos,
    )
    return adapter.build_reference_frames()


def _wxyz_batch_to_xyzw(quats_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quats_wxyz, dtype=np.float32)
    return q[:, [1, 2, 3, 0]]
