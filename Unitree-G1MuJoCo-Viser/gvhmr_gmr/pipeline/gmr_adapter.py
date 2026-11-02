from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gvhmr_gmr.pipeline.obs_builder import ReferenceFrame, quat_xyzw_to_wxyz

ANCHOR_BODY_INDEX = 0


@dataclass
class GMRMotion:
    fps: float
    root_pos: np.ndarray  # (T, 3)
    root_rot_xyzw: np.ndarray  # (T, 4) — GMR saved pickle uses xyzw
    dof_pos: np.ndarray  # (T, 29)


class GMRMotionAdapter:
    """Load GMR / retarget pickle, resample to policy fps, build ReferenceFrame list."""

    def __init__(self, motion_path: str | Path, target_fps: float = 50.0) -> None:
        self.motion_path = Path(motion_path)
        self.target_fps = float(target_fps)
        if not self.motion_path.exists():
            raise FileNotFoundError(f"GMR motion not found: {self.motion_path}")
        self.motion = self._load_motion(self.motion_path)

    @staticmethod
    def _load_motion(path: Path) -> GMRMotion:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"expected dict pickle, got {type(data)}")

        fps = float(data["fps"])
        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_rot = np.asarray(data["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError("root_pos must have shape (T, 3)")
        if root_rot.ndim != 2 or root_rot.shape[1] != 4:
            raise ValueError("root_rot must have shape (T, 4)")
        if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
            raise ValueError("dof_pos must have shape (T, 29)")
        if not (len(root_pos) == len(root_rot) == len(dof_pos)):
            raise ValueError("root_pos/root_rot/dof_pos length mismatch")

        return GMRMotion(fps=fps, root_pos=root_pos, root_rot_xyzw=root_rot, dof_pos=dof_pos)

    @staticmethod
    def _interp_array(arr: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return np.interp(t_tgt, t_src, arr).astype(np.float32)
        cols = [np.interp(t_tgt, t_src, arr[:, i]) for i in range(arr.shape[1])]
        return np.stack(cols, axis=-1).astype(np.float32)

    def resample(self) -> GMRMotion:
        src_fps = self.motion.fps
        if abs(src_fps - self.target_fps) < 1e-6:
            return self.motion

        t_src = np.arange(len(self.motion.dof_pos), dtype=np.float32) / src_fps
        t_tgt = np.arange(0.0, t_src[-1] + 1e-6, 1.0 / self.target_fps, dtype=np.float32)
        dof_pos = self._interp_array(self.motion.dof_pos, t_src, t_tgt)
        root_pos = self._interp_array(self.motion.root_pos, t_src, t_tgt)
        root_rot_xyzw = self._interp_array(self.motion.root_rot_xyzw, t_src, t_tgt)
        norms = np.linalg.norm(root_rot_xyzw, axis=1, keepdims=True)
        root_rot_xyzw = root_rot_xyzw / np.clip(norms, 1e-8, None)

        return GMRMotion(
            fps=self.target_fps,
            root_pos=root_pos,
            root_rot_xyzw=root_rot_xyzw,
            dof_pos=dof_pos,
        )

    def build_reference_frames(self) -> list[ReferenceFrame]:
        motion = self.resample()
        dt = 1.0 / motion.fps
        ref_joint_vel = np.gradient(motion.dof_pos, dt, axis=0).astype(np.float32)

        frames: list[ReferenceFrame] = []
        for i in range(len(motion.dof_pos)):
            frames.append(
                ReferenceFrame(
                    ref_joint_pos=motion.dof_pos[i],
                    ref_joint_vel=ref_joint_vel[i],
                    anchor_pos_w=motion.root_pos[i],
                    anchor_quat_wxyz=quat_xyzw_to_wxyz(motion.root_rot_xyzw[i]),
                    fps=motion.fps,
                    source="gmr",
                )
            )
        return frames

    def to_qpos_sequence(self, nq: int) -> tuple[np.ndarray, float]:
        """Convert retarget motion to MuJoCo qpos for stable Viser playback."""
        motion = self.resample()
        qpos = np.zeros((len(motion.dof_pos), nq), dtype=np.float64)
        for i in range(len(motion.dof_pos)):
            qpos[i, :3] = motion.root_pos[i]
            qpos[i, 3:7] = quat_xyzw_to_wxyz(motion.root_rot_xyzw[i])
            qpos[i, 7:36] = motion.dof_pos[i]
        return qpos, motion.fps
