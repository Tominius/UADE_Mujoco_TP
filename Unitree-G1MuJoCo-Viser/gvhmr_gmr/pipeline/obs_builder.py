from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_JOINT_POS = np.zeros(29, dtype=np.float32)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    return q[[3, 0, 1, 2]]


def quat_to_rot_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_6d(rot_matrix: np.ndarray) -> np.ndarray:
    return rot_matrix[:, :2].T.reshape(-1).astype(np.float32)


def transform_anchor_to_body_frame(
    robot_pos_w: np.ndarray,
    robot_quat_wxyz: np.ndarray,
    anchor_pos_w: np.ndarray,
    anchor_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r_robot = quat_to_rot_matrix(robot_quat_wxyz)
    delta_pos = np.asarray(anchor_pos_w, dtype=np.float32) - np.asarray(robot_pos_w, dtype=np.float32)
    anchor_pos_b = r_robot.T @ delta_pos
    r_anchor = quat_to_rot_matrix(anchor_quat_wxyz)
    r_rel = r_robot.T @ r_anchor
    anchor_ori_b = rotation_matrix_to_6d(r_rel)
    return anchor_pos_b.astype(np.float32), anchor_ori_b.astype(np.float32)


@dataclass
class SimState:
    robot_pos_w: np.ndarray
    robot_quat_wxyz: np.ndarray
    base_ang_vel_b: np.ndarray
    base_lin_vel_b: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    last_action: np.ndarray


@dataclass
class ReferenceFrame:
    ref_joint_pos: np.ndarray
    ref_joint_vel: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_wxyz: np.ndarray
    fps: float
    source: str = "npz"


@dataclass
class ObsBuildResult:
    obs160: np.ndarray
    valid: bool
    warnings: list[str] = field(default_factory=list)
    source: str = "npz"


class ObsBuilder160:
    """Build obs160 — order matches g1-moves run_policy.py (no external normalization)."""

    def __init__(self, default_joint_pos: np.ndarray | None = None) -> None:
        self.default_joint_pos = (
            np.asarray(default_joint_pos, dtype=np.float32)
            if default_joint_pos is not None
            else DEFAULT_JOINT_POS.copy()
        )
        if self.default_joint_pos.shape != (29,):
            raise ValueError("default_joint_pos must be shape (29,)")

    def build(self, sim: SimState, ref: ReferenceFrame) -> ObsBuildResult:
        warnings: list[str] = []

        def _shape(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape != shape:
                raise ValueError(f"{name} shape mismatch: expected {shape}, got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains NaN/Inf")
            return arr

        robot_pos_w = _shape("robot_pos_w", sim.robot_pos_w, (3,))
        robot_quat_wxyz = _shape("robot_quat_wxyz", sim.robot_quat_wxyz, (4,))
        base_ang_vel_b = _shape("base_ang_vel_b", sim.base_ang_vel_b, (3,))
        base_lin_vel_b = _shape("base_lin_vel_b", sim.base_lin_vel_b, (3,))
        joint_pos = _shape("joint_pos", sim.joint_pos, (29,))
        joint_vel = _shape("joint_vel", sim.joint_vel, (29,))
        last_action = _shape("last_action", sim.last_action, (29,))
        ref_joint_pos = _shape("ref_joint_pos", ref.ref_joint_pos, (29,))
        ref_joint_vel = _shape("ref_joint_vel", ref.ref_joint_vel, (29,))
        anchor_pos_w = _shape("anchor_pos_w", ref.anchor_pos_w, (3,))
        anchor_quat_wxyz = _shape("anchor_quat_wxyz", ref.anchor_quat_wxyz, (4,))

        qn = float(np.linalg.norm(robot_quat_wxyz))
        if abs(qn - 1.0) > 1e-2:
            warnings.append(f"robot_quat_wxyz not unit length: {qn:.4f}")
            robot_quat_wxyz = robot_quat_wxyz / max(qn, 1e-8)

        an = float(np.linalg.norm(anchor_quat_wxyz))
        if abs(an - 1.0) > 1e-2:
            warnings.append(f"anchor_quat_wxyz not unit length: {an:.4f}")
            anchor_quat_wxyz = anchor_quat_wxyz / max(an, 1e-8)

        anchor_pos_b, anchor_ori_b = transform_anchor_to_body_frame(
            robot_pos_w, robot_quat_wxyz, anchor_pos_w, anchor_quat_wxyz
        )

        obs = np.concatenate(
            [
                ref_joint_pos,
                ref_joint_vel,
                anchor_pos_b,
                anchor_ori_b,
                base_ang_vel_b,
                base_lin_vel_b,
                joint_pos - self.default_joint_pos,
                joint_vel,
                last_action,
            ]
        ).astype(np.float32)

        if obs.shape != (160,):
            raise RuntimeError(f"obs shape must be (160,), got {obs.shape}")
        if not np.isfinite(obs).all():
            raise RuntimeError("obs contains NaN/Inf after build")

        return ObsBuildResult(obs160=obs, valid=True, warnings=warnings, source=ref.source)
