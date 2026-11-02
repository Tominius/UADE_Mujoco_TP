from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionStats:
    intensity_raw: float
    intensity_norm: float
    intensity_level: str
    num_frames: int
    duration_sec: float
    root_path_m: float
    joint_delta_mean: float


def _joint_view(qpos: np.ndarray) -> np.ndarray:
    if qpos.ndim != 2:
        raise ValueError(f"qpos 必须是二维数组，得到 shape={qpos.shape}")
    if qpos.shape[1] < 8:
        raise ValueError(f"qpos 列数过小，无法分离 floating base，得到 shape={qpos.shape}")
    return qpos[:, 7:]


def _root_xyz_view(qpos: np.ndarray) -> np.ndarray:
    return qpos[:, :3]


def compute_intensity_raw(qpos: np.ndarray) -> float:
    """
    计算动作原始强度。

    定义:
        I_raw = 1/(T-1) * sum_t || q_joint[t] - q_joint[t-1] ||_2

    注意:
        只统计根节点之后的关节自由度，不把 root quaternion 直接拿来做线性差分度量。
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    if len(qpos) < 2:
        return 0.0
    joint_qpos = _joint_view(qpos)
    diff = joint_qpos[1:] - joint_qpos[:-1]
    step_norm = np.linalg.norm(diff, axis=1)
    return float(np.mean(step_norm))


def compute_root_path_m(qpos: np.ndarray) -> float:
    """计算根节点累计平移长度（米）。"""
    qpos = np.asarray(qpos, dtype=np.float64)
    if len(qpos) < 2:
        return 0.0
    xyz = _root_xyz_view(qpos)
    diff = xyz[1:] - xyz[:-1]
    return float(np.sum(np.linalg.norm(diff, axis=1)))


def compute_joint_delta_mean(qpos: np.ndarray) -> float:
    """计算所有关节维度上一阶绝对差分的均值，用于辅助排查"看起来没动"的动作。"""
    qpos = np.asarray(qpos, dtype=np.float64)
    if len(qpos) < 2:
        return 0.0
    diff = np.abs(_joint_view(qpos)[1:] - _joint_view(qpos)[:-1])
    return float(np.mean(diff))


def intensity_level(raw_value: float, thresholds: tuple[float, float]) -> str:
    """
    用数据集分位阈值分类：
        <= q33 → 低
        <= q66 → 中
        > q66 → 高
    """
    low_hi, mid_hi = thresholds
    if raw_value <= low_hi:
        return "低"
    if raw_value <= mid_hi:
        return "中"
    return "高"


def make_motion_stats(
    qpos: np.ndarray,
    fps: float,
    thresholds: tuple[float, float],
    raw_min: float,
    raw_max: float,
) -> MotionStats:
    """
    将单条动作轨迹汇总为 MotionStats。

    输入:
        qpos: shape=(T, nq)
        fps: 动作采样帧率
        thresholds: (q33, q66)
        raw_min/raw_max: 全动作库原始强度范围
    输出:
        MotionStats
    异常:
        ValueError: 维度异常
    """
    raw = compute_intensity_raw(qpos)
    denom = max(raw_max - raw_min, 1e-12)
    norm = float((raw - raw_min) / denom)
    return MotionStats(
        intensity_raw=raw,
        intensity_norm=float(np.clip(norm, 0.0, 1.0)),
        intensity_level=intensity_level(raw, thresholds),
        num_frames=int(qpos.shape[0]),
        duration_sec=float(qpos.shape[0] / fps),
        root_path_m=compute_root_path_m(qpos),
        joint_delta_mean=compute_joint_delta_mean(qpos),
    )
