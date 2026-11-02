from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AmplifyConfig:
    root_xyz_scale: float = 1.0
    joint_scale: float = 1.35


def amplify_qpos(
    qpos: np.ndarray,
    config: AmplifyConfig | None = None,
    joint_scale_override: float | None = None,
) -> np.ndarray:
    """
    对动作做幅值增强。

    数学形式:
        root_xyz' = root_xyz_ref + alpha_root * (root_xyz - root_xyz_ref)
        root_quat' = root_quat  # 不做线性缩放
        joints' = joints_ref + alpha_joint * (joints - joints_ref)

    输入:
        qpos: shape=(T, nq)
        config: AmplifyConfig
        joint_scale_override: 临时覆盖关节放大量
    输出:
        amplified: shape=(T, nq)
    异常:
        ValueError: qpos 不是二维数组，或 nq < 8
    """
    cfg = config or AmplifyConfig()
    arr = np.asarray(qpos, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"qpos 必须是二维数组，得到 shape={arr.shape}")
    if arr.shape[1] < 8:
        raise ValueError(f"qpos 列数过小，无法分离 floating base，得到 shape={arr.shape}")

    out = arr.copy()
    ref = arr[0].copy()
    root_xyz_scale = float(cfg.root_xyz_scale)
    joint_scale = float(
        joint_scale_override if joint_scale_override is not None else cfg.joint_scale
    )

    # root position: x, y, z
    out[:, 0:3] = ref[0:3] + root_xyz_scale * (arr[:, 0:3] - ref[0:3])
    # root quaternion: 3:7，不做错误的线性幅值缩放
    out[:, 3:7] = arr[:, 3:7]
    # actuated joints
    out[:, 7:] = ref[7:] + joint_scale * (arr[:, 7:] - ref[7:])
    return out
