from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_qpos_csv(csv_path: str | Path, expected_nq: int | None = None) -> np.ndarray:
    """
    读取 qpos CSV。

    输入:
        csv_path: CSV 路径
        expected_nq: 期望列数；若提供则严格校验
    输出:
        np.ndarray, shape=(T, nq), dtype=float64
    异常:
        FileNotFoundError: 文件不存在
        RuntimeError: pandas 读取失败
        ValueError: CSV 为空、不是二维表、列数不匹配
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"动作文件不存在: {path}")

    try:
        df = pd.read_csv(path, header=None)
    except Exception as exc:
        raise RuntimeError(f"读取 CSV 失败: {path}") from exc

    if df.empty:
        raise ValueError(f"CSV 为空: {path}")

    arr = df.to_numpy(dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"CSV 不是二维数组: shape={arr.shape}, path={path}")

    if expected_nq is not None and arr.shape[1] != expected_nq:
        raise ValueError(
            f"qpos 维度不匹配: path={path}, csv列数={arr.shape[1]}, expected_nq={expected_nq}"
        )
    return arr
