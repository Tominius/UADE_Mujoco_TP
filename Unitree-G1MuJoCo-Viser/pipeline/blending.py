from __future__ import annotations

import numpy as np


def cosine_alpha(t: float) -> float:
    """Cosine ease-in-out on [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - np.cos(np.pi * t))


def blend_qpos(q_a: np.ndarray, q_b: np.ndarray, t: float) -> np.ndarray:
    """
    Blend two single-frame qpos vectors.
    Root translation and joints use cosine interpolation; quaternion held from target.
    """
    a = np.asarray(q_a, dtype=np.float64)
    b = np.asarray(q_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"blend shape mismatch: {a.shape} vs {b.shape}")
    s = cosine_alpha(t)
    out = a.copy()
    out[:3] = (1.0 - s) * a[:3] + s * b[:3]
    quat = (1.0 - s) * a[3:7] + s * b[3:7]
    norm = float(np.linalg.norm(quat))
    out[3:7] = quat / norm if norm > 1e-12 else b[3:7]
    out[7:] = (1.0 - s) * a[7:] + s * b[7:]
    return out
