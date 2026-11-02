from __future__ import annotations

import numpy as np

from pipeline.action_adapter import DEFAULT_JOINT_POS, G1_ACTION_SCALE


class ActionController:
    """Scale, clip, smooth, and optionally blend policy actions with reference qpos."""

    ACTION_SCALE = 0.5
    ACTION_CLIP = 1.0
    SMOOTH_PREV = 0.7
    SMOOTH_NOW = 0.3
    REF_BLEND = 0.7
    POLICY_BLEND = 0.3

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def process(self, raw_action: np.ndarray) -> np.ndarray:
        a = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 29:
            raise ValueError(f"expected 29 actions, got {a.shape[0]}")
        a = np.clip(a, -self.ACTION_CLIP, self.ACTION_CLIP)
        a = a * self.ACTION_SCALE
        if self._prev is None:
            self._prev = np.zeros(29, dtype=np.float32)
        a = self.SMOOTH_PREV * self._prev + self.SMOOTH_NOW * a
        self._prev = a.copy()
        return a

    def to_joint_targets(
        self,
        processed_action: np.ndarray,
        ref_joint_pos: np.ndarray | None = None,
        *,
        use_mjlab_scale: bool = True,
        blend_with_ref: bool = True,
    ) -> np.ndarray:
        a = np.asarray(processed_action, dtype=np.float32)
        if use_mjlab_scale:
            policy_target = DEFAULT_JOINT_POS + a * G1_ACTION_SCALE
        else:
            policy_target = DEFAULT_JOINT_POS + a
        if blend_with_ref and ref_joint_pos is not None:
            ref = np.asarray(ref_joint_pos, dtype=np.float32)
            return self.REF_BLEND * ref + self.POLICY_BLEND * policy_target
        return policy_target
