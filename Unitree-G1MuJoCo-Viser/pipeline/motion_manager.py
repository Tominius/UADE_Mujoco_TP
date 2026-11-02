from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class PlaybackMode(str, Enum):
    IDLE = "idle"
    QPOS = "qpos"
    POLICY = "policy"
    RETARGET = "retarget"


@dataclass
class RetargetClip:
    """Unified GMR/G1 Moves PKL clip (fps + root + dof)."""

    name: str
    source: str
    fps: float
    root_pos: np.ndarray | None
    root_rot: np.ndarray | None
    dof_pos: np.ndarray
    path: str

    @property
    def num_frames(self) -> int:
        return int(self.dof_pos.shape[0])


def load_pkl_clip(path: str | Path, name: str, source: str) -> RetargetClip:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected dict pickle, got {type(data)}")

    required = {"fps", "dof_pos"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"PKL missing keys: {sorted(missing)} in {path}")

    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    root_pos = (
        np.asarray(data["root_pos"], dtype=np.float32)
        if "root_pos" in data and data["root_pos"] is not None
        else None
    )
    root_rot = (
        np.asarray(data["root_rot"], dtype=np.float32)
        if "root_rot" in data and data["root_rot"] is not None
        else None
    )

    return RetargetClip(
        name=name,
        source=source,
        fps=float(data.get("fps", 60)),
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        path=str(path),
    )


@dataclass
class MotionEntry(Generic[T]):
    key: str
    display_name: str
    category: str
    qpos: np.ndarray
    policy_bundle: T | None = None


class MotionManager(Generic[T]):
    """Central registry and state machine for qpos / policy / retarget playback."""

    def __init__(self, entries: dict[str, MotionEntry[T]], clip_order: list[str]) -> None:
        self._entries = entries
        self._clip_order = clip_order
        self.current_key: str | None = None
        self.mode = PlaybackMode.IDLE
        self.retarget_clip: RetargetClip | None = None
        self.retarget_frame_idx: int = 0

    def load_retarget_clip(self, clip: RetargetClip) -> None:
        self.retarget_clip = clip
        self.retarget_frame_idx = 0
        self.current_key = None
        self.mode = PlaybackMode.RETARGET

    def stop_retarget(self) -> None:
        self.retarget_clip = None
        self.retarget_frame_idx = 0

    def step_retarget(self) -> dict | None:
        clip = self.retarget_clip
        if clip is None:
            return None
        idx = min(self.retarget_frame_idx, clip.num_frames - 1)
        frame = {
            "fps": clip.fps,
            "dof_pos": clip.dof_pos[idx],
            "root_pos": None if clip.root_pos is None else clip.root_pos[idx],
            "root_rot": None if clip.root_rot is None else clip.root_rot[idx],
        }
        self.retarget_frame_idx += 1
        return frame

    @property
    def entries(self) -> dict[str, MotionEntry[T]]:
        return self._entries

    @property
    def clip_order(self) -> list[str]:
        return self._clip_order

    def get(self, key: str) -> MotionEntry[T]:
        if key not in self._entries:
            raise KeyError(f"未知动作 key: {key}")
        return self._entries[key]

    def find_key(self, name: str) -> str | None:
        for key, entry in self._entries.items():
            if name in key or name in entry.display_name:
                return key
        return None

    def play_motion(self, key: str, *, mode: PlaybackMode = PlaybackMode.QPOS) -> MotionEntry[T]:
        entry = self.get(key)
        self.current_key = key
        self.mode = mode
        return entry

    def switch_motion(self, key: str, *, mode: PlaybackMode = PlaybackMode.QPOS) -> MotionEntry[T]:
        return self.play_motion(key, mode=mode)

    def stop_motion(self) -> None:
        self.current_key = None
        self.mode = PlaybackMode.IDLE
        self.stop_retarget()

    def next_key(self) -> str | None:
        if not self._clip_order:
            return None
        if self.current_key is None:
            return self._clip_order[0]
        idx = self._clip_order.index(self.current_key)
        return self._clip_order[(idx + 1) % len(self._clip_order)]

    def has_policy(self, key: str) -> bool:
        return self._entries[key].policy_bundle is not None

    def policy_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.policy_bundle is not None)
