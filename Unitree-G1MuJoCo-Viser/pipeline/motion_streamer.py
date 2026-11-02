from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaybackFrame:
    qpos: np.ndarray
    frame_index: int
    num_frames: int
    finished: bool
    preview_mode: bool


class MotionStreamer:
    """
    线程安全的动作播放状态机。

    功能:
        - load()
        - play() / pause() / stop()
        - set_speed()
        - next_frame()
    """

    def __init__(self, clip_fps: float = 60.0) -> None:
        self._clip_fps = float(clip_fps)
        self._lock = threading.RLock()
        self._motion: np.ndarray | None = None
        self._playing: bool = False
        self._loop: bool = True
        self._frame_cursor: float = 0.0
        self._speed: float = 1.0
        self._preview_end_frame: int | None = None

    def set_clip_fps(self, fps: float) -> None:
        with self._lock:
            self._clip_fps = max(1e-3, float(fps))

    def get_clip_fps(self) -> float:
        with self._lock:
            return self._clip_fps

    def load(
        self,
        motion: np.ndarray,
        *,
        loop: bool = True,
        preview_seconds: float | None = None,
    ) -> None:
        arr = np.asarray(motion, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"motion 必须是二维数组，得到 shape={arr.shape}")
        if arr.shape[0] < 1:
            raise ValueError("motion 至少需要 1 帧")

        with self._lock:
            self._motion = arr
            self._playing = True
            self._loop = bool(loop)
            self._frame_cursor = 0.0
            if preview_seconds is None:
                self._preview_end_frame = None
            else:
                self._preview_end_frame = max(
                    1,
                    min(arr.shape[0], int(round(preview_seconds * self._clip_fps))),
                )

    def set_speed(self, speed: float) -> None:
        with self._lock:
            self._speed = max(0.1, float(speed))

    def get_speed(self) -> float:
        with self._lock:
            return self._speed

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def play(self) -> None:
        with self._lock:
            if self._motion is not None:
                self._playing = True

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            self._frame_cursor = 0.0
            self._preview_end_frame = None

    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def current_num_frames(self) -> int:
        with self._lock:
            return 0 if self._motion is None else int(self._motion.shape[0])

    def next_frame(self, render_fps: float) -> PlaybackFrame | None:
        """根据当前 speed 和 render_fps 取下一帧。"""
        with self._lock:
            if not self._playing or self._motion is None:
                return None

            motion = self._motion
            num_frames = int(motion.shape[0])
            frame_index = min(int(self._frame_cursor), num_frames - 1)
            qpos = motion[frame_index].copy()

            cursor_increment = self._speed * self._clip_fps / max(render_fps, 1e-6)
            self._frame_cursor += cursor_increment

            preview_mode = self._preview_end_frame is not None
            finished = False
            limit = self._preview_end_frame if preview_mode else num_frames
            assert limit is not None

            if self._frame_cursor >= limit:
                if preview_mode:
                    self._playing = False
                    self._frame_cursor = 0.0
                    finished = True
                elif self._loop:
                    self._frame_cursor = 0.0
                    finished = True
                else:
                    self._frame_cursor = float(num_frames - 1)
                    self._playing = False
                    finished = True

            return PlaybackFrame(
                qpos=qpos,
                frame_index=frame_index,
                num_frames=num_frames,
                finished=finished,
                preview_mode=preview_mode,
            )
