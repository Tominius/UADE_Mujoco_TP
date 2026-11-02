from __future__ import annotations

from pathlib import Path

from gvhmr_gmr.pipeline.gmr_adapter import GMRMotionAdapter
from gvhmr_gmr.pipeline.npz_reference import load_reference_frames_from_npz
from gvhmr_gmr.pipeline.obs_builder import ReferenceFrame


def resolve_retarget_pkl(category: str, clip_name: str, data_root: str | Path) -> Path | None:
    """G1 Moves dataset retarget PKL (GMR-compatible format)."""
    p = Path(data_root) / category / clip_name / "retarget" / f"{clip_name}.pkl"
    return p if p.exists() else None


class ReferenceProvider:
    """Unified reference motion source: npz | gmr | pkl."""

    def __init__(
        self,
        *,
        mode: str = "npz",
        target_fps: float = 50.0,
    ) -> None:
        self.mode = mode
        self.target_fps = target_fps
        self._frames: list[ReferenceFrame] = []
        self._last_valid: ReferenceFrame | None = None

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def load_npz(self, npz_path: str | Path) -> None:
        self.mode = "npz"
        self._frames = load_reference_frames_from_npz(npz_path, self.target_fps)

    def load_gmr_pkl(self, pkl_path: str | Path) -> None:
        self.mode = "gmr"
        self._frames = GMRMotionAdapter(pkl_path, target_fps=self.target_fps).build_reference_frames()

    def load(self, path: str | Path, mode: str | None = None) -> None:
        p = Path(path)
        m = mode or self.mode
        if m == "npz" or p.suffix == ".npz":
            self.load_npz(p)
        elif m in ("gmr", "pkl") or p.suffix == ".pkl":
            self.load_gmr_pkl(p)
        else:
            raise ValueError(f"unsupported reference path: {p}")

    def get_frame(self, index: int) -> ReferenceFrame:
        if not self._frames:
            raise RuntimeError("reference not loaded")
        idx = max(0, min(index, len(self._frames) - 1))
        frame = self._frames[idx]
        self._last_valid = frame
        return frame

    def get_frame_safe(self, index: int) -> tuple[ReferenceFrame, bool]:
        try:
            return self.get_frame(index), True
        except Exception:
            if self._last_valid is not None:
                return self._last_valid, False
            raise
