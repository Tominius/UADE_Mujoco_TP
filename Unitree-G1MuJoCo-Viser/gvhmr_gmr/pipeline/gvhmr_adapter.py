from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class GVHMRResult:
    pred_path: Path
    clip_name: str


class GVHMRAdapter:
    """Manage GVHMR offline outputs (Phase D — video → hmr4d_results.pt)."""

    def __init__(self, gvhmr_root: str | Path | None = None) -> None:
        self.gvhmr_root = Path(gvhmr_root) if gvhmr_root else None

    @staticmethod
    def validate_pred_file(path: str | Path) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"GVHMR pred file not found: {p}")
        if p.suffix != ".pt":
            raise ValueError(f"expected .pt file, got {p}")
        return p

    def default_output_path(self, clip_name: str = "demo") -> Path:
        if self.gvhmr_root is None:
            from gvhmr_gmr.config import GVHMR_OUT

            return GVHMR_OUT / clip_name / "hmr4d_results.pt"
        return self.gvhmr_root / "outputs" / "demo" / clip_name / "hmr4d_results.pt"

    def run_offline_demo(
        self,
        video_path: str | Path,
        *,
        static_camera: bool = True,
        gvhmr_repo: str | Path,
    ) -> GVHMRResult:
        """Run GVHMR tools/demo/demo.py (requires separate gvhmr conda env)."""
        repo = Path(gvhmr_repo)
        script = repo / "tools" / "demo" / "demo.py"
        if not script.exists():
            raise FileNotFoundError(f"GVHMR demo script not found: {script}")

        cmd = [sys.executable, str(script), f"--video={video_path}"]
        if static_camera:
            cmd.append("-s")
        subprocess.run(cmd, cwd=str(repo), check=True)

        clip = Path(video_path).stem
        pred = self.default_output_path(clip)
        if not pred.exists():
            pred = repo / "outputs" / "demo" / clip / "hmr4d_results.pt"
        self.validate_pred_file(pred)
        return GVHMRResult(pred_path=pred, clip_name=clip)
