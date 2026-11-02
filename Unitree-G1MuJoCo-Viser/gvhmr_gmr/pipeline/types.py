from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelinePaths:
    gvhmr_repo: Path | None = None
    gmr_repo: Path | None = None
    gvhmr_env: str = "gvhmr"
    gmr_env: str = "gmr"
    robot_name: str = "unitree_g1"
    use_static_camera: bool = True
