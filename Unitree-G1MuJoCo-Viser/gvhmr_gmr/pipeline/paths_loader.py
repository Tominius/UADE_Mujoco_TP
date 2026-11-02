from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from gvhmr_gmr.config import ROOT
from gvhmr_gmr.pipeline.types import PipelinePaths


@dataclass
class LocalPaths:
    gvhmr_repo: Path | None = None
    gmr_repo: Path | None = None
    gvhmr_env: str = "gvhmr"
    gmr_env: str = "gmr"
    gvhmr_python: str = "python"
    gmr_python: str = "python"
    robot_name: str = "unitree_g1"
    use_static_camera: bool = True
    artifacts_root: Path | None = None

    @property
    def gvhmr_ready(self) -> bool:
        return self.gvhmr_repo is not None and self.gvhmr_repo.exists()

    @property
    def gmr_ready(self) -> bool:
        return self.gmr_repo is not None and self.gmr_repo.exists()


def _parse_flat_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


def load_local_paths() -> LocalPaths:
    yaml_path = ROOT / "local_paths.yaml"
    gvhmr_repo = os.environ.get("GVHMR_REPO")
    gmr_repo = os.environ.get("GMR_REPO")
    gvhmr_env = os.environ.get("GVHMR_ENV", "gvhmr")
    gmr_env = os.environ.get("GMR_ENV", "gmr")
    gvhmr_python = os.environ.get("GVHMR_PYTHON", "python")
    gmr_python = os.environ.get("GMR_PYTHON", "python")
    robot_name = os.environ.get("GMR_ROBOT", "unitree_g1")
    use_static_camera = os.environ.get("GVHMR_STATIC_CAMERA", "1") not in ("0", "false", "False")
    artifacts_root: str | None = os.environ.get("GVHMR_GMR_ARTIFACTS")

    if yaml_path.exists():
        data = _parse_flat_yaml(yaml_path)
        gvhmr_repo = data.get("gvhmr_repo", gvhmr_repo)
        gmr_repo = data.get("gmr_repo", gmr_repo)
        gvhmr_env = data.get("gvhmr_env", gvhmr_env)
        gmr_env = data.get("gmr_env", gmr_env)
        gvhmr_python = data.get("gvhmr_python", gvhmr_python)
        gmr_python = data.get("gmr_python", gmr_python)
        robot_name = data.get("robot_name", robot_name)
        if "use_static_camera" in data:
            use_static_camera = str(data["use_static_camera"]).lower() in ("1", "true", "yes")
        artifacts_root = data.get("artifacts_root", artifacts_root)

    return LocalPaths(
        gvhmr_repo=Path(gvhmr_repo).expanduser() if gvhmr_repo else None,
        gmr_repo=Path(gmr_repo).expanduser() if gmr_repo else None,
        gvhmr_env=str(gvhmr_env),
        gmr_env=str(gmr_env),
        gvhmr_python=str(gvhmr_python),
        gmr_python=str(gmr_python),
        robot_name=str(robot_name),
        use_static_camera=bool(use_static_camera),
        artifacts_root=Path(artifacts_root).expanduser() if artifacts_root else None,
    )


def load_pipeline_paths() -> PipelinePaths:
    lp = load_local_paths()
    return PipelinePaths(
        gvhmr_repo=lp.gvhmr_repo,
        gmr_repo=lp.gmr_repo,
        gvhmr_env=lp.gvhmr_env,
        gmr_env=lp.gmr_env,
        robot_name=lp.robot_name,
        use_static_camera=lp.use_static_camera,
    )
