from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gvhmr_gmr.config import PROJECT_ROOT, ROOT
from gvhmr_gmr.pipeline.paths_loader import load_local_paths

VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
PKL_EXTS = {".pkl"}
MOTION_RAW_EXTS = {
    ".bvh",
    ".fbx",
    ".npz",
    ".pkl",
    ".npy",
    ".pt",
}


@dataclass
class SampleItem:
    name: str
    path: str
    source: str
    kind: str  # video | generated_pkl | motion_raw
    route: str  # gvhmr_then_gmr | gmr_smplx_batch | gmr_bvh_batch | gmr_fbx_batch | playback_only


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _list_files(root: Path, exts: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _list_motion_raw(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in MOTION_RAW_EXTS
        and p.name != "README.md"
    )


def build_registry(project_root: Path | None = None) -> dict:
    root = (project_root or PROJECT_ROOT).resolve()
    cfg = _load_yaml(ROOT / "config" / "dataset_sources.yaml")
    local = load_local_paths()

    data: dict = {
        "repo_status": {
            "gvhmr_repo": str(local.gvhmr_repo or ""),
            "gmr_repo": str(local.gmr_repo or ""),
            "gvhmr_ready": bool(local.gvhmr_repo and local.gvhmr_repo.exists()),
            "gmr_ready": bool(local.gmr_repo and local.gmr_repo.exists()),
            "robot_name": local.robot_name,
            "use_static_camera": local.use_static_camera,
        },
        "video_samples": {},
        "generated_groups": {},
        "raw_motion_groups": {},
        "group_labels": cfg.get("generated_group_labels") or {},
    }

    for source_name, meta in (cfg.get("video_routes") or {}).items():
        src_dir = root / meta["input_dir"]
        items = [
            asdict(
                SampleItem(
                    name=p.stem,
                    path=str(p.resolve()),
                    source=source_name,
                    kind="video",
                    route=meta["route"],
                )
            )
            for p in _list_files(src_dir, VIDEO_EXTS)
        ]
        data["video_samples"][source_name] = items

        out_dir = root / meta["gmr_out_dir"]
        generated = [
            asdict(
                SampleItem(
                    name=p.stem,
                    path=str(p.resolve()),
                    source=source_name,
                    kind="generated_pkl",
                    route="playback_only",
                )
            )
            for p in _list_files(out_dir, PKL_EXTS)
        ]
        data["generated_groups"][source_name] = generated

    for source_name, meta in (cfg.get("motion_routes") or {}).items():
        src_dir = root / meta["input_dir"]
        raw_items = [
            asdict(
                SampleItem(
                    name=p.stem,
                    path=str(p.resolve()),
                    source=source_name,
                    kind=meta["kind"],
                    route=meta["route"],
                )
            )
            for p in _list_motion_raw(src_dir)
        ]
        data["raw_motion_groups"][source_name] = raw_items

        out_dir = root / meta["gmr_out_dir"]
        generated = [
            asdict(
                SampleItem(
                    name=p.stem,
                    path=str(p.resolve()),
                    source=source_name,
                    kind="generated_pkl",
                    route="playback_only",
                )
            )
            for p in _list_files(out_dir, PKL_EXTS)
        ]
        data["generated_groups"][source_name] = generated

    return data


def get_route_meta(source_name: str, project_root: Path | None = None) -> dict | None:
    root = (project_root or PROJECT_ROOT).resolve()
    cfg = _load_yaml(ROOT / "config" / "dataset_sources.yaml")
    for section in ("video_routes", "motion_routes"):
        meta = (cfg.get(section) or {}).get(source_name)
        if meta:
            resolved = dict(meta)
            for key in ("input_dir", "gvhmr_out_dir", "gmr_out_dir"):
                if key in resolved:
                    resolved[key] = str((root / resolved[key]).resolve())
            return resolved
    return None


def gmr_out_pkl(project_root: Path, source_name: str, clip_stem: str) -> Path:
    meta = get_route_meta(source_name, project_root)
    if meta is None:
        raise KeyError(f"unknown source: {source_name}")
    return Path(meta["gmr_out_dir"]) / f"{clip_stem}.pkl"


if __name__ == "__main__":
    reg = build_registry()
    print(json.dumps(reg, ensure_ascii=False, indent=2))
