from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyBundle:
    clip_stem: str
    category: str
    clip_name: str
    onnx_path: Path
    npz_path: Path
    agent_yaml: Path | None
    env_yaml: Path | None


def csv_stem_to_policy_paths(stem: str, data_root: str | Path) -> tuple[str, str, Path, Path]:
    """
    dance_B_DadDance_web -> (dance, B_DadDance, onnx, npz)
    """
    base = stem.replace("_web", "")
    if "_" not in base:
        raise ValueError(f"无法解析 clip stem: {stem}")
    category, clip_name = base.split("_", 1)
    root = Path(data_root)
    clip_dir = root / category / clip_name
    onnx_path = clip_dir / "policy" / f"{clip_name}_policy.onnx"
    npz_path = clip_dir / "training" / f"{clip_name}.npz"
    return category, clip_name, onnx_path, npz_path


def find_policy_bundle(stem: str, data_root: str | Path) -> PolicyBundle | None:
    try:
        category, clip_name, onnx_path, npz_path = csv_stem_to_policy_paths(stem, data_root)
    except ValueError:
        return None
    if not onnx_path.exists() or not npz_path.exists():
        return None
    policy_dir = onnx_path.parent
    agent_yaml = policy_dir / "agent.yaml"
    env_yaml = policy_dir / "env.yaml"
    return PolicyBundle(
        clip_stem=stem,
        category=category,
        clip_name=clip_name,
        onnx_path=onnx_path,
        npz_path=npz_path,
        agent_yaml=agent_yaml if agent_yaml.exists() else None,
        env_yaml=env_yaml if env_yaml.exists() else None,
    )


def scan_policy_bundles(data_root: str | Path) -> dict[str, PolicyBundle]:
    root = Path(data_root)
    out: dict[str, PolicyBundle] = {}
    if not root.exists():
        return out
    for onnx_path in sorted(root.glob("*/*/policy/*_policy.onnx")):
        clip_dir = onnx_path.parent.parent
        clip_name = clip_dir.name
        category = clip_dir.parent.name
        stem = f"{category}_{clip_name}_web"
        npz_path = clip_dir / "training" / f"{clip_name}.npz"
        if not npz_path.exists():
            continue
        policy_dir = onnx_path.parent
        out[stem] = PolicyBundle(
            clip_stem=stem,
            category=category,
            clip_name=clip_name,
            onnx_path=onnx_path,
            npz_path=npz_path,
            agent_yaml=(policy_dir / "agent.yaml") if (policy_dir / "agent.yaml").exists() else None,
            env_yaml=(policy_dir / "env.yaml") if (policy_dir / "env.yaml").exists() else None,
        )
    return out
