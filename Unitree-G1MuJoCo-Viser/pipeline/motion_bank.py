from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MotionClip:
    key: str
    display_name: str
    category: str
    csv_path: Path
    source_name: str


def classify_clip(stem: str) -> str:
    s = stem.lower()
    martial_keywords = (
        "fence",
        "karate",
        "kick",
        "punch",
        "boxing",
        "sword",
        "combat",
        "move",
    )
    gesture_keywords = ("hands", "wave", "clap", "point", "greet", "gesture")
    dance_keywords = (
        "dance",
        "salsa",
        "hype",
        "shuffle",
        "broadway",
        "woah",
        "tiktok",
        "dad",
    )
    if any(k in s for k in martial_keywords):
        return "武术"
    if any(k in s for k in gesture_keywords):
        return "手势"
    if any(k in s for k in dance_keywords):
        return "舞蹈"
    if s.startswith("bonus_"):
        return "手势"
    if s.startswith("dance_"):
        return "舞蹈"
    return "其他"


def beautify_name(stem: str) -> str:
    name = stem.replace("_web", "")
    name = name.replace("dance_", "").replace("bonus_", "")
    name = name.replace("_", " ")
    return name


def scan_motion_bank(csv_dir: str | Path) -> list[MotionClip]:
    """
    自动扫描动作目录，返回动作元数据列表。

    输入:
        csv_dir: CSV 目录路径
    输出:
        list[MotionClip]
    异常:
        FileNotFoundError: 目录不存在
        RuntimeError: 目录中没有找到 CSV
    """
    root = Path(csv_dir)
    if not root.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {root}")

    clips: list[MotionClip] = []
    for csv_path in sorted(root.glob("*.csv")):
        stem = csv_path.stem
        category = classify_clip(stem)
        pretty = beautify_name(stem)
        key = f"{category}::{stem}"
        display_name = f"{category} • {pretty}"
        clips.append(
            MotionClip(
                key=key,
                display_name=display_name,
                category=category,
                csv_path=csv_path,
                source_name=stem,
            )
        )

    if not clips:
        raise RuntimeError(f"CSV 目录里没有找到动作文件: {root}")
    return clips


def group_by_category(clips: Iterable[MotionClip]) -> dict[str, list[MotionClip]]:
    out: dict[str, list[MotionClip]] = {"舞蹈": [], "武术": [], "手势": [], "其他": []}
    for clip in clips:
        out.setdefault(clip.category, []).append(clip)
    return {k: v for k, v in out.items() if v}
