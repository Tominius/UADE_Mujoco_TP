from __future__ import annotations

from pipeline.motion_analyzer import MotionStats
from pipeline.motion_bank import MotionClip

_CATEGORY_EMOJI = {
    "舞蹈": "🕺",
    "武术": "🥋",
    "手势": "🤝",
    "其他": "🎯",
}

_INTENSITY_COLOR = {
    "低": "🟢",
    "中": "🟡",
    "高": "🔴",
}


def render_motion_card(clip: MotionClip, stats: MotionStats, *, has_policy: bool = False) -> str:
    emoji = _CATEGORY_EMOJI.get(clip.category, "🎯")
    level_icon = _INTENSITY_COLOR.get(stats.intensity_level, "⚪")
    bar_count = max(1, int(round(stats.intensity_norm * 10.0)))
    bar = "█" * bar_count + "░" * (10 - bar_count)
    pretty = clip.display_name.split("•", 1)[-1].strip()
    policy_line = "| Policy | ⚡ ONNX 可用 |\n" if has_policy else ""
    return (
        f"### {emoji} {pretty}\n\n"
        f"| 属性 | 数值 |\n"
        f"|:---|:---|\n"
        f"| 类型 | {clip.category} |\n"
        f"{policy_line}"
        f"| 时长 | **{stats.duration_sec:.1f}s** |\n"
        f"| 帧数 | {stats.num_frames} |\n"
        f"| 强度 | {level_icon} **{stats.intensity_level}** ({stats.intensity_raw:.4f}) |\n"
        f"| 根位移 | {stats.root_path_m:.2f} m |\n\n"
        f"**强度条** `{bar}`"
    )
