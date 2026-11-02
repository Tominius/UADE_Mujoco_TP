from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
GMR_OUT = ARTIFACTS / "gmr"
GVHMR_OUT = ARTIFACTS / "gvhmr"

# G1 Moves policy control clock (official run_policy.py).
POLICY_HZ = 50.0
POLICY_CONTROL_DT = 1.0 / POLICY_HZ


@dataclass(frozen=True)
class IntegrationConfig:
    reference_mode: str = "npz"  # npz | gmr | pkl
    action_mode: str = "residual"  # teacher | residual | raw_policy
    gmr_motion_path: str = ""
    gvhmr_pred_path: str = ""
    target_fps: float = POLICY_HZ
    residual_scale: float = 0.15
    action_clip: float = 0.35
    default_gmr_demo: str = field(
        default_factory=lambda: str(
            GMR_OUT / "demo_g1.pkl"
        )
    )
