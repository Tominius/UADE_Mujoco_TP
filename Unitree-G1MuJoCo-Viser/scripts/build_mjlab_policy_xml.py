#!/usr/bin/env python3
"""Build G1 Moves policy MuJoCo XML from mjlab g1.xml + training actuators."""
from __future__ import annotations

from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
MJLAB_G1 = Path("/tmp/mjlab_g1.xml")
OUT_DIR = ROOT / "assets" / "proto_g1" / "mjlab_g1"
OUT_XML = OUT_DIR / "g1_policy_mode15.xml"

# PD / effort from g1-moves run_policy.py + env.yaml
JOINTS = [
    ("left_hip_pitch_joint", 40.2, 2.6, -88, 88),
    ("left_hip_roll_joint", 99.1, 6.3, -139, 139),
    ("left_hip_yaw_joint", 40.2, 2.6, -88, 88),
    ("left_knee_joint", 99.1, 6.3, -139, 139),
    ("left_ankle_pitch_joint", 28.6, 1.8, -50, 50),
    ("left_ankle_roll_joint", 28.6, 1.8, -50, 50),
    ("right_hip_pitch_joint", 40.2, 2.6, -88, 88),
    ("right_hip_roll_joint", 99.1, 6.3, -139, 139),
    ("right_hip_yaw_joint", 40.2, 2.6, -88, 88),
    ("right_knee_joint", 99.1, 6.3, -139, 139),
    ("right_ankle_pitch_joint", 28.6, 1.8, -50, 50),
    ("right_ankle_roll_joint", 28.6, 1.8, -50, 50),
    ("waist_yaw_joint", 40.2, 2.6, -88, 88),
    ("waist_roll_joint", 28.6, 1.8, -50, 50),
    ("waist_pitch_joint", 28.6, 1.8, -50, 50),
    ("left_shoulder_pitch_joint", 14.3, 0.9, -25, 25),
    ("left_shoulder_roll_joint", 14.3, 0.9, -25, 25),
    ("left_shoulder_yaw_joint", 14.3, 0.9, -25, 25),
    ("left_elbow_joint", 14.3, 0.9, -25, 25),
    ("left_wrist_roll_joint", 14.3, 0.9, -25, 25),
    ("left_wrist_pitch_joint", 16.8, 1.1, -5, 5),
    ("left_wrist_yaw_joint", 16.8, 1.1, -5, 5),
    ("right_shoulder_pitch_joint", 14.3, 0.9, -25, 25),
    ("right_shoulder_roll_joint", 14.3, 0.9, -25, 25),
    ("right_shoulder_yaw_joint", 14.3, 0.9, -25, 25),
    ("right_elbow_joint", 14.3, 0.9, -25, 25),
    ("right_wrist_roll_joint", 14.3, 0.9, -25, 25),
    ("right_wrist_pitch_joint", 16.8, 1.1, -5, 5),
    ("right_wrist_yaw_joint", 16.8, 1.1, -5, 5),
]


def main() -> None:
    if not MJLAB_G1.exists():
        raise FileNotFoundError(f"Missing {MJLAB_G1}; curl mjlab g1.xml first")

    spec = mujoco.MjSpec.from_file(str(MJLAB_G1))
    assets_dir = (OUT_DIR / "assets").resolve()
    spec.compiler.meshdir = str(assets_dir)

    # Training scene: flat ground + implicitfast integrator.
    if spec.worldbody.geoms:
        pass
    wb = spec.worldbody
    wb.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, 0),
        size=(10, 10, 0.05),
        rgba=(0.08, 0.08, 0.08, 1),
        contype=1,
        conaffinity=1,
    )

    spec.option.timestep = 0.005
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO

    # Foot condim=3, friction=0.6 (mjlab FULL_COLLISION).
    for g in spec.geoms:
        if g.name and ("foot" in g.name and "collision" in g.name):
            g.condim = 3
            g.friction[0] = 0.6

    for jname, kp, kv, fmin, fmax in JOINTS:
        spec.add_actuator(
            name=jname,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=jname,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_NONE,
            gainprm=[1.0] + [0.0] * 9,
            forcelimited=True,
            forcerange=[fmin, fmax],
        )

    model = spec.compile()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_XML.write_text(spec.to_xml())
    print(f"Wrote {OUT_XML} nu={model.nu} nq={model.nq} nsensor={model.nsensor}")


if __name__ == "__main__":
    main()
