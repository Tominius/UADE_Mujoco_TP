#!/usr/bin/env python3
"""Validate G1 Moves policy deployment (obs / IMU / rollout)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from pipeline.policy_discovery import find_policy_bundle
from pipeline.policy_onnx import G1MovesPolicyRunner, validate_policy_xml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-stem", default="dance_B_DadDance_web")
    parser.add_argument("--policy-model", default="assets/proto_g1/g1_policy_mode15_web.xml")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ROOT / args.policy_model))
    validate_policy_xml(model)
    bundle = find_policy_bundle(args.clip_stem, str(ROOT / "g1_moves/g1_moves_data"))
    if bundle is None:
        raise SystemExit(f"no policy bundle for {args.clip_stem}")

    data = mujoco.MjData(model)
    runner = G1MovesPolicyRunner(bundle, model, data)
    runner.reset(speed=1.0)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")

    errs: list[float] = []
    max_qacc = 0.0
    for i in range(args.steps):
        frame = runner.frame_index
        runner.simulate_step()
        errs.append(float(np.abs(data.qpos[7:36] - runner.ref_joint_pos[frame]).mean()))
        max_qacc = max(max_qacc, float(np.max(np.abs(data.qacc))))
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qacc)):
            print(f"FAIL: NaN at step {i + 1}")
            sys.exit(1)
        if data.qpos[2] < 0.35 or data.xpos[head][2] <= data.xpos[pelvis][2]:
            tail = np.mean(errs[max(0, len(errs) - min(10, len(errs))) :])
            print(
                f"FAIL: fallen at step {i + 1}, root_z={float(data.qpos[2]):.3f}, "
                f"max_|qacc|={max_qacc:.1f}, tail_joint_mae={tail:.3f} rad"
            )
            sys.exit(1)

    tail = np.mean(errs[max(0, len(errs) - 100) :])
    print(
        f"OK: {args.steps} steps, tail_joint_mae={tail:.3f} rad, "
        f"last={errs[-1]:.3f}, max_|qacc|={max_qacc:.1f}"
    )


if __name__ == "__main__":
    main()
