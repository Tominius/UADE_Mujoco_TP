#!/usr/bin/env python3
"""Smoke validation for GVHMR/GMR integration (Phase A)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from gvhmr_gmr.pipeline.gmr_adapter import GMRMotionAdapter
from gvhmr_gmr.pipeline.obs_builder import ObsBuilder160, ReferenceFrame, SimState
from gvhmr_gmr.pipeline.reference_provider import ReferenceProvider, resolve_retarget_pkl
from pipeline.policy_discovery import find_policy_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-stem", default="dance_B_DadDance_web")
    parser.add_argument("--pkl", default=None, help="GMR/retarget PKL path")
    parser.add_argument("--data-root", default="g1_moves/g1_moves_data")
    args = parser.parse_args()

    bundle = find_policy_bundle(args.clip_stem, ROOT / args.data_root)
    if bundle is None:
        raise SystemExit(f"no bundle for {args.clip_stem}")

    pkl = Path(args.pkl) if args.pkl else resolve_retarget_pkl(
        bundle.category, bundle.clip_name, ROOT / args.data_root
    )
    if pkl is None or not pkl.exists():
        raise SystemExit("retarget PKL not found")

    provider = ReferenceProvider(mode="pkl")
    provider.load(pkl)
    builder = ObsBuilder160()

    sim = SimState(
        robot_pos_w=np.zeros(3, dtype=np.float32),
        robot_quat_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        joint_pos=np.zeros(29, dtype=np.float32),
        joint_vel=np.zeros(29, dtype=np.float32),
        last_action=np.zeros(29, dtype=np.float32),
    )

    t0 = time.perf_counter()
    ref = provider.get_frame(0)
    out = builder.build(sim, ref)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert out.obs160.shape == (160,)
    assert np.isfinite(out.obs160).all()
    print(f"OK: obs160 from {pkl.name}, source={out.source}, build={dt_ms:.2f}ms")
    print(f"frames={provider.num_frames}, ref_vel_std={ref.ref_joint_vel.std():.4f}")


if __name__ == "__main__":
    main()
