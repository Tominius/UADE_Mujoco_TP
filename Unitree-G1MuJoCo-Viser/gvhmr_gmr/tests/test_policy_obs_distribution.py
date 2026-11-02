from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from gvhmr_gmr.pipeline.gmr_adapter import GMRMotionAdapter
from gvhmr_gmr.pipeline.obs_builder import ObsBuilder160, SimState
from pipeline.policy_discovery import find_policy_bundle


def test_policy_obs_distribution_npz_vs_pkl():
    bundle = find_policy_bundle("dance_B_DadDance_web", "g1_moves/g1_moves_data")
    assert bundle is not None

    pkl = (
        Path("g1_moves/g1_moves_data")
        / bundle.category
        / bundle.clip_name
        / "retarget"
        / f"{bundle.clip_name}.pkl"
    )
    pkl_frames = GMRMotionAdapter(pkl, target_fps=50.0).build_reference_frames()

    sess = ort.InferenceSession(str(bundle.onnx_path), providers=["CPUExecutionProvider"])
    builder = ObsBuilder160()

    sim = SimState(
        robot_pos_w=pkl_frames[0].anchor_pos_w,
        robot_quat_wxyz=pkl_frames[0].anchor_quat_wxyz,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        joint_pos=pkl_frames[0].ref_joint_pos,
        joint_vel=np.zeros(29, dtype=np.float32),
        last_action=np.zeros(29, dtype=np.float32),
    )

    obs = builder.build(sim, pkl_frames[0]).obs160
    actions = sess.run(["actions"], {"obs": obs[None]})[0][0]
    assert actions.shape == (29,)
    assert np.isfinite(actions).all()
    assert float(np.abs(actions).max()) < 10.0
