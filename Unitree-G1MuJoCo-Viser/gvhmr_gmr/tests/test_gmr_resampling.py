from __future__ import annotations

from pathlib import Path

import numpy as np

from gvhmr_gmr.pipeline.gmr_adapter import GMRMotionAdapter
from pipeline.policy_discovery import find_policy_bundle


def test_gmr_resample_to_50hz():
    bundle = find_policy_bundle("dance_B_DadDance_web", "g1_moves/g1_moves_data")
    assert bundle is not None
    pkl = Path("g1_moves/g1_moves_data") / bundle.category / bundle.clip_name / "retarget" / f"{bundle.clip_name}.pkl"
    adapter = GMRMotionAdapter(pkl, target_fps=50.0)
    motion = adapter.resample()
    assert motion.dof_pos.shape[1] == 29
    assert abs(motion.fps - 50.0) < 1e-6
    vel = np.gradient(motion.dof_pos, 1.0 / 50.0, axis=0)
    assert np.isfinite(vel).all()
    # no large step noise in velocity
    assert float(np.abs(np.diff(vel, axis=0)).mean()) < 2.0
