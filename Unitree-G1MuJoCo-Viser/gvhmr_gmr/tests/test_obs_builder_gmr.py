from __future__ import annotations

import numpy as np

from gvhmr_gmr.pipeline.obs_builder import ObsBuilder160, ReferenceFrame, SimState, quat_xyzw_to_wxyz


def test_quat_xyzw_to_wxyz_matches_npz():
    xyzw = np.array([-0.0414, 0.01585, -0.6828, 0.7293], dtype=np.float32)
    wxyz = quat_xyzw_to_wxyz(xyzw)
    expected = np.array([0.7293, -0.0414, 0.01585, -0.6828], dtype=np.float32)
    assert np.allclose(wxyz, expected, atol=1e-3)


def test_obs_builder_shape_and_finiteness():
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
    ref = ReferenceFrame(
        ref_joint_pos=np.zeros(29, dtype=np.float32),
        ref_joint_vel=np.zeros(29, dtype=np.float32),
        anchor_pos_w=np.zeros(3, dtype=np.float32),
        anchor_quat_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        fps=50.0,
        source="test",
    )
    out = builder.build(sim, ref)
    assert out.obs160.shape == (160,)
    assert np.isfinite(out.obs160).all()
