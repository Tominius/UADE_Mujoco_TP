"""Unit tests for v2 control layer."""
from __future__ import annotations

import numpy as np

from pipeline.action_controller import ActionController
from pipeline.pd_controller import build_v2_pd_gains


def test_action_controller_smooth_and_scale():
    ctrl = ActionController()
    a0 = ctrl.process(np.ones(29, dtype=np.float32))
    a1 = ctrl.process(np.ones(29, dtype=np.float32))
    assert np.all(a0 >= -1.0) and np.all(a0 <= 1.0)
    assert a1[0] < 1.0  # smoothed below raw clip*scale


def test_v2_pd_gains_layout():
    kp, kd = build_v2_pd_gains()
    assert kp.shape == (29,)
    assert float(kp[0]) == 80.0
    assert float(kp[14]) == 60.0
    assert float(kp[20]) == 40.0
    assert float(kd[0]) == 8.0
