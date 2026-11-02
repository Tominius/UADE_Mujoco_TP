from __future__ import annotations

import mujoco
import numpy as np

from pipeline.policy_discovery import find_policy_bundle
from pipeline.policy_onnx import G1MovesPolicyRunner, validate_policy_xml


def test_policy_xml_sensors():
    m = mujoco.MjModel.from_xml_path("assets/proto_g1/g1_policy_mode15_web.xml")
    validate_policy_xml(m)
    assert m.nsensordata >= 6


def test_onnx_io_dad_dance():
    bundle = find_policy_bundle("dance_B_DadDance_web", "g1_moves/g1_moves_data")
    assert bundle is not None
    from pipeline.policy_onnx import inspect_onnx

    info = inspect_onnx(bundle.onnx_path)
    assert info.input_shape[-1] == 160
    assert info.output_shape[-1] == 29


def test_policy_rollout_dad_dance():
    """Policy v2: no root hack, v2 control loop, no NaN in early rollout."""
    bundle = find_policy_bundle("dance_B_DadDance_web", "g1_moves/g1_moves_data")
    assert bundle is not None
    model = mujoco.MjModel.from_xml_path("assets/proto_g1/g1_policy_mode15_web.xml")
    data = mujoco.MjData(model)
    runner = G1MovesPolicyRunner(bundle, model, data)
    runner.reset(speed=1.0)
    assert not hasattr(runner, "root_guided")
    assert runner.DECIMATION == 50
    assert abs(float(model.opt.timestep) - 0.001) < 1e-9

    for _ in range(25):
        runner.simulate_step()
        assert not np.any(np.isnan(data.qpos))
        assert not np.any(np.isnan(data.qacc))
