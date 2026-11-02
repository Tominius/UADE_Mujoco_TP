from __future__ import annotations

from pathlib import Path

from pipeline.policy_discovery import find_policy_bundle
from pipeline.policy_onnx import inspect_onnx


def test_onnx_io_dad_dance():
    bundle = find_policy_bundle("dance_B_DadDance_web", "g1_moves/g1_moves_data")
    assert bundle is not None
    info = inspect_onnx(bundle.onnx_path)
    assert info.input_name == "obs"
    assert info.output_name == "actions"
    assert info.input_shape[-1] == 160
    assert info.output_shape[-1] == 29
