from __future__ import annotations

from pathlib import Path

from pipeline.policy_discovery import find_policy_bundle, scan_policy_bundles


def test_scan_policy_bundles():
    root = Path("g1_moves/g1_moves_data")
    bundles = scan_policy_bundles(root)
    assert len(bundles) >= 20


def test_match_dad_dance():
    bundle = find_policy_bundle("dance_B_DadDance_web", "g1_moves/g1_moves_data")
    assert bundle is not None
    assert bundle.onnx_path.exists()
    assert bundle.npz_path.exists()
