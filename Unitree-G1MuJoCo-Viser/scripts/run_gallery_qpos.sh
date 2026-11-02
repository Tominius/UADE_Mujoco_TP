#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
conda activate g1_mujoco 2>/dev/null || true
python main.py --mode qpos --port "${1:-9878}"
