#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CLIP="${1:-B_DadDance}"
PORT="${2:-9878}"
conda activate g1_mujoco 2>/dev/null || true
python main.py --mode policy --clip "${CLIP}" --port "${PORT}"
