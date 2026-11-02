#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/g1_moves/g1_moves_data"

echo "=== ONNX policies ==="
find "${DATA}" -type f \( -name "*_policy.onnx" -o -name "*.onnx" \) 2>/dev/null | sort

echo
echo "=== training NPZ ==="
find "${DATA}" -type f -path "*/training/*.npz" 2>/dev/null | sort

echo
echo "=== YAML metadata ==="
find "${DATA}" -type f \( -name "agent.yaml" -o -name "env.yaml" \) 2>/dev/null | sort

echo
echo "=== count ==="
echo -n "onnx: "; find "${DATA}" -type f -name "*_policy.onnx" 2>/dev/null | wc -l
echo -n "npz:  "; find "${DATA}" -type f -path "*/training/*.npz" 2>/dev/null | wc -l
