#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


SRC_ROOT = Path("g1_moves/g1_moves_data")
OUT_ROOT = Path("g1_moves/g1_moves_outputs/web_csv")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def convert_one(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, header=None)

    if df.shape[1] != 36:
        raise ValueError(f"{input_csv} expected 36 columns, got {df.shape[1]}")

    # G1 Moves retarget CSV 常见布局：
    # 0-2  : root position
    # 3-6  : root quaternion x,y,z,w
    # 7-35 : 29 G1 joints
    #
    # 我们当前 MuJoCo qpos-style viewer 更可能需要：
    # 0-2  : root position
    # 3-6  : root quaternion w,x,y,z
    # 7-35 : 29 G1 joints
    #
    # 因此这里做 xyzw -> wxyz。
    out_df = pd.concat(
        [
            df.iloc[:, 0:3],
            df.iloc[:, [6, 3, 4, 5]],
            df.iloc[:, 7:36],
        ],
        axis=1,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, header=False)

    print(f"OK {input_csv} -> {output_csv} {df.shape} -> {out_df.shape}")


def main():
    files = sorted(SRC_ROOT.glob("**/retarget/*.csv"))
    if not files:
        raise SystemExit("No retarget CSV found.")

    for p in files:
        # p: g1_moves_data/dance/B_DadDance/retarget/B_DadDance.csv
        rel = p.relative_to(SRC_ROOT)
        category = rel.parts[0]
        clip = rel.parts[1]
        out = OUT_ROOT / f"{category}_{clip}_web.csv"
        convert_one(p, out)

    print("\nconverted output dir:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
