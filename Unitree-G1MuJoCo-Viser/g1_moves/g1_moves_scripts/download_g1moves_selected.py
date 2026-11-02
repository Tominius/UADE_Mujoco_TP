#!/usr/bin/env python3
from pathlib import Path
from huggingface_hub import hf_hub_download

ENDPOINT = "https://hf-mirror.com"
REPO_ID = "exptech/g1-moves"
REPO_TYPE = "dataset"

ROOT = Path("g1_moves/g1_moves_data")
FILE_LIST = ROOT / "hf_file_list.txt"

files = [line.strip() for line in FILE_LIST.read_text().splitlines() if line.strip()]

# 先选展示价值最高、文件量可控的一批。这里不是全量下载。
# 优先 dance，再加少量 bonus。后面如果某些动作效果不好，再换。
retarget_csvs = [f for f in files if f.startswith("dance/") and "/retarget/" in f and f.endswith(".csv")]
bonus_csvs = [f for f in files if f.startswith("bonus/") and "/retarget/" in f and f.endswith(".csv")]

selected_csvs = retarget_csvs[:20] + bonus_csvs[:5]

if not selected_csvs:
    raise SystemExit("No retarget CSV found. Please check hf_file_list.txt")

# 对每个 CSV，顺手下载同 clip 的 README / policy onnx / training npz，如果存在。
selected = set()
for csv_path in selected_csvs:
    selected.add(csv_path)

    parts = csv_path.split("/")
    # 形如 dance/B_DadDance/retarget/B_DadDance.csv
    if len(parts) >= 4:
        category, clip = parts[0], parts[1]
        stem = Path(csv_path).stem

        candidates = [
            f"{category}/{clip}/README.md",
            f"{category}/{clip}/training/{stem}.npz",
            f"{category}/{clip}/policy/{stem}_policy.onnx",
            f"{category}/{clip}/policy/agent.yaml",
            f"{category}/{clip}/policy/env.yaml",
            f"{category}/{clip}/retarget/{stem}.pkl",
        ]
        for c in candidates:
            if c in files:
                selected.add(c)

selected = sorted(selected)

print("Will download file count:", len(selected))
for f in selected:
    print("  ", f)

for f in selected:
    print("\nDownloading:", f)
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=f,
        endpoint=ENDPOINT,
        local_dir=str(ROOT),
        local_dir_use_symlinks=False,
        resume_download=True,
        etag_timeout=60,
    )

print("\nDONE")
