# LAFAN1 (BVH, 30 FPS)

1. 从 [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 获取 `lafan1.zip`（需 git-lfs）
2. 解压 BVH 到此目录
3. Viser **🎬 视频 → G1** → **LAFAN1 → G1 (BVH batch)**，或 CLI：

```bash
python -m gvhmr_gmr.pipeline_runner batch-motion lafan1 --dry-run
```

输出：`gvhmr_gmr/artifacts/gmr/lafan1/*.pkl`（播放时使用 PKL 内 `fps`，不硬编码 60Hz）
