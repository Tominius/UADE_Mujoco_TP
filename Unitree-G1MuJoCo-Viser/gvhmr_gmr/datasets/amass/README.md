# AMASS (SMPL-X)

GMR 官方支持 `smplx_to_robot_dataset.py` 批处理 AMASS **raw SMPL-X**（非 SMPL+H）。

1. 在 [AMASS 官网](https://amass.is.tue.mpg.de/) 注册并下载 SMPL-X 数据
2. 解压到此目录
3. 配置 `gvhmr_gmr/local_paths.yaml` 中的 `gmr_repo`
4. Viser **🎬 视频 → G1** → **AMASS → G1 (SMPL-X batch)**，或 CLI：

```bash
python -m gvhmr_gmr.pipeline_runner batch-motion amass --dry-run
```

输出：`gvhmr_gmr/artifacts/gmr/amass/*.pkl`
