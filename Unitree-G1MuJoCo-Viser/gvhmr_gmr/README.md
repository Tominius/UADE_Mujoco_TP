# GVHMR + GMR 双路线 → G1 Viser 系统

三层结构（对齐工程落地方案）：

```
数据源注册层  dataset_registry + dataset_sources.yaml
     ↓
管线执行层    pipeline_runner (官方 GVHMR/GMR 脚本包装)
     ↓
统一播放层    RetargetClip / GMRMotionAdapter → Viser qpos 回放
```

## 三类输入

| 路线 | 输入 | 管线 | 输出 |
|------|------|------|------|
| **官方库** | G1 Moves CSV | 已有 gallery | 25 动作展示 |
| **视频** | 用户 mp4 / AIST++ 样例 | GVHMR → GMR | `artifacts/gmr/{user_videos,aistpp}/` |
| **Mocap** | AMASS SMPL-X / LAFAN1 BVH | GMR batch | `artifacts/gmr/{amass,lafan1}/` |

统一 PKL 格式：`fps`, `root_pos`, `root_rot`, `dof_pos`（与 G1 Moves retarget 一致）。

## 目录

```
gvhmr_gmr/
├── config/dataset_sources.yaml   # 数据源路由注册
├── dataset_registry.py           # 扫描样例 / 生成 PKL / repo 状态
├── pipeline_runner.py            # 官方脚本 CLI 包装 + dry-run
├── local_paths.yaml.example      # GVHMR/GMR 仓库路径
├── videos/aist_samples/          # AIST++ 示例视频（手动放置）
├── datasets/{amass,lafan1,...}/  # mocap 原始数据（手动放置）
└── artifacts/gmr/                # 统一 G1 PKL 输出
```

## 启动 Viser

```bash
conda activate g1_mujoco
python main.py --port 9878
```

- **🎬 视频 → G1**：上传视频 / AIST++ 下拉 → 开始分析 → G1 播放
- **📚 动作库**：官方 25 动作 + **✨ 生成动作库**（用户/AIST++/AMASS/LAFAN1 分组）

## 配置外部仓库

```bash
cp gvhmr_gmr/local_paths.yaml.example gvhmr_gmr/local_paths.yaml
# 编辑 gvhmr_repo / gmr_repo 为本地 clone 绝对路径
```

未配置时：UI 显示「外部仓库未配置」，仍可 **手动放置 PKL** 到 `artifacts/gmr/` 后播放。

## 自检

```bash
python -m py_compile gvhmr_gmr/dataset_registry.py gvhmr_gmr/pipeline_runner.py pipeline/motion_manager.py main.py
python main.py --registry
python main.py --dry-run-batch amass
python main.py --dry-run-batch lafan1
python gvhmr_gmr/scripts/validate_integration.py
```

## CLI 批处理

```bash
python -m gvhmr_gmr.pipeline_runner batch-motion amass --dry-run
python -m gvhmr_gmr.pipeline_runner batch-motion lafan1 --dry-run
python -m gvhmr_gmr.pipeline_runner video /path/to/dance.mp4 --source user_upload --dry-run
```
