from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import viser

from gvhmr_gmr.config import PROJECT_ROOT
from gvhmr_gmr.dataset_registry import build_registry, gmr_out_pkl
from gvhmr_gmr.pipeline.gmr_adapter import GMRMotionAdapter
from gvhmr_gmr.pipeline.paths_loader import load_local_paths
from gvhmr_gmr.pipeline.types import PipelinePaths
from gvhmr_gmr.pipeline.video_pipeline import PipelineResult, VideoPipelineRunner


@dataclass
class WorkbenchState:
    video_path: str = ""
    pkl_path: str = ""
    clip_name: str = "custom"
    source_group: str = "user_upload"
    static_camera: bool = True
    gvhmr_ready: bool = False
    gmr_ready: bool = False
    pipeline_running: bool = False
    aist_items: list[dict] = field(default_factory=list)


class VideoToRobotWorkbench:
    """Video / AIST++ → GVHMR → GMR → G1 Viser playback."""

    def __init__(
        self,
        gui: viser.GuiApi,
        *,
        pipeline_paths: PipelinePaths | None = None,
        on_play_pkl: Callable[[Path, str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_registry_refresh: Callable[[], None] | None = None,
    ) -> None:
        self.state = WorkbenchState()
        self._runner = VideoPipelineRunner(pipeline_paths)
        self._on_play_pkl = on_play_pkl
        self._on_status = on_status
        self._on_registry_refresh = on_registry_refresh
        self._registry = build_registry(PROJECT_ROOT)
        self._build(gui)
        self._refresh_repo_status()

    def refresh_registry(self) -> None:
        self._registry = build_registry(PROJECT_ROOT)
        self._refresh_aist_dropdown()
        self._refresh_repo_status()

    def _set_status(self, text: str) -> None:
        self.pipeline_status.content = f"**Pipeline** · {text}"
        if self._on_status:
            self._on_status(text)

    def _refresh_repo_status(self) -> None:
        lp = load_local_paths()
        self.state.gvhmr_ready = lp.gvhmr_ready
        self.state.gmr_ready = lp.gmr_ready
        if lp.gvhmr_ready and lp.gmr_ready:
            self.gvhmr_led.content = "**GVHMR** · 🟢 已配置"
            self.gmr_led.content = "**GMR** · 🟢 已配置"
        elif lp.gvhmr_ready or lp.gmr_repo:
            self.gvhmr_led.content = (
                "**GVHMR** · 🟢 已配置" if lp.gvhmr_ready else "**GVHMR** · 🔴 未配置"
            )
            self.gmr_led.content = (
                "**GMR** · 🟢 已配置" if lp.gmr_ready else "**GMR** · 🔴 未配置"
            )
        else:
            self.gvhmr_led.content = "**GVHMR** · ⚪ 未配置"
            self.gmr_led.content = "**GMR** · ⚪ 未配置"

    def _refresh_aist_dropdown(self) -> None:
        items = self._registry.get("video_samples", {}).get("aistpp_samples", [])
        self.state.aist_items = items
        names = [x["name"] for x in items]
        options = names if names else ["暂无 AIST++ 示例视频"]
        self.aist_dropdown.options = options
        if names:
            self.aist_dropdown.value = names[0]
        else:
            self.aist_dropdown.value = "暂无 AIST++ 示例视频"

    def _build(self, gui: viser.GuiApi) -> None:
        gui.add_markdown(
            """
### 🎬 视频 → G1 机器人
导入**人跳舞/做动作**的视频，或选择 **AIST++ 示例**，经 **GVHMR → GMR** 后在 Viser 看 G1 跳同款。
"""
        )

        self.source_dropdown = gui.add_dropdown(
            "输入来源",
            options=["用户上传视频", "AIST++ 示例视频"],
            initial_value="用户上传视频",
        )
        self.aist_dropdown = gui.add_dropdown(
            "AIST++ 示例视频",
            options=["暂无 AIST++ 示例视频"],
            initial_value="暂无 AIST++ 示例视频",
        )

        self.video_path = gui.add_text(
            "视频路径",
            initial_value="",
            hint="绝对路径，例如 /home/.../dance.mp4",
        )
        self.clip_name = gui.add_text("片段名称", initial_value="custom")
        self.static_cam = gui.add_checkbox("静态相机 (-s)", initial_value=True)

        self.upload_btn = gui.add_upload_button(
            "📤 上传视频",
            mime_type="video/*",
            icon=viser.Icon.UPLOAD,
        )

        self.run_pipeline_btn = gui.add_button("🔬 开始分析 (GVHMR → GMR)", color="green")
        self.pkl_path = gui.add_text(
            "GMR 输出 PKL",
            initial_value="",
        )
        self.load_pkl_btn = gui.add_button("▶ 加载 PKL 并在 Viser 播放", color="teal")

        gui.add_divider()
        gui.add_markdown("**Mocap 批处理**（需配置 GMR，数据手动放入 `datasets/`）")
        self.batch_amass_btn = gui.add_button("📦 AMASS → G1 (SMPL-X batch)", color="gray")
        self.batch_lafan1_btn = gui.add_button("📦 LAFAN1 → G1 (BVH batch)", color="gray")

        gui.add_divider()
        self.gvhmr_led = gui.add_markdown("**GVHMR** · ⚪ 未配置")
        self.gmr_led = gui.add_markdown("**GMR** · ⚪ 未配置")
        self.obs_led = gui.add_markdown("**Obs160** · ⚪ 等待")
        self.pipeline_status = gui.add_markdown("**Pipeline** · 等待导入视频")
        self.log_md = gui.add_markdown("```\n(日志)\n```")

        self._refresh_aist_dropdown()

        @self.source_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            is_aist = self.source_dropdown.value == "AIST++ 示例视频"
            self.state.source_group = "aistpp_samples" if is_aist else "user_upload"

        @self.aist_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            self._apply_aist_selection()

        @self.upload_btn.on_upload
        def _(_: viser.GuiEvent) -> None:
            f = self.upload_btn.value
            if not f.name or not f.content:
                self._set_status("上传失败：空文件")
                return
            dest = self._runner.save_upload(
                f.name,
                f.content,
                self.clip_name.value or None,
                source_group=self.state.source_group,
            )
            self.state.video_path = str(dest)
            self.video_path.value = str(dest)
            _, out_pkl = self._runner.expected_outputs_for(
                self.state.source_group, dest.stem
            )
            self.pkl_path.value = str(out_pkl)
            self._set_status(f"已保存视频 → {dest.name}")
            self._notify("视频已上传", dest.name)

        @self.run_pipeline_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            self._start_pipeline()

        @self.load_pkl_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            self._load_and_play_pkl()

        @self.batch_amass_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            self._start_motion_batch("amass")

        @self.batch_lafan1_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            self._start_motion_batch("lafan1")

    def _apply_aist_selection(self) -> None:
        name = self.aist_dropdown.value
        if name == "暂无 AIST++ 示例视频":
            return
        item = next((x for x in self.state.aist_items if x["name"] == name), None)
        if item is None:
            return
        self.state.source_group = "aistpp_samples"
        self.state.video_path = item["path"]
        self.video_path.value = item["path"]
        self.clip_name.value = name
        out_pkl = gmr_out_pkl(PROJECT_ROOT, "aistpp_samples", name)
        self.pkl_path.value = str(out_pkl)
        self._set_status(f"已选择 AIST++ 样例: {name}")

    def _notify(self, title: str, body: str) -> None:
        pass

    def _current_source_group(self) -> str:
        if self.source_dropdown.value == "AIST++ 示例视频":
            return "aistpp_samples"
        return "user_upload"

    def _start_pipeline(self) -> None:
        if self._runner.is_running:
            self._set_status("Pipeline 运行中…")
            return

        source_group = self._current_source_group()
        self.state.source_group = source_group

        if source_group == "aistpp_samples":
            self._apply_aist_selection()

        vp = Path(str(self.video_path.value)).expanduser()
        if not vp.exists():
            self._set_status("请先填写或上传有效视频路径")
            return

        lp = load_local_paths()
        if not lp.gvhmr_ready or not lp.gmr_ready:
            self._set_status(
                "外部仓库未配置。请填写 gvhmr_gmr/local_paths.yaml；"
                "或手动放置 PKL 后点「加载 PKL」/ 动作库「生成动作」分组播放。"
            )
            return

        self.state.pipeline_running = True
        self.gvhmr_led.content = "**GVHMR** · 🟡 分析中…"
        self.gmr_led.content = "**GMR** · ⚪ 等待"
        self._set_status("GVHMR → GMR 运行中（可能数分钟）…")

        def on_done(result: PipelineResult) -> None:
            self.state.pipeline_running = False
            tail = "\n".join(result.log[-8:]) if result.log else ""
            self.log_md.content = f"```\n{tail}\n```"
            if result.ok and result.gmr_pkl:
                self.state.gvhmr_ready = True
                self.state.gmr_ready = True
                self.state.pkl_path = str(result.gmr_pkl)
                self.pkl_path.value = str(result.gmr_pkl)
                self.gvhmr_led.content = "**GVHMR** · 🟢 完成"
                self.gmr_led.content = "**GMR** · 🟢 完成"
                self.obs_led.content = "**Obs160** · 🟢 可构建"
                self._set_status(result.message)
                self.refresh_registry()
                if self._on_registry_refresh:
                    self._on_registry_refresh()
                self._load_and_play_pkl()
            else:
                self.gvhmr_led.content = "**GVHMR** · 🔴 失败"
                self.gmr_led.content = "**GMR** · 🔴 失败"
                self._set_status(result.message)

        self._runner.run_async(
            vp,
            static_camera=bool(self.static_cam.value),
            source_group=source_group,
            on_done=on_done,
        )

    def _start_motion_batch(self, source_name: str) -> None:
        if self._runner.is_running:
            self._set_status("Pipeline 运行中…")
            return
        lp = load_local_paths()
        if not lp.gmr_ready:
            self._set_status("GMR 未配置，无法批处理 mocap 数据")
            return
        self._set_status(f"{source_name} 批处理运行中…")

        def _worker() -> None:
            result = self._runner.run_motion_batch(source_name, dry_run=False)
            tail = "\n".join(result.log[-6:]) if result.log else result.message
            self.log_md.content = f"```\n{tail}\n```"
            self._set_status(result.message)
            if result.ok:
                self.refresh_registry()
                if self._on_registry_refresh:
                    self._on_registry_refresh()

        import threading

        threading.Thread(target=_worker, daemon=True).start()

    def _load_and_play_pkl(self) -> None:
        pkl = Path(str(self.pkl_path.value)).expanduser()
        if not pkl.exists():
            self._set_status(f"PKL 不存在: {pkl}")
            self.gmr_led.content = "**GMR** · 🔴 文件缺失"
            return
        t0 = time.perf_counter()
        try:
            adapter = GMRMotionAdapter(pkl, target_fps=float(adapter_fps := _read_pkl_fps(pkl)))
            _qpos, fps = adapter.to_qpos_sequence(36)
            dt = (time.perf_counter() - t0) * 1000.0
            self.state.gmr_ready = True
            self.state.pkl_path = str(pkl)
            self.gmr_led.content = "**GMR** · 🟢 已加载"
            self.obs_led.content = f"**Obs160** · 🟢 OK ({dt:.0f}ms)"
            self._set_status(
                f"已加载 {pkl.name} · {adapter.motion.dof_pos.shape[0]} 帧 @ {fps:.0f}Hz"
            )
            if self._on_play_pkl:
                self._on_play_pkl(pkl, self.state.source_group)
        except Exception as exc:
            self.gmr_led.content = "**GMR** · 🔴 解析失败"
            self._set_status(str(exc))


def _read_pkl_fps(pkl: Path) -> float:
    import pickle

    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return float(data.get("fps", 60))
