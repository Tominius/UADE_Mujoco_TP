from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import viser

from pipeline.blending import blend_qpos
from pipeline.motion_amplifier import AmplifyConfig, amplify_qpos
from pipeline.motion_analyzer import MotionStats, compute_intensity_raw, make_motion_stats
from pipeline.motion_bank import MotionClip, group_by_category, scan_motion_bank
from pipeline.motion_loader import load_qpos_csv
from pipeline.motion_manager import MotionEntry, MotionManager, PlaybackMode
from pipeline.motion_streamer import MotionStreamer
from pipeline.policy_playback import load_qpos_from_bundle
from pipeline.policy_discovery import PolicyBundle, find_policy_bundle, scan_policy_bundles
from pipeline.mujoco_scene import (
    MujocoViserScene,
    SceneConfig,
    load_standing_qpos_from_csv,
)
from gvhmr_gmr.config import PROJECT_ROOT
from gvhmr_gmr.dataset_registry import build_registry
from gvhmr_gmr.pipeline.paths_loader import load_pipeline_paths
from gvhmr_gmr.ui.workbench import VideoToRobotWorkbench
from pipeline.motion_manager import load_pkl_clip
from ui.cards import render_motion_card
from ui.theme import apply_gallery_theme


@dataclass(frozen=True)
class AppConfig:
    csv_dir: str = "g1_moves/g1_moves_outputs/web_csv"
    policy_data_root: str = "g1_moves/g1_moves_data"
    model_path: str = "assets/proto_g1/g1_holo_compat_web.xml"
    policy_model_path: str = "assets/proto_g1/g1_policy_mode15_web.xml"
    host: str = "0.0.0.0"
    port: int = 9878
    render_fps: float = 60.0
    clip_fps: float = 60.0
    preview_seconds: float = 1.5
    default_speed: float = 1.0
    amplify: AmplifyConfig = field(default_factory=AmplifyConfig)
    use_mj_step: bool = False
    blend_seconds: float = 0.35
    default_mode: str = "qpos"
    startup_clip: str | None = None


@dataclass
class ClipAsset:
    clip: MotionClip
    qpos_raw: np.ndarray
    qpos_amp: np.ndarray
    stats: MotionStats
    policy: PolicyBundle | None = None


class G1MovesWebGalleryFinal:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

        tmp_model = mujoco.MjModel.from_xml_path(str(Path(config.model_path).resolve()))
        default_qpos = load_standing_qpos_from_csv(config.csv_dir, tmp_model)

        self._scene_config = SceneConfig(
            model_path=config.model_path,
            host=config.host,
            port=config.port,
            use_mj_step=config.use_mj_step,
            default_qpos=tuple(float(x) for x in default_qpos),
        )
        self.scene = MujocoViserScene(self._scene_config)
        self.server = self.scene.server
        self.gui = self.server.gui
        apply_gallery_theme(self.server)

        self.streamer = MotionStreamer(clip_fps=config.clip_fps)
        self._loop_should_stop = threading.Event()
        self._catalog_lock = threading.RLock()
        self._assets: dict[str, ClipAsset] = {}
        self._clip_order: list[str] = []
        self._current_key: str | None = None
        self._current_preview: bool = False
        self._display_qpos: np.ndarray | None = None
        self._blend_from: np.ndarray | None = None
        self._blend_to: np.ndarray | None = None
        self._blend_t: float = 1.0
        self._blend_duration: float = config.blend_seconds
        self._policies: dict[str, PolicyBundle] = {}
        self._policy_playing: bool = False
        self._gmr_playing: bool = False
        self._custom_video_mode: bool = False
        self._custom_clip_label: str = "自定义视频"
        self._workbench: VideoToRobotWorkbench | None = None
        self._generated_registry: dict = {}
        self._generated_dropdowns: dict[str, object] = {}

        self._scan_and_prepare_assets()
        self._build_gui()
        if config.startup_clip:
            self._try_startup_clip(config.startup_clip, config.default_mode)

    @property
    def nq(self) -> int:
        return int(self.scene.model.nq)

    def _scan_and_prepare_assets(self) -> None:
        clips = scan_motion_bank(self.config.csv_dir)
        raw_scores: list[float] = []
        preload: list[tuple[MotionClip, np.ndarray, np.ndarray, float]] = []

        for clip in clips:
            qpos = load_qpos_csv(clip.csv_path, expected_nq=self.nq)
            raw = compute_intensity_raw(qpos)
            amp = amplify_qpos(qpos, config=self.config.amplify)
            raw_scores.append(raw)
            preload.append((clip, qpos, amp, raw))

        if not preload:
            raise RuntimeError("动作库为空，无法启动 Gallery。")

        raw_arr = np.asarray(raw_scores, dtype=np.float64)
        q_low, q_mid = np.quantile(raw_arr, [1.0 / 3.0, 2.0 / 3.0])
        raw_min = float(raw_arr.min())
        raw_max = float(raw_arr.max())
        thresholds = (float(q_low), float(q_mid))

        assets: dict[str, ClipAsset] = {}
        clip_order: list[str] = []
        for clip, qpos, amp, _ in preload:
            stats = make_motion_stats(
                qpos,
                fps=self.config.clip_fps,
                thresholds=thresholds,
                raw_min=raw_min,
                raw_max=raw_max,
            )
            policy = find_policy_bundle(clip.source_name, self.config.policy_data_root)
            assets[clip.key] = ClipAsset(
                clip=clip,
                qpos_raw=qpos,
                qpos_amp=amp,
                stats=stats,
                policy=policy,
            )
            clip_order.append(clip.key)

        self._assets = assets
        self._clip_order = clip_order
        self._display_qpos = self.scene._default_qpos.copy()
        self._policies = scan_policy_bundles(self.config.policy_data_root)
        manager_entries = {
            key: MotionEntry(
                key=key,
                display_name=asset.clip.display_name,
                category=asset.clip.category,
                qpos=asset.qpos_raw,
                policy_bundle=asset.policy,
            )
            for key, asset in assets.items()
        }
        self._motion_manager = MotionManager(manager_entries, clip_order)
        policy_count = self._motion_manager.policy_count()
        print(f"[G1 Moves] 已匹配 Policy 动作数: {policy_count}")

    def _build_gui(self) -> None:
        self.gui.reset()
        apply_gallery_theme(self.server)
        self.gui.set_panel_label("G1 Studio")

        self.status_md = self.gui.add_markdown("**状态** · 就绪")
        self.frame_md = self.gui.add_markdown("**帧** · 0 / 0")
        policy_n = self._motion_manager.policy_count() if self._motion_manager else 0
        self.mode_md = self.gui.add_markdown(
            f"**模式** · 展示 (qpos) · 可用 Policy **{policy_n}** 个"
        )
        self.gui.add_divider()

        root_tabs = self.gui.add_tab_group()
        with root_tabs.add_tab("🎬 视频 → G1"):
            self._build_video_workbench_tab()
        with root_tabs.add_tab("📚 动作库"):
            self._build_gallery_tab()

        self.streamer.set_speed(self.config.default_speed)

    def _build_video_workbench_tab(self) -> None:
        self._workbench = VideoToRobotWorkbench(
            self.gui,
            pipeline_paths=load_pipeline_paths(),
            on_play_pkl=self.play_generated_pkl,
            on_status=self._set_global_status,
            on_registry_refresh=self._refresh_generated_groups,
        )
        if self._workbench:
            self._workbench._notify = self._notify

    def _refresh_generated_groups(self) -> None:
        self._generated_registry = build_registry(PROJECT_ROOT)
        labels = self._generated_registry.get("group_labels") or {}
        groups = self._generated_registry.get("generated_groups") or {}
        display = [
            ("user_upload", labels.get("user_upload", "用户视频生成")),
            ("aistpp_samples", labels.get("aistpp_samples", "AIST++ 生成")),
            ("amass", labels.get("amass", "AMASS 生成")),
            ("lafan1", labels.get("lafan1", "LAFAN1 生成")),
        ]
        for source, _label in display:
            items = groups.get(source, [])
            names = [x["name"] for x in items]
            dd = self._generated_dropdowns.get(source)
            if dd is None:
                continue
            options = names if names else [f"暂无生成动作"]
            dd.options = options
            dd.value = options[0]

    def _build_gallery_tab(self) -> None:
        self.info_md = self.gui.add_markdown(
            """
### 📚 G1 Moves 动作库
内置 **25** 个官方展示动作（CSV 轨迹回放）
"""
        )

        with self.gui.add_folder("▶ 播放控制", order=0):
            self.pause_btn = self.gui.add_button("⏸ 暂停 / 继续", color="blue")
            self.stop_btn = self.gui.add_button("⏹ 停止", color="gray")
            self.next_btn = self.gui.add_button("⏭ 下一个", color="blue")
            self.loop_checkbox = self.gui.add_checkbox("🔁 循环播放", initial_value=True)
            self.speed_slider = self.gui.add_slider(
                "⚡ 速度倍率",
                min=0.25,
                max=2.0,
                step=0.05,
                initial_value=self.config.default_speed,
            )
            self.mj_step_checkbox = self.gui.add_checkbox(
                "🧪 启用 mj_step（动力学）",
                initial_value=self.config.use_mj_step,
            )

        @self.pause_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            if self.streamer.is_playing():
                self.streamer.pause()
                self.status_md.content = "**状态** · 已暂停"
            else:
                self.streamer.play()
                label = self._playback_label()
                self.status_md.content = f"**状态** · 继续播放 ({label}) · {self._current_display_name()}"

        @self.stop_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            self._stop_policy_playback()
            self.streamer.stop()
            self._start_blend_to_default()
            self.status_md.content = "**状态** · 已停止"
            self.frame_md.content = "**帧** · 0 / 0"
            policy_n = self._motion_manager.policy_count() if self._motion_manager else 0
            self.mode_md.content = f"**模式** · 展示 (qpos) · 可用 Policy **{policy_n}** 个"

        @self.next_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            self.play_next(loop=bool(self.loop_checkbox.value))

        @self.speed_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self.streamer.set_speed(float(self.speed_slider.value))

        @self.mj_step_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            use_step = bool(self.mj_step_checkbox.value)
            self._scene_config = SceneConfig(
                model_path=self._scene_config.model_path,
                host=self._scene_config.host,
                port=self._scene_config.port,
                use_mj_step=use_step,
                default_qpos=tuple(self.scene._default_qpos)
                if self.scene._default_qpos is not None
                else None,
            )
            self.scene.config = self._scene_config
            mode = "动力学 (mj_step)" if use_step else "展示 (qpos-gallery)"
            self.mode_md.content = f"**模式** · {mode}"

        self._generated_registry = build_registry(PROJECT_ROOT)
        labels = self._generated_registry.get("group_labels") or {}
        display_groups = [
            ("user_upload", labels.get("user_upload", "用户视频生成")),
            ("aistpp_samples", labels.get("aistpp_samples", "AIST++ 生成")),
            ("amass", labels.get("amass", "AMASS 生成")),
            ("lafan1", labels.get("lafan1", "LAFAN1 生成")),
        ]

        with self.gui.add_folder("✨ 生成动作库", order=1):
            self.gui.add_markdown(
                "来自 **视频→G1** 或 **AMASS/LAFAN1 批处理** 的 PKL（`artifacts/gmr/`）"
            )
            refresh_gen_btn = self.gui.add_button("🔄 刷新生成库", color="gray")

            @refresh_gen_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                self._refresh_generated_groups()
                self._notify("已刷新", "生成动作库")

            for source, label in display_groups:
                items = self._generated_registry.get("generated_groups", {}).get(source, [])
                names = [x["name"] for x in items]
                options = names if names else ["暂无生成动作"]
                dd = self.gui.add_dropdown(f"{label}", options=options, initial_value=options[0])
                self._generated_dropdowns[source] = dd
                play_gen_btn = self.gui.add_button(f"▶ 播放 · {label}", color="teal")

                @play_gen_btn.on_click
                def _(_: viser.GuiEvent, src=source, dropdown=dd) -> None:
                    name = dropdown.value
                    if name == "暂无生成动作":
                        self._notify("无动作", "请先生成或手动放置 PKL")
                        return
                    groups = self._generated_registry.get("generated_groups", {}).get(src, [])
                    item = next((x for x in groups if x["name"] == name), None)
                    if item is None:
                        self._refresh_generated_groups()
                        groups = self._generated_registry.get("generated_groups", {}).get(src, [])
                        item = next((x for x in groups if x["name"] == name), None)
                    if item is None:
                        self._notify("未找到", name)
                        return
                    self.play_generated_pkl(Path(item["path"]), src)

        grouped = group_by_category(asset.clip for asset in self._assets.values())
        sorted_categories = [c for c in ["舞蹈", "武术", "手势", "其他"] if c in grouped]
        category_icons = {"舞蹈": "🕺", "武术": "🥋", "手势": "🤝", "其他": "🎯"}

        tab_group = self.gui.add_tab_group()
        for category in sorted_categories:
            icon = category_icons.get(category, "🎯")
            with tab_group.add_tab(f"{icon} {category}"):
                for clip in grouped[category]:
                    asset = self._assets[clip.key]
                    with self.gui.add_folder(asset.clip.display_name, order=0):
                        self.gui.add_markdown(
                            render_motion_card(
                                asset.clip, asset.stats, has_policy=asset.policy is not None
                            )
                        )
                        play_btn = self.gui.add_button("▶ 展示播放", color="blue")
                        preview_btn = self.gui.add_button("👁 预览 1.5s", color="gray")

                        @play_btn.on_click
                        def _(_: viser.GuiEvent, key=asset.clip.key) -> None:
                            self.play_key(key, preview=False, loop=bool(self.loop_checkbox.value))

                        @preview_btn.on_click
                        def _(_: viser.GuiEvent, key=asset.clip.key) -> None:
                            self.play_key(key, preview=True, loop=False)

    def _playback_label(self) -> str:
        if self._custom_video_mode:
            return "视频→G1"
        if self._gmr_playing:
            return "GMR"
        if self._policy_playing:
            return "Policy"
        return "展示"

    def _set_global_status(self, text: str) -> None:
        if hasattr(self, "status_md"):
            self.status_md.content = f"**状态** · {text}"

    def _notify(self, title: str, body: str) -> None:
        try:
            self.server.add_notification(title=title, body=body, auto_close_seconds=2.5)
        except Exception:
            pass

    def _current_display_name(self) -> str:
        if self._custom_video_mode:
            return self._custom_clip_label
        if self._current_key is None:
            return "无"
        return self._assets[self._current_key].clip.display_name

    def _start_blend_to_default(self) -> None:
        if self.scene._default_qpos is None or self._display_qpos is None:
            self.scene.reset_pose()
            return
        self._blend_from = self._display_qpos.copy()
        self._blend_to = self.scene._default_qpos.copy()
        self._blend_t = 0.0

    def _start_blend_to(self, target_qpos: np.ndarray) -> None:
        if self._display_qpos is None:
            self._display_qpos = target_qpos.copy()
            return
        self._blend_from = self._display_qpos.copy()
        self._blend_to = target_qpos.copy()
        self._blend_t = 0.0

    def _stop_policy_playback(self) -> None:
        self._policy_playing = False
        self._gmr_playing = False
        self._custom_video_mode = False

    def _stop_policy(self) -> None:
        self._stop_policy_playback()
        if self._motion_manager is not None and self._motion_manager.mode == PlaybackMode.POLICY:
            self._motion_manager.stop_motion()

    def _try_startup_clip(self, clip_name: str, mode: str) -> None:
        if self._motion_manager is None:
            return
        key = self._motion_manager.find_key(clip_name)
        if key is None:
            print(f"[G1 Moves] 未找到启动 clip: {clip_name}")
            return
        if mode == "policy":
            self.play_policy_key(key)
        else:
            self.play_key(key, preview=False, loop=True)

    def play_policy_key(self, key: str) -> None:
        """Stable Policy playback: NPZ training reference on holo model (same path as qpos)."""
        with self._catalog_lock:
            if self._motion_manager is None:
                return
            entry = self._motion_manager.get(key)
            if entry.policy_bundle is None:
                self._notify("无 Policy 资产", entry.display_name)
                return
            self._motion_manager.play_motion(key, mode=PlaybackMode.POLICY)
            self._stop_policy_playback()
            self.streamer.stop()

            holo = Path(self.config.model_path).resolve()
            if Path(self.scene.config.model_path).resolve() != holo:
                self.scene.reload_model(str(holo))

            qpos, fps = load_qpos_from_bundle(entry.policy_bundle, self.nq)
            self.streamer.set_clip_fps(fps)
            self.streamer.set_speed(float(self.speed_slider.value))
            self.streamer.load(qpos, loop=bool(self.loop_checkbox.value))
            self._policy_playing = True
            self._current_key = key
            self._current_preview = False
            self._start_blend_to(qpos[0])
            num_frames = int(qpos.shape[0])
            self.status_md.content = f"**状态** · Policy 播放 · {entry.display_name}"
            self.frame_md.content = f"**帧** · 1 / {num_frames}"
            self.mode_md.content = (
                "**模式** · Policy 轨迹 (NPZ 训练参考 · Viser 稳定回放)"
            )
            self._notify("Policy 已启动", entry.display_name)

    def play_generated_pkl(self, pkl_path: Path, source_group: str = "user_upload") -> None:
        """Play GMR PKL using native fps from the clip."""
        from gvhmr_gmr.pipeline.gmr_adapter import GMRMotionAdapter

        with self._catalog_lock:
            clip = load_pkl_clip(pkl_path, pkl_path.stem, source_group)
            if self._motion_manager is not None:
                self._motion_manager.load_retarget_clip(clip)

            self._stop_policy_playback()
            self.streamer.stop()

            holo = Path(self.config.model_path).resolve()
            if Path(self.scene.config.model_path).resolve() != holo:
                self.scene.reload_model(str(holo))

            qpos, fps = GMRMotionAdapter(
                pkl_path, target_fps=clip.fps
            ).to_qpos_sequence(self.nq)
            self.streamer.set_clip_fps(fps)
            loop = bool(self.loop_checkbox.value) if hasattr(self, "loop_checkbox") else True
            speed = float(self.speed_slider.value) if hasattr(self, "speed_slider") else 1.0
            self.streamer.set_speed(speed)
            self.streamer.load(qpos, loop=loop)
            self._custom_video_mode = True
            self._custom_clip_label = clip.name
            self._current_key = None
            self._start_blend_to(qpos[0])

            group_labels = self._generated_registry.get("group_labels") or {}
            label = group_labels.get(source_group, source_group)
            self.status_md.content = f"**状态** · {label} · {pkl_path.name}"
            self.frame_md.content = f"**帧** · 1 / {qpos.shape[0]}"
            self.mode_md.content = f"**模式** · 🎬 生成动作 ({label} · {clip.fps:.0f} Hz)"
            self._notify("G1 动作已加载", pkl_path.name)

    def play_custom_gmr_pkl(self, pkl_path: Path) -> None:
        self.play_generated_pkl(pkl_path, "user_upload")

    def play_key(self, key: str, preview: bool, loop: bool) -> None:
        with self._catalog_lock:
            self._stop_policy_playback()
            if self._motion_manager is not None:
                self._motion_manager.play_motion(key, mode=PlaybackMode.QPOS)
            if Path(self.config.model_path).resolve() != Path(
                self.scene.config.model_path
            ).resolve():
                self.scene.reload_model(self.config.model_path)
            if key not in self._assets:
                raise KeyError(f"未知动作 key: {key}")
            asset = self._assets[key]
            self._current_key = key
            self._current_preview = bool(preview)
            preview_seconds = self.config.preview_seconds if preview else None
            self.streamer.set_clip_fps(self.config.clip_fps)
            self.streamer.load(asset.qpos_raw, loop=loop, preview_seconds=preview_seconds)
            self._start_blend_to(asset.qpos_raw[0])
            label = "预览" if preview else "播放"
            self.status_md.content = f"**状态** · 正在{label} · {asset.clip.display_name}"
            self.frame_md.content = f"**帧** · 0 / {asset.stats.num_frames}"
            self._notify(f"已切换动作", asset.clip.display_name)
            policy_n = self._motion_manager.policy_count() if self._motion_manager else 0
            self.mode_md.content = f"**模式** · 展示 (qpos) · 可用 Policy **{policy_n}** 个"

    def play_next(self, loop: bool) -> None:
        if not self._clip_order:
            return
        if self._current_key is None:
            self.play_key(self._clip_order[0], preview=False, loop=loop)
            return
        idx = self._clip_order.index(self._current_key)
        next_key = self._clip_order[(idx + 1) % len(self._clip_order)]
        self.play_key(next_key, preview=False, loop=loop)

    def _update_blend(self, dt: float) -> None:
        if self._blend_from is None or self._blend_to is None:
            return
        if self._blend_t >= 1.0:
            return
        duration = max(self._blend_duration, 1e-6)
        self._blend_t = min(1.0, self._blend_t + dt / duration)
        self._display_qpos = blend_qpos(self._blend_from, self._blend_to, self._blend_t)
        self.scene.apply_qpos(self._display_qpos)

    def run_forever(self) -> None:
        dt = 1.0 / max(self.config.render_fps, 1e-6)
        last_log = time.time()
        print(f"[G1 Moves] 打开浏览器: http://localhost:{self.config.port}")
        print(f"[G1 Moves] 已扫描动作数: {len(self._assets)}")

        while not self._loop_should_stop.is_set():
            self._update_blend(dt)
            frame = self.streamer.next_frame(self.config.render_fps)
            if frame is not None:
                self._display_qpos = frame.qpos.copy()
                self.scene.apply_qpos(frame.qpos)
                self._blend_t = 1.0
                self.frame_md.content = (
                    f"**帧** · {frame.frame_index + 1} / {frame.num_frames}"
                )
                if frame.finished and frame.preview_mode:
                    self.status_md.content = (
                        f"**状态** · 预览结束 · {self._current_display_name()}"
                    )

            now = time.time()
            if now - last_log > 5.0:
                mode = "video_g1" if self._custom_video_mode else (
                    "gmr" if self._gmr_playing else ("policy" if self._policy_playing else "qpos")
                )
                print(
                    f"[G1 Moves] clip={self._current_display_name()} "
                    f"mode={mode} playing={self.streamer.is_playing()} "
                    f"speed={self.streamer.get_speed():.2f}"
                )
                last_log = now

            time.sleep(dt)

    def stop_forever(self) -> None:
        self._loop_should_stop.set()
