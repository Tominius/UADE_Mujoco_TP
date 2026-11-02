from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from gvhmr_gmr.config import ARTIFACTS, GVHMR_OUT, PROJECT_ROOT
from gvhmr_gmr.pipeline.types import PipelinePaths
from gvhmr_gmr.pipeline_runner import run_motion_batch, run_video_route


@dataclass
class PipelineResult:
    ok: bool
    video_path: Path | None = None
    gvhmr_pt: Path | None = None
    gmr_pkl: Path | None = None
    message: str = ""
    log: list[str] = field(default_factory=list)


class VideoPipelineRunner:
    """Offline video → GVHMR → GMR → PKL pipeline."""

    def __init__(self, paths: PipelinePaths | None = None) -> None:
        self.paths = paths or PipelinePaths()
        self._lock = threading.Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def ingest_video(
        self,
        src: Path,
        clip_name: str | None = None,
        *,
        source_group: str = "user_upload",
    ) -> Path:
        from gvhmr_gmr.dataset_registry import get_route_meta

        src = Path(src).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        meta = get_route_meta(source_group, PROJECT_ROOT)
        if meta and "input_dir" in meta:
            dest_dir = Path(meta["input_dir"])
        else:
            dest_dir = ARTIFACTS / "videos"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if src != dest:
            shutil.copy2(src, dest)
        return dest

    def save_upload(
        self,
        filename: str,
        data: bytes,
        clip_name: str | None = None,
        *,
        source_group: str = "user_upload",
    ) -> Path:
        from gvhmr_gmr.dataset_registry import get_route_meta

        meta = get_route_meta(source_group, PROJECT_ROOT)
        if meta and "input_dir" in meta:
            dest_dir = Path(meta["input_dir"])
        else:
            name = clip_name or Path(filename).stem or "upload"
            dest_dir = ARTIFACTS / "videos" / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / (Path(filename).name or "upload.mp4")
        dest.write_bytes(data)
        return dest

    def expected_outputs_for(self, source_group: str, clip_name: str) -> tuple[Path, Path]:
        meta = get_route_meta(source_group, PROJECT_ROOT)
        if meta and "gvhmr_out_dir" in meta:
            gvhmr_pt = Path(meta["gvhmr_out_dir"]) / clip_name / "hmr4d_results.pt"
        else:
            gvhmr_pt = GVHMR_OUT / clip_name / "hmr4d_results.pt"
        gmr_pkl = gmr_out_pkl(PROJECT_ROOT, source_group, clip_name)
        return gvhmr_pt, gmr_pkl

    def expected_outputs(self, clip_name: str) -> tuple[Path, Path]:
        return self.expected_outputs_for("user_upload", clip_name)

    def run(
        self,
        video_path: Path,
        *,
        static_camera: bool = True,
        clip_name: str | None = None,
        source_group: str = "user_upload",
    ) -> PipelineResult:
        video_path = Path(video_path).resolve()
        clip = clip_name or video_path.stem
        log: list[str] = []
        result = PipelineResult(ok=False, video_path=video_path, log=log)

        if not video_path.exists():
            result.message = f"视频不存在: {video_path}"
            return result

        out = run_video_route(
            PROJECT_ROOT,
            video_path,
            source_group,
            static_camera=static_camera,
            dry_run=False,
        )
        result.log.extend(out.log or [])
        result.message = out.message
        result.ok = out.ok
        if out.gvhmr_pt:
            result.gvhmr_pt = out.gvhmr_pt
        if out.gmr_pkl:
            result.gmr_pkl = out.gmr_pkl
        elif out.ok:
            _, gmr_pkl = self.expected_outputs_for(source_group, clip)
            result.gmr_pkl = gmr_pkl
        return result

    def run_motion_batch(
        self,
        source_name: str,
        *,
        dry_run: bool = False,
    ) -> PipelineResult:
        out = run_motion_batch(PROJECT_ROOT, source_name, dry_run=dry_run)
        return PipelineResult(
            ok=out.ok,
            message=out.message,
            log=out.log or [],
        )

    def run_async(
        self,
        video_path: Path,
        *,
        static_camera: bool = True,
        source_group: str = "user_upload",
        on_done: Callable[[PipelineResult], None] | None = None,
    ) -> None:
        def _worker() -> None:
            with self._lock:
                self._running = True
            try:
                out = self.run(
                    video_path,
                    static_camera=static_camera,
                    source_group=source_group,
                )
            finally:
                with self._lock:
                    self._running = False
            if on_done:
                on_done(out)

        threading.Thread(target=_worker, daemon=True).start()
