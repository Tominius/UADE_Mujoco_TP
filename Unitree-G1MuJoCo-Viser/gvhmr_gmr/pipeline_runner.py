from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from gvhmr_gmr.config import PROJECT_ROOT
from gvhmr_gmr.dataset_registry import gmr_out_pkl, get_route_meta
from gvhmr_gmr.pipeline.paths_loader import LocalPaths, load_local_paths


@dataclass
class RunResult:
    ok: bool
    message: str = ""
    log: list[str] | None = None
    gvhmr_pt: Path | None = None
    gmr_pkl: Path | None = None


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
    timeout: int = 3600,
) -> tuple[int, str, str]:
    line = " ".join(shlex.quote(x) for x in cmd)
    print("[CMD]", line)
    if dry_run:
        return 0, "", ""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _conda_python(env: str, script: str) -> list[str]:
    return ["conda", "run", "-n", env, "python", script]


def _resolve_lp(project_root: Path | None = None) -> LocalPaths:
    lp = load_local_paths()
    if lp.gvhmr_repo is None or lp.gmr_repo is None:
        raise RuntimeError(
            "外部仓库未配置。请填写 gvhmr_gmr/local_paths.yaml 或设置 GVHMR_REPO / GMR_REPO。"
        )
    if not lp.gvhmr_repo.exists():
        raise RuntimeError(f"GVHMR 仓库不存在: {lp.gvhmr_repo}")
    if not lp.gmr_repo.exists():
        raise RuntimeError(f"GMR 仓库不存在: {lp.gmr_repo}")
    return lp


def _resolve_gmr_lp(*, dry_run: bool = False) -> LocalPaths:
    lp = load_local_paths()
    if lp.gmr_repo is None:
        raise RuntimeError("GMR 未配置 (gmr_repo)")
    if not dry_run and not lp.gmr_repo.exists():
        raise RuntimeError(f"GMR 仓库不存在: {lp.gmr_repo}")
    return lp


def gvhmr_video_to_pt(
    project_root: Path,
    video_path: Path,
    *,
    dry_run: bool = False,
    lp: LocalPaths | None = None,
) -> Path:
    lp = lp or _resolve_lp(project_root)
    demo = lp.gvhmr_repo / "tools" / "demo" / "demo.py"
    if not demo.exists() and not dry_run:
        raise FileNotFoundError(f"GVHMR demo 不存在: {demo}")

    cmd = _conda_python(lp.gvhmr_env, str(demo)) + [f"--video={video_path}"]
    if lp.use_static_camera:
        cmd.append("-s")

    code, out, err = run_cmd(cmd, cwd=lp.gvhmr_repo, dry_run=dry_run)
    if code != 0 and not dry_run:
        raise RuntimeError(f"GVHMR 失败 (code {code}): {err[-500:]}")

    return lp.gvhmr_repo / "outputs" / "demo" / video_path.stem / "hmr4d_results.pt"


def gmr_from_gvhmr_pt(
    project_root: Path,
    pt_path: Path,
    out_pkl: Path,
    *,
    dry_run: bool = False,
    lp: LocalPaths | None = None,
) -> Path:
    lp = lp or _resolve_lp(project_root)
    out_pkl = Path(out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    script = lp.gmr_repo / "scripts" / "gvhmr_to_robot.py"
    if not script.exists() and not dry_run:
        raise FileNotFoundError(f"GMR gvhmr_to_robot.py 不存在: {script}")

    cmd = _conda_python(lp.gmr_env, str(script)) + [
        "--gvhmr_pred_file",
        str(pt_path),
        "--robot",
        lp.robot_name,
        "--save_path",
        str(out_pkl),
    ]
    code, out, err = run_cmd(cmd, cwd=lp.gmr_repo, dry_run=dry_run)
    if code != 0 and not dry_run:
        raise RuntimeError(f"GMR 失败 (code {code}): {err[-500:]}")
    return out_pkl


def run_video_route(
    project_root: Path,
    video_path: Path,
    source_group: str = "user_upload",
    *,
    static_camera: bool | None = None,
    dry_run: bool = False,
) -> RunResult:
    video_path = Path(video_path).expanduser().resolve()
    log: list[str] = []
    try:
        lp = load_local_paths()
        if static_camera is not None:
            lp = LocalPaths(
                gvhmr_repo=lp.gvhmr_repo,
                gmr_repo=lp.gmr_repo,
                gvhmr_env=lp.gvhmr_env,
                gmr_env=lp.gmr_env,
                gvhmr_python=lp.gvhmr_python,
                gmr_python=lp.gmr_python,
                robot_name=lp.robot_name,
                use_static_camera=static_camera,
            )
        if not lp.gvhmr_ready or not lp.gmr_ready:
            return RunResult(
                ok=False,
                message="外部仓库未配置或路径不存在。可手动放置 PKL 到 artifacts/gmr/ 后从动作库播放。",
                log=log,
            )

        out_pkl = gmr_out_pkl(project_root, source_group, video_path.stem)
        meta = get_route_meta(source_group, project_root)
        gvhmr_artifact = (
            Path(meta["gvhmr_out_dir"]) / video_path.stem / "hmr4d_results.pt"
            if meta and "gvhmr_out_dir" in meta
            else None
        )

        pt_path = gvhmr_video_to_pt(project_root, video_path, dry_run=dry_run, lp=lp)
        log.append(f"GVHMR pt: {pt_path}")

        if not dry_run and pt_path.exists() and gvhmr_artifact is not None:
            gvhmr_artifact.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pt_path, gvhmr_artifact)
            pt_path = gvhmr_artifact

        gmr_from_gvhmr_pt(project_root, pt_path, out_pkl, dry_run=dry_run, lp=lp)
        log.append(f"GMR pkl: {out_pkl}")

        if not dry_run and not out_pkl.exists():
            return RunResult(ok=False, message=f"GMR 未生成 PKL: {out_pkl}", log=log)

        return RunResult(
            ok=True,
            message=f"完成: {out_pkl.name}",
            log=log,
            gvhmr_pt=pt_path,
            gmr_pkl=out_pkl,
        )
    except Exception as exc:
        log.append(str(exc))
        return RunResult(ok=False, message=str(exc), log=log)


def gmr_from_smplx_batch(
    project_root: Path,
    src_dir: Path,
    dst_dir: Path,
    *,
    dry_run: bool = False,
    lp: LocalPaths | None = None,
) -> None:
    lp = lp or _resolve_gmr_lp(dry_run=dry_run)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    script = (lp.gmr_repo or Path(".")) / "scripts" / "smplx_to_robot_dataset.py"
    cmd = _conda_python(lp.gmr_env, str(script)) + [
        "--src_folder",
        str(src_dir),
        "--tgt_folder",
        str(dst_dir),
        "--robot",
        lp.robot_name,
    ]
    code, _, err = run_cmd(cmd, cwd=lp.gmr_repo or Path("."), dry_run=dry_run)
    if code != 0 and not dry_run:
        raise RuntimeError(f"SMPL-X batch 失败: {err[-500:]}")


def gmr_from_bvh_batch(
    project_root: Path,
    src_dir: Path,
    dst_dir: Path,
    *,
    bvh_format: str = "lafan1",
    dry_run: bool = False,
    lp: LocalPaths | None = None,
) -> None:
    lp = lp or _resolve_gmr_lp(dry_run=dry_run)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    script = (lp.gmr_repo or Path(".")) / "scripts" / "bvh_to_robot_dataset.py"
    cmd = _conda_python(lp.gmr_env, str(script)) + [
        "--src_folder",
        str(src_dir),
        "--tgt_folder",
        str(dst_dir),
        "--robot",
        lp.robot_name,
    ]
    if bvh_format:
        cmd.extend(["--format", bvh_format])
    code, _, err = run_cmd(cmd, cwd=lp.gmr_repo or Path("."), dry_run=dry_run)
    if code != 0 and not dry_run:
        raise RuntimeError(f"BVH batch 失败: {err[-500:]}")


def fbx_to_robot_single(
    project_root: Path,
    fbx_path: Path,
    extracted_motion_pkl: Path,
    out_pkl: Path,
    *,
    root_joint: str = "Hips",
    fps: int = 120,
    dry_run: bool = False,
    lp: LocalPaths | None = None,
) -> Path:
    lp = lp or _resolve_lp(project_root)
    extracted_motion_pkl = Path(extracted_motion_pkl)
    out_pkl = Path(out_pkl)
    extracted_motion_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    import_cmd = _conda_python(
        lp.gmr_env,
        str(lp.gmr_repo / "third_party" / "poselib" / "fbx_importer.py"),
    ) + [
        "--input",
        str(fbx_path),
        "--output",
        str(extracted_motion_pkl),
        "--root-joint",
        root_joint,
        "--fps",
        str(fps),
    ]
    code, _, err = run_cmd(import_cmd, cwd=lp.gmr_repo / "third_party", dry_run=dry_run)
    if code != 0 and not dry_run:
        raise RuntimeError(f"FBX import 失败: {err[-500:]}")

    retarget_cmd = _conda_python(
        lp.gmr_env,
        str(lp.gmr_repo / "scripts" / "fbx_offline_to_robot.py"),
    ) + [
        "--motion_file",
        str(extracted_motion_pkl),
        "--robot",
        lp.robot_name,
        "--save_path",
        str(out_pkl),
    ]
    code, _, err = run_cmd(retarget_cmd, cwd=lp.gmr_repo, dry_run=dry_run)
    if code != 0 and not dry_run:
        raise RuntimeError(f"FBX retarget 失败: {err[-500:]}")
    return out_pkl


def run_motion_batch(
    project_root: Path,
    source_name: str,
    *,
    dry_run: bool = False,
) -> RunResult:
    meta = get_route_meta(source_name, project_root)
    if meta is None:
        return RunResult(ok=False, message=f"未知 motion source: {source_name}")

    src = Path(meta["input_dir"])
    dst = Path(meta["gmr_out_dir"])
    route = meta.get("route", "")
    log: list[str] = []

    try:
        if route == "gmr_smplx_batch":
            gmr_from_smplx_batch(project_root, src, dst, dry_run=dry_run)
        elif route == "gmr_bvh_batch":
            gmr_from_bvh_batch(
                project_root,
                src,
                dst,
                bvh_format=meta.get("bvh_format", "lafan1"),
                dry_run=dry_run,
            )
        else:
            return RunResult(ok=False, message=f"不支持的 batch route: {route}")

        return RunResult(
            ok=True,
            message=f"批处理命令已{'打印' if dry_run else '执行'}: {source_name} → {dst}",
            log=log,
        )
    except Exception as exc:
        return RunResult(ok=False, message=str(exc), log=log)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="GVHMR/GMR pipeline runner")
    sub = parser.add_subparsers(dest="cmd")

    v = sub.add_parser("video", help="video → GVHMR → GMR → PKL")
    v.add_argument("video", type=Path)
    v.add_argument("--source", default="user_upload")
    v.add_argument("--dry-run", action="store_true")

    b = sub.add_parser("batch-smplx", help="AMASS/SMPL-X batch")
    b.add_argument("--src", type=Path, required=True)
    b.add_argument("--dst", type=Path, required=True)
    b.add_argument("--dry-run", action="store_true")

    bvh = sub.add_parser("batch-bvh", help="LAFAN1/BVH batch")
    bvh.add_argument("--src", type=Path, required=True)
    bvh.add_argument("--dst", type=Path, required=True)
    bvh.add_argument("--format", default="lafan1")
    bvh.add_argument("--dry-run", action="store_true")

    mb = sub.add_parser("batch-motion", help="run configured motion route by name")
    mb.add_argument("source", help="e.g. amass, lafan1")
    mb.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    root = PROJECT_ROOT

    if args.cmd == "video":
        r = run_video_route(root, args.video, args.source, dry_run=args.dry_run)
        print(r.message)
        return 0 if r.ok else 1
    if args.cmd == "batch-smplx":
        gmr_from_smplx_batch(root, args.src, args.dst, dry_run=args.dry_run)
        return 0
    if args.cmd == "batch-bvh":
        gmr_from_bvh_batch(
            root, args.src, args.dst, bvh_format=args.format, dry_run=args.dry_run
        )
        return 0
    if args.cmd == "batch-motion":
        r = run_motion_batch(root, args.source, dry_run=args.dry_run)
        print(r.message)
        return 0 if r.ok else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
