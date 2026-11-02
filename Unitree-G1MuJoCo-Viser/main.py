from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.dashboard import AppConfig, G1MovesWebGalleryFinal


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G1 Moves Web Gallery")
    parser.add_argument(
        "--mode",
        choices=("qpos", "policy"),
        default="qpos",
        help="qpos=展示轨迹；policy=启动后自动跑指定 clip 的 ONNX 动力学",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="policy 模式下自动启动的动作名，如 B_DadDance",
    )
    parser.add_argument("--csv-dir", default="g1_moves/g1_moves_outputs/web_csv")
    parser.add_argument("--policy-data-root", default="g1_moves/g1_moves_data")
    parser.add_argument("--model", default="assets/proto_g1/g1_holo_compat_web.xml")
    parser.add_argument(
        "--policy-model",
        default="assets/proto_g1/g1_policy_mode15_web.xml",
        help="Policy 动力学专用 MuJoCo XML（含 imu_ang_vel / imu_lin_vel）",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9878)
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument("--clip-fps", type=float, default=60.0)
    parser.add_argument("--preview-seconds", type=float, default=1.5)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--joint-scale", type=float, default=1.35)
    parser.add_argument("--root-scale", type=float, default=1.0)
    parser.add_argument(
        "--mj-step",
        action="store_true",
        help="qpos 展示模式下启用 mj_step（Policy 模式始终走完整 mj_step）",
    )
    parser.add_argument("--blend-seconds", type=float, default=0.35)
    parser.add_argument(
        "--registry",
        action="store_true",
        help="打印 dataset registry JSON 后退出（自检）",
    )
    parser.add_argument(
        "--dry-run-video",
        type=Path,
        default=None,
        metavar="VIDEO",
        help="dry-run GVHMR→GMR 命令后退出",
    )
    parser.add_argument(
        "--dry-run-batch",
        choices=("amass", "lafan1", "bvh_generic", "smplx_generic"),
        default=None,
        help="dry-run mocap 批处理命令后退出",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.registry:
        import json

        from gvhmr_gmr.dataset_registry import build_registry

        print(json.dumps(build_registry(ROOT), ensure_ascii=False, indent=2))
        return

    if args.dry_run_video is not None:
        from gvhmr_gmr.pipeline_runner import run_video_route

        r = run_video_route(ROOT, args.dry_run_video, "user_upload", dry_run=True)
        print(r.message)
        for line in r.log or []:
            print(line)
        sys.exit(0 if r.ok else 1)

    if args.dry_run_batch is not None:
        from gvhmr_gmr.pipeline_runner import run_motion_batch

        r = run_motion_batch(ROOT, args.dry_run_batch, dry_run=True)
        print(r.message)
        sys.exit(0 if r.ok else 1)

    from pipeline.motion_amplifier import AmplifyConfig

    if args.mode == "policy" and not args.clip:
        parser.error("policy 模式需要 --clip，例如 --clip B_DadDance")

    config = AppConfig(
        csv_dir=args.csv_dir,
        policy_data_root=args.policy_data_root,
        model_path=args.model,
        policy_model_path=args.policy_model,
        host=args.host,
        port=args.port,
        render_fps=args.render_fps,
        clip_fps=args.clip_fps,
        preview_seconds=args.preview_seconds,
        default_speed=args.speed,
        amplify=AmplifyConfig(root_xyz_scale=args.root_scale, joint_scale=args.joint_scale),
        use_mj_step=args.mj_step,
        blend_seconds=args.blend_seconds,
        default_mode=args.mode,
        startup_clip=args.clip,
    )
    app = G1MovesWebGalleryFinal(config)
    app.run_forever()


if __name__ == "__main__":
    main()
