from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from pipeline.action_controller import ActionController
from pipeline.action_adapter import DEFAULT_JOINT_POS
from pipeline.pd_controller import apply_pd_targets
from pipeline.policy_discovery import PolicyBundle

# NPZ body_names order (mjlab tracking); index 0 = pelvis anchor per run_policy.py.
ANCHOR_BODY_INDEX = 0


@dataclass
class PolicyIoInfo:
    input_name: str
    input_shape: tuple
    output_name: str
    output_shape: tuple


def inspect_onnx(onnx_path: str | Path) -> PolicyIoInfo:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    return PolicyIoInfo(
        input_name=inp.name,
        input_shape=tuple(inp.shape),
        output_name=out.name,
        output_shape=tuple(out.shape),
    )


def _quat_to_rot_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_6d(rot_matrix: np.ndarray) -> np.ndarray:
    return rot_matrix[:, :2].T.flatten()


def transform_to_body_frame(
    pos_world: np.ndarray,
    quat_world_wxyz: np.ndarray,
    anchor_pos_world: np.ndarray,
    anchor_quat_world_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r_robot = _quat_to_rot_matrix(quat_world_wxyz)
    anchor_pos_b = r_robot.T @ (anchor_pos_world - pos_world)
    r_anchor = _quat_to_rot_matrix(anchor_quat_world_wxyz)
    r_rel = r_robot.T @ r_anchor
    anchor_ori_b = _rotation_matrix_to_6d(r_rel)
    return anchor_pos_b.astype(np.float32), anchor_ori_b.astype(np.float32)


def build_tracking_obs(
    *,
    ref_joint_pos: np.ndarray,
    ref_joint_vel: np.ndarray,
    anchor_pos_b: np.ndarray,
    anchor_ori_b: np.ndarray,
    base_ang_vel: np.ndarray,
    base_lin_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """160-dim observation — order matches g1-moves run_policy.py / dataset README."""
    return np.concatenate(
        [
            ref_joint_pos.astype(np.float32),
            ref_joint_vel.astype(np.float32),
            anchor_pos_b.astype(np.float32),
            anchor_ori_b.astype(np.float32),
            base_ang_vel.astype(np.float32),
            base_lin_vel.astype(np.float32),
            joint_pos.astype(np.float32) - DEFAULT_JOINT_POS,
            joint_vel.astype(np.float32),
            last_action.astype(np.float32),
        ]
    ).astype(np.float32)


def read_policy_imu(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """
    Read IMU exactly as g1-moves run_policy.py:
    sensordata[0:3] = imu_ang_vel, sensordata[3:6] = imu_lin_vel.
    Requires g1_policy_mode15_web.xml (policy sensors listed first).
    """
    if data.sensordata.size >= 6:
        return (
            data.sensordata[0:3].astype(np.float32),
            data.sensordata[3:6].astype(np.float32),
        )
    return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)


def validate_policy_xml(model: mujoco.MjModel) -> None:
    if model.nsensor < 2:
        raise RuntimeError("Policy XML 缺少 IMU 传感器，请使用 g1_policy_mode15_web.xml")
    n0 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, 0) or ""
    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, 1) or ""
    if n0 != "imu_ang_vel" or n1 != "imu_lin_vel":
        raise RuntimeError(
            f"Policy XML 传感器顺序错误: 期望 imu_ang_vel, imu_lin_vel 在前，"
            f"实际为 {n0}, {n1}。请使用 assets/proto_g1/g1_policy_mode15_web.xml"
        )


class G1MovesPolicyRunner:
    """
    G1 Moves Policy Stability Fix v2:
    obs(160) -> ONNX (baked-in norm) -> ActionController -> PD -> mj_step.
    """

    SIM_HZ = 1000.0
    POLICY_HZ = 20.0
    SIM_DT = 1.0 / SIM_HZ
    CONTROL_DT = 1.0 / POLICY_HZ

    def __init__(self, bundle: PolicyBundle, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        validate_policy_xml(model)
        self.bundle = bundle
        self.model = model
        self.data = data
        self._io = inspect_onnx(bundle.onnx_path)
        self._session = ort.InferenceSession(
            str(bundle.onnx_path), providers=["CPUExecutionProvider"]
        )
        self._action_ctrl = ActionController()

        motion = np.load(bundle.npz_path)
        self.ref_joint_pos = motion["joint_pos"].astype(np.float32)
        self.ref_joint_vel = motion["joint_vel"].astype(np.float32)
        self.ref_body_pos = motion["body_pos_w"].astype(np.float64)
        self.ref_body_quat = motion["body_quat_w"].astype(np.float64)
        self.fps = float(motion["fps"])
        self.num_frames = int(self.ref_joint_pos.shape[0])

        self.last_action = np.zeros(29, dtype=np.float32)
        self.motion_time = 0.0
        self.speed = 1.0
        self.playing = False

        self.model.opt.timestep = self.SIM_DT
        self.model.opt.iterations = 6
        self.model.opt.solver = 1  # mjSOL_CG
        self._decimation = max(1, int(round(self.SIM_HZ / self.POLICY_HZ)))
        self._use_position_actuators = self._detect_position_actuators(model)

    @property
    def DECIMATION(self) -> int:
        return self._decimation

    @staticmethod
    def _detect_position_actuators(model: mujoco.MjModel) -> bool:
        if model.nu < 1:
            return False
        return float(model.actuator_gainprm[0, 0]) > 0.0 and float(model.actuator_biasprm[0, 1]) != 0.0

    def reset(self, speed: float = 1.0) -> None:
        self.speed = max(0.1, float(speed))
        self.motion_time = 0.0
        self.last_action = np.zeros(29, dtype=np.float32)
        self._action_ctrl.reset()
        self.playing = True
        self.data.qpos[:3] = self.ref_body_pos[0, ANCHOR_BODY_INDEX]
        self.data.qpos[3:7] = self.ref_body_quat[0, ANCHOR_BODY_INDEX]
        self.data.qpos[7:36] = self.ref_joint_pos[0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def stop(self) -> None:
        self.playing = False

    @property
    def frame_index(self) -> int:
        return int(self.motion_time * self.fps) % self.num_frames

    def control_step(self) -> None:
        frame = self.frame_index
        ref_jp = self.ref_joint_pos[frame]
        ref_jv = self.ref_joint_vel[frame]

        robot_pos = self.data.qpos[:3].astype(np.float64)
        robot_quat = self.data.qpos[3:7].astype(np.float64)
        joint_pos = self.data.qpos[7:36].astype(np.float32)
        joint_vel = self.data.qvel[6:35].astype(np.float32)

        anchor_pos_w = self.ref_body_pos[frame, ANCHOR_BODY_INDEX]
        anchor_quat_w = self.ref_body_quat[frame, ANCHOR_BODY_INDEX]
        anchor_pos_b, anchor_ori_b = transform_to_body_frame(
            robot_pos, robot_quat, anchor_pos_w, anchor_quat_w
        )

        base_ang_vel, base_lin_vel = read_policy_imu(self.data)

        obs = build_tracking_obs(
            ref_joint_pos=ref_jp,
            ref_joint_vel=ref_jv,
            anchor_pos_b=anchor_pos_b,
            anchor_ori_b=anchor_ori_b,
            base_ang_vel=base_ang_vel,
            base_lin_vel=base_lin_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_action=self.last_action,
        )
        if obs.shape[0] != 160:
            raise RuntimeError(f"obs 维度错误: {obs.shape[0]}")

        # ONNX export includes baked-in observation normalization — feed raw obs.
        raw_actions = self._session.run(
            [self._io.output_name],
            {self._io.input_name: obs[None]},
        )[0][0]

        processed = self._action_ctrl.process(raw_actions)
        self.last_action = processed.copy()

        target_pos = self._action_ctrl.to_joint_targets(
            processed,
            ref_joint_pos=ref_jp,
            use_mjlab_scale=True,
            blend_with_ref=True,
        )
        apply_pd_targets(
            self.data,
            target_pos,
            joint_pos,
            joint_vel,
            use_position_actuators=self._use_position_actuators,
            model=self.model,
            use_official_gains=False,
        )
        self.motion_time += self.CONTROL_DT * self.speed

    def simulate_step(self) -> None:
        self.control_step()
        for _ in range(self._decimation):
            mujoco.mj_step(self.model, self.data)
