"""
Aura Walker: G1 camina y farmea aura al mismo tiempo.
Combina la política pre-entrenada walker.onnx (balance y locomoción)
con inyección de comandos de velocidad y override de brazos.
"""

import json
import time
import math
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

CHALLENGE_DIR = Path(__file__).resolve().parent / "g1-manipulation-challenge"

class ONNXPolicy:
    def __init__(self, path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def __call__(self, obs):
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        return self.sess.run([self.out_name], {self.in_name: obs.astype(np.float32)})[0][0]

def set_armature(model, joint_names):
    ARM_5020, ARM_7520_14, ARM_7520_22, ARM_4010, ARM_2x5020 = 0.00360972, 0.01017752, 0.02510192, 0.00425000, 0.00721945
    for i, name in enumerate(joint_names):
        dof = 6 + i
        if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
            model.dof_armature[dof] = ARM_5020
        elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            model.dof_armature[dof] = ARM_7520_14
        elif "hip_roll" in name or "knee" in name:
            model.dof_armature[dof] = ARM_7520_22
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            model.dof_armature[dof] = ARM_4010
        elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
            model.dof_armature[dof] = ARM_2x5020
        else:
            model.dof_armature[dof] = ARM_5020

def quat_apply_inverse(quat, vec):
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))

def ease(t):
    return 0.5 * (1 - math.cos(math.pi * max(0.0, min(1.0, t))))

def main():
    with open(CHALLENGE_DIR / "model_config.json") as f:
        config = json.load(f)
    joint_names = config["joint_names"]
    num_joints = len(joint_names)

    default_pos = np.zeros(num_joints, dtype=np.float32)
    for name, val in config["default_joint_pos"].items():
        if name in joint_names:
            default_pos[joint_names.index(name)] = val

    action_scales = np.array([config["action_scales"][n] for n in joint_names], dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(str(CHALLENGE_DIR / "scene.xml"))
    model.opt.timestep = 0.005
    set_armature(model, joint_names)
    data = mujoco.MjData(model)

    # Spawn position
    data.qpos[0] = -0.6
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1, 0, 0, 0]
    for name, val in config["default_joint_pos"].items():
        if name in joint_names:
            data.qpos[7 + joint_names.index(name)] = val
    mujoco.mj_forward(model, data)

    print("Cargando walker.onnx...")
    walker = ONNXPolicy(CHALLENGE_DIR / "walker.onnx")
    walker(np.zeros((1, 99), dtype=np.float32))

    arm_patterns = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    arm_indices = [i for i, n in enumerate(joint_names) if any(p in n for p in arm_patterns)]
    joint_idx = {n: i for i, n in enumerate(joint_names)}

    # ── Poses ──
    arms_stand = {n: default_pos[joint_idx[n]] for n in joint_names if any(p in n for p in arm_patterns)}
    arms_stand.update({"waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0})

    arms_dab = arms_stand.copy()
    arms_dab.update({
        "waist_yaw_joint": 0.3,
        "left_shoulder_pitch_joint": -1.8, "left_shoulder_roll_joint": 1.0,
        "right_shoulder_pitch_joint": -2.5, "right_shoulder_roll_joint": -0.3, "right_shoulder_yaw_joint": 1.0, "right_elbow_joint": 2.0,
    })

    arms_flex = arms_stand.copy()
    arms_flex.update({
        "left_shoulder_pitch_joint": -0.5, "left_shoulder_roll_joint": 1.3, "left_elbow_joint": 2.0,
        "right_shoulder_pitch_joint": -0.5, "right_shoulder_roll_joint": -1.3, "right_elbow_joint": 2.0,
    })

    arms_salute = arms_stand.copy()
    arms_salute.update({
        "right_shoulder_pitch_joint": -1.5, "right_shoulder_roll_joint": -0.2, "right_shoulder_yaw_joint": 0.5, 
        "right_elbow_joint": 2.2, "right_wrist_pitch_joint": -0.5,
    })

    arms_tpose = arms_stand.copy()
    arms_tpose.update({
        "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 1.5, "left_elbow_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": -1.5, "right_elbow_joint": 0.0,
    })

    # ── Secuencia: Comando + Poses ──
    # Comandos: [vx, vy, yaw_rate]
    cmd_fwd = [0.4, 0.0, 0.0]
    cmd_stop = [0.0, 0.0, 0.0]
    cmd_turn_L = [0.0, 0.0, 1.0]
    cmd_turn_R = [0.0, 0.0, -1.0]

    WARMUP_TIME = 3.0
    # (cmd, pose_from, pose_to, t_transicion, t_hold)
    sequence = [
        (cmd_fwd,    arms_stand,  arms_flex,   1.0, 2.5),  # Camina adelante 3.5s y muestra biceps
        (cmd_stop,   arms_flex,   arms_dab,    1.0, 2.0),  # Frena, tira un dab 3s
        (cmd_turn_L, arms_dab,    arms_salute, 1.0, 2.2),  # Gira en U (3.2s) saludando
        (cmd_fwd,    arms_salute, arms_tpose,  1.0, 2.5),  # Vuelve caminando 3.5s en T-pose
        (cmd_stop,   arms_tpose,  arms_stand,  1.0, 1.0),  # Frena
        (cmd_turn_R, arms_stand,  arms_stand,  0.5, 2.7),  # Gira en U a la derecha de vuelta
    ]

    phase_times = []
    t_accum = 0.0
    for (cmd, pf, pt, tt, th) in sequence:
        phase_times.append((t_accum, tt, th, pf, pt, cmd))
        t_accum += tt + th
    TOTAL_CYCLE = t_accum

    last_action = np.zeros(num_joints, dtype=np.float32)
    qpos_idx = {n: 7 + i for i, n in enumerate(joint_names)}
    qvel_idx = {n: 6 + i for i, n in enumerate(joint_names)}

    def get_walker_obs(current_cmd):
        pos = data.qpos[:3].copy()
        quat = data.qpos[3:7].copy()
        lin_vel_world = data.qvel[:3].copy()
        ang_vel_body = data.qvel[3:6].copy()

        lin_vel_body = quat_apply_inverse(quat, lin_vel_world)
        proj_grav = quat_apply_inverse(quat, np.array([0, 0, -1.0]))

        jpos = np.array([data.qpos[qpos_idx[n]] - default_pos[i] for i, n in enumerate(joint_names)], dtype=np.float32)
        jvel = np.array([data.qvel[qvel_idx[n]] for n in joint_names], dtype=np.float32)

        return np.concatenate([
            lin_vel_body, ang_vel_body, proj_grav,
            jpos, jvel, last_action, np.array(current_cmd, dtype=np.float32)
        ]).astype(np.float32)

    print("\n" + "="*45)
    print("  AURA WALKER: PATRULLAJE Y POSES")
    print("="*45)
    print("  Arrancando en 3...2...1...")

    decimation = 4
    control_step = 0
    target_pos = default_pos.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        sim_time = 0.0

        while viewer.is_running():
            wall = time.time() - t0
            if wall - sim_time > 0.05:
                sim_time = wall - 0.05

            while sim_time < wall:
                if control_step % decimation == 0:
                    current_cmd = [0.0, 0.0, 0.0]

                    if wall > WARMUP_TIME:
                        t = (wall - WARMUP_TIME) % TOTAL_CYCLE
                        for (t_start, t_trans, t_hold, p_from, p_to, cmd) in phase_times:
                            t_end = t_start + t_trans + t_hold
                            if t < t_end:
                                current_cmd = cmd
                                elapsed = t - t_start
                                if t_trans > 0 and elapsed < t_trans:
                                    progress = ease(elapsed / t_trans)
                                    for name in p_from:
                                        if name in joint_idx:
                                            target_pos[joint_idx[name]] = lerp(p_from[name], p_to[name], progress)
                                else:
                                    for name, val in p_to.items():
                                        if name in joint_idx:
                                            target_pos[joint_idx[name]] = val
                                break
                    else:
                        for i in arm_indices:
                            target_pos[i] = default_pos[i]

                    # 1. Walker: balance
                    obs = get_walker_obs(current_cmd)
                    action = walker(obs)
                    
                    # Merge targets
                    walker_targets = default_pos + action * action_scales
                    
                    # Piernas usan el walker, brazos usan nuestro target (o default)
                    for i in range(num_joints):
                        if i not in arm_indices and joint_names[i] not in ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]:
                            target_pos[i] = walker_targets[i]

                    last_action[:] = action

                for i, name in enumerate(joint_names):
                    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                    if act_id >= 0:
                        data.ctrl[act_id] = target_pos[i]

                mujoco.mj_step(model, data)
                control_step += 1
                sim_time += model.opt.timestep

            viewer.sync()

if __name__ == "__main__":
    main()
