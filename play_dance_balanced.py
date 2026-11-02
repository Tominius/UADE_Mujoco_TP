"""
Reproductor de Bailes (Equilibrio Dinamico)
Usa walker.onnx para que las piernas mantengan el balance perfecto
mientras el torso y los brazos reproducen las animaciones CSV complejas.
"""

import sys
import time
import json
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# Rutas
WORKSPACE = Path(__file__).resolve().parent
VISER_REPO = WORKSPACE / "Unitree-G1MuJoCo-Viser"
CHALLENGE_DIR = WORKSPACE / "g1-manipulation-challenge"
CSV_DIR = VISER_REPO / "g1_moves" / "g1_moves_outputs" / "web_csv"
MODEL_PATH = CHALLENGE_DIR / "scene.xml"

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

def list_dances():
    if not CSV_DIR.exists():
        return []
    return [f.stem for f in CSV_DIR.glob("*.csv")]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dance_name", nargs="?", default="dance_B_DadDance_web")
    args = parser.parse_args()

    # Buscar CSV
    csv_path = CSV_DIR / f"{args.dance_name}.csv"
    if not csv_path.exists():
        matches = [f for f in list_dances() if args.dance_name.lower() in f.lower()]
        if len(matches) == 1:
            csv_path = CSV_DIR / f"{matches[0]}.csv"
        else:
            print("Baile no encontrado.")
            return

    print(f"Cargando animacion: {csv_path.name}")
    trajectory = np.loadtxt(csv_path, delimiter=",")
    num_frames = trajectory.shape[0]
    fps = 60.0
    duration = num_frames / fps

    # Cargar Config de control
    with open(CHALLENGE_DIR / "model_config.json") as f:
        config = json.load(f)
    joint_names = config["joint_names"]
    num_joints = len(joint_names)
    default_pos = np.zeros(num_joints, dtype=np.float32)
    for name, val in config["default_joint_pos"].items():
        if name in joint_names:
            default_pos[joint_names.index(name)] = val
    action_scales = np.array([config["action_scales"][n] for n in joint_names], dtype=np.float32)

    # Identificar que joints son del tren superior (brazos y cintura)
    # Dejamos las piernas para que el RL (walker) haga su magia
    upper_body_patterns = ["shoulder", "elbow", "wrist", "waist"]
    upper_body_indices = [i for i, n in enumerate(joint_names) if any(p in n for p in upper_body_patterns)]

    # Mapeo de joints del CSV (estan en orden de MuJoCo) al config
    csv_joint_idx = {name: 7 + i for i, name in enumerate(joint_names)}

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = 0.005
    set_armature(model, joint_names)
    data = mujoco.MjData(model)

    data.qpos[0] = -0.6
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1, 0, 0, 0]
    for i in range(num_joints):
        data.qpos[7+i] = default_pos[i]
    mujoco.mj_forward(model, data)

    print("Cargando walker.onnx para el balance...")
    walker = ONNXPolicy(CHALLENGE_DIR / "walker.onnx")
    walker(np.zeros((1, 99), dtype=np.float32))

    last_action = np.zeros(num_joints, dtype=np.float32)
    qpos_idx = {n: 7 + i for i, n in enumerate(joint_names)}
    qvel_idx = {n: 6 + i for i, n in enumerate(joint_names)}

    def get_walker_obs():
        pos = data.qpos[:3].copy()
        quat = data.qpos[3:7].copy()
        lin_vel_world = data.qvel[:3].copy()
        ang_vel_body = data.qvel[3:6].copy()
        lin_vel_body = quat_apply_inverse(quat, lin_vel_world)
        proj_grav = quat_apply_inverse(quat, np.array([0, 0, -1.0]))
        jpos = np.array([data.qpos[qpos_idx[n]] - default_pos[i] for i, n in enumerate(joint_names)], dtype=np.float32)
        jvel = np.array([data.qvel[qvel_idx[n]] for n in joint_names], dtype=np.float32)
        cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([lin_vel_body, ang_vel_body, proj_grav, jpos, jvel, last_action, cmd]).astype(np.float32)

    decimation = 4
    control_step = 0
    target_pos = default_pos.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        sim_time = 0.0

        while viewer.is_running():
            wall = time.time() - t0
            if wall - sim_time > 0.05: sim_time = wall - 0.05

            while sim_time < wall:
                if control_step % decimation == 0:
                    # 1. RL Policy calcula todo para balancear
                    obs = get_walker_obs()
                    action = walker(obs)
                    walker_targets = default_pos + action * action_scales
                    
                    # 2. Leer frame del CSV
                    current_time = wall % duration
                    frame_idx = min(int(current_time * fps), num_frames - 1)
                    csv_frame = trajectory[frame_idx]

                    # 3. Fusionar: Piernas usan RL, Torso/Brazos usan el baile
                    for i in range(num_joints):
                        if i in upper_body_indices:
                            target_pos[i] = csv_frame[csv_joint_idx[joint_names[i]]]
                        else:
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
