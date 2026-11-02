"""
DAB con gravedad y equilibrio perfecto.
Reutiliza el walker.onnx del repo g1-manipulation-challenge para el balance
y sobreescribe los brazos con la pose del DAB.
"""

import json
import time
import math
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# Ruta al repo con los modelos pre-entrenados
CHALLENGE_DIR = Path(__file__).resolve().parent / "g1-manipulation-challenge"


# ── ONNX Policy (copiado de run.py) ──
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


# ── Armature setup (copiado de run.py) ──
def set_armature(model, joint_names):
    ARM_5020 = 0.00360972
    ARM_7520_14 = 0.01017752
    ARM_7520_22 = 0.02510192
    ARM_4010 = 0.00425000
    ARM_2x5020 = 0.00721945
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


# ── Helpers de fisica (copiados de run.py) ──
def quat_apply_inverse(quat, vec):
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def ease(t):
    return 0.5 * (1 - math.cos(math.pi * max(0.0, min(1.0, t))))


def main():
    # ── Cargar config ──
    with open(CHALLENGE_DIR / "model_config.json") as f:
        config = json.load(f)
    joint_names = config["joint_names"]
    num_joints = len(joint_names)

    # Posiciones default de cada joint
    default_pos = np.zeros(num_joints, dtype=np.float32)
    for name, val in config["default_joint_pos"].items():
        if name in joint_names:
            default_pos[joint_names.index(name)] = val

    # Action scales del walker
    action_scales = np.array([config["action_scales"][n] for n in joint_names], dtype=np.float32)

    # ── Cargar modelo MuJoCo (usamos la escena del challenge) ──
    model = mujoco.MjModel.from_xml_path(str(CHALLENGE_DIR / "scene.xml"))
    model.opt.timestep = 0.005  # 200 Hz (debe coincidir con el entrenamiento)
    set_armature(model, joint_names)
    data = mujoco.MjData(model)

    # Pose inicial
    data.qpos[0] = -0.6  # Atras de la mesa, igual que run.py
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1, 0, 0, 0]
    for name, val in config["default_joint_pos"].items():
        if name in joint_names:
            data.qpos[7 + joint_names.index(name)] = val
    mujoco.mj_forward(model, data)

    # ── Cargar walker policy ──
    print("Cargando walker.onnx...")
    walker = ONNXPolicy(CHALLENGE_DIR / "walker.onnx")
    walker(np.zeros((1, 99), dtype=np.float32))  # Warm up
    print("Walker cargado y listo.")

    # ── Indices de joints de brazos ──
    arm_patterns = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    arm_indices = [i for i, n in enumerate(joint_names) if any(p in n for p in arm_patterns)]

    # Indices especificos para el DAB
    joint_idx = {n: i for i, n in enumerate(joint_names)}

    # ── Poses de brazos ──
    # STAND: brazos relajados (default del config)
    arms_stand = {n: default_pos[joint_idx[n]] for n in joint_names if any(p in n for p in arm_patterns)}
    # Tambien incluimos el waist
    arms_stand["waist_yaw_joint"] = 0.0
    arms_stand["waist_roll_joint"] = 0.0
    arms_stand["waist_pitch_joint"] = 0.0

    # DAB: brazo izquierdo extendido diagonal, cara en el codo derecho
    arms_dab = {
        # Torso girado ligeramente
        "waist_yaw_joint": 0.3,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        # Brazo izquierdo: extendido diagonal arriba
        "left_shoulder_pitch_joint": -1.8,
        "left_shoulder_roll_joint": 1.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        # Brazo derecho: codo arriba, tapando la cara
        "right_shoulder_pitch_joint": -2.5,
        "right_shoulder_roll_joint": -0.3,
        "right_shoulder_yaw_joint": 1.0,
        "right_elbow_joint": 2.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    # ── Variables del walker ──
    last_action = np.zeros(num_joints, dtype=np.float32)
    qpos_idx = {n: 7 + i for i, n in enumerate(joint_names)}
    qvel_idx = {n: 6 + i for i, n in enumerate(joint_names)}

    def get_walker_obs():
        """Construye la observacion de 99 dims para el walker (igual que run.py)."""
        pos = data.qpos[:3].copy()
        quat = data.qpos[3:7].copy()
        lin_vel_world = data.qvel[:3].copy()
        ang_vel_body = data.qvel[3:6].copy()

        lin_vel_body = quat_apply_inverse(quat, lin_vel_world)
        proj_grav = quat_apply_inverse(quat, np.array([0, 0, -1.0]))

        jpos = np.array([data.qpos[qpos_idx[n]] - default_pos[i]
                         for i, n in enumerate(joint_names)], dtype=np.float32)
        jvel = np.array([data.qvel[qvel_idx[n]]
                         for n in joint_names], dtype=np.float32)

        # Comando: velocidad cero (parado quieto)
        cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return np.concatenate([
            lin_vel_body, ang_vel_body, proj_grav,
            jpos, jvel, last_action, cmd
        ]).astype(np.float32)

    # ── Secuencia de la coreografia ──
    # (pose_from, pose_to, t_transicion, t_hold)
    WARMUP_TIME = 3.0  # Segundos para que el walker se estabilice antes de mover brazos
    sequence = [
        (arms_stand, arms_stand, 0.0, 2.0),    # Pararse tranquilo 2s
        (arms_stand, arms_dab,   1.5, 3.0),    # Transicion suave al DAB, mantener 3s
        (arms_dab,   arms_stand, 1.5, 2.0),    # Volver a stand
    ]
    phase_times = []
    t_accum = 0.0
    for (pf, pt, tt, th) in sequence:
        phase_times.append((t_accum, tt, th, pf, pt))
        t_accum += tt + th
    TOTAL_CYCLE = t_accum

    print(f"\n{'='*45}")
    print("  DAB CON GRAVEDAD Y BALANCE NEURONAL")
    print(f"{'='*45}")
    print(f"  Balance: walker.onnx (RL pre-entrenado)")
    print(f"  Ciclo:   {TOTAL_CYCLE:.1f}s (loop infinito)")
    print(f"  Cerra la ventana para salir.")
    print(f"{'='*45}\n")

    # ── Loop de simulacion ──
    decimation = 4
    control_step = 0
    target_pos = default_pos.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        sim_time = 0.0

        while viewer.is_running():
            wall = time.time() - t0
            # Cap de catchup para evitar lag acumulado (copiado de run.py)
            max_catchup = 0.05
            if wall - sim_time > max_catchup:
                sim_time = wall - max_catchup

            while sim_time < wall:
                if control_step % decimation == 0:
                    # 1. Walker: calcula targets para todo el cuerpo (balance)
                    obs = get_walker_obs()
                    action = walker(obs)
                    target_pos = default_pos + action * action_scales

                    # 2. Sobreescribir brazos SOLO despues del warmup
                    if wall > WARMUP_TIME:
                        t = (wall - WARMUP_TIME) % TOTAL_CYCLE
                        for (t_start, t_trans, t_hold, p_from, p_to) in phase_times:
                            t_end = t_start + t_trans + t_hold
                            if t < t_end:
                                elapsed = t - t_start
                                if t_trans > 0 and elapsed < t_trans:
                                    progress = ease(elapsed / t_trans)
                                    for name in p_from:
                                        if name in joint_idx:
                                            val = lerp(p_from[name], p_to[name], progress)
                                            target_pos[joint_idx[name]] = val
                                else:
                                    for name, val in p_to.items():
                                        if name in joint_idx:
                                            target_pos[joint_idx[name]] = val
                                break
                    else:
                        # Durante warmup: brazos en default (walker decide todo)
                        for i in arm_indices:
                            target_pos[i] = default_pos[i]

                    last_action[:] = action

                # 3. Aplicar targets a los actuadores
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
