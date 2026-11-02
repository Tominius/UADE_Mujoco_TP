"""
G1 Humanoide - Demostración de Push Recovery (Rechazo de Perturbaciones).
El robot mantiene el equilibrio con gravedad real usando Deep Reinforcement Learning (walker.onnx).
Aplica fuerzas externas (empujones) periódicos o por teclado, y el robot absorbe el impacto
reacomodando sus articulaciones sin caerse.
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

CHALLENGE_DIR = Path(__file__).resolve().parent / "models" / "unitree_g1"
if not CHALLENGE_DIR.exists():
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

def quat_apply_inverse(quat, vec):
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

def main():
    # Cargar configuracion
    with open(CHALLENGE_DIR / "model_config.json") as f:
        config = json.load(f)
    joint_names = config["joint_names"]
    num_joints = len(joint_names)

    default_pos = np.zeros(num_joints, dtype=np.float32)
    for name, val in config["default_joint_pos"].items():
        if name in joint_names:
            default_pos[joint_names.index(name)] = val

    action_scales = np.array([config["action_scales"][n] for n in joint_names], dtype=np.float32)

    # Cargar modelo
    model = mujoco.MjModel.from_xml_path(str(CHALLENGE_DIR / "scene.xml"))
    model.opt.timestep = 0.005
    set_armature(model, joint_names)
    data = mujoco.MjData(model)

    # Spawn detras de la mesa
    data.qpos[0] = -0.6
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1, 0, 0, 0]
    for name, val in config["default_joint_pos"].items():
        if name in joint_names:
            data.qpos[7 + joint_names.index(name)] = val
    mujoco.mj_forward(model, data)

    # ID del torso donde se aplican las fuerzas
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link_rev_1_0")
    if torso_id < 0:
        torso_id = 1  # Pelvis fallback

    # Cargar politica ONNX
    print("Cargando red neuronal de equilibrio (walker.onnx)...")
    walker = ONNXPolicy(CHALLENGE_DIR / "walker.onnx")
    walker(np.zeros((1, 99), dtype=np.float32))

    last_action = np.zeros(num_joints, dtype=np.float32)
    qpos_idx = {n: 7 + i for i, n in enumerate(joint_names)}
    qvel_idx = {n: 6 + i for i, n in enumerate(joint_names)}

    def get_walker_obs():
        quat = data.qpos[3:7].copy()
        lin_vel_world = data.qvel[:3].copy()
        ang_vel_body = data.qvel[3:6].copy()
        lin_vel_body = quat_apply_inverse(quat, lin_vel_world)
        proj_grav = quat_apply_inverse(quat, np.array([0, 0, -1.0]))
        jpos = np.array([data.qpos[qpos_idx[n]] - default_pos[i] for i, n in enumerate(joint_names)], dtype=np.float32)
        jvel = np.array([data.qvel[qvel_idx[n]] for n in joint_names], dtype=np.float32)
        cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([lin_vel_body, ang_vel_body, proj_grav, jpos, jvel, last_action, cmd]).astype(np.float32)

    # Variables de empuje manual e interactivo
    manual_force = np.zeros(6, dtype=np.float64)
    force_until_time = 0.0
    active_push_label = ""

    def trigger_push(label, fx, fy):
        nonlocal force_until_time, active_push_label
        manual_force[:3] = [fx, fy, 0.0]
        force_until_time = data.time + 0.18
        active_push_label = label
        print(f"\n>>> [{data.time:.2f}s] {label} (Fx={fx:+.0f}N, Fy={fy:+.0f}N) -> Observa la reaccion!")

    def key_callback(keycode):
        # GLFW keycodes
        if keycode in (87, 265):    # W o Arriba
            trigger_push("¡EMPUJON MANUAL FRONTAL!", 65.0, 0.0)
        elif keycode in (83, 264):  # S o Abajo
            trigger_push("¡EMPUJON MANUAL TRASERO!", -65.0, 0.0)
        elif keycode in (65, 263):  # A o Izquierda
            trigger_push("¡EMPUJON MANUAL HACIA LA IZQUIERDA!", 0.0, 50.0)
        elif keycode in (68, 262):  # D o Derecha
            trigger_push("¡EMPUJON MANUAL HACIA LA DERECHA!", 0.0, -50.0)
        elif keycode == 32:         # Espacio -> Empujon aleatorio
            angle = np.random.uniform(0, 2 * math.pi)
            fx = 60.0 * math.cos(angle)
            fy = 45.0 * math.sin(angle)
            trigger_push("¡EMPUJON ALEATORIO!", fx, fy)

    print("\n" + "=" * 65)
    print("   HUMANOIDE G1 - DEMOSTRACION DE RECUPERACION DE EMPUJONES")
    print("                 (Push Recovery & Disturbance Rejection)")
    print("=" * 65)
    print("  * Estabilizacion por Deep RL (walker.onnx a 200 Hz)")
    print("  * El sistema aplica empujones ciclicos automaticos cada 4 segundos:")
    print("      1) Empujon Frontal (+60 N)")
    print("      2) Empujon Trasero (-60 N)")
    print("      3) Empujon Lateral Izquierdo (+45 N)")
    print("      4) Empujon Lateral Derecho (-45 N)")
    print()
    print("  [CONTROLES EN VIVO]")
    print("  * Teclado: W/S/A/D o Flechas para empujar en cualquier direccion")
    print("  * Barra espaciadora: Empujon en direccion aleatoria")
    print("  * Mouse: Manten presionado Ctrl + Click Derecho y arrastra")
    print("           sobre el robot para jalarlo con el cursor.")
    print("=" * 65 + "\n")

    # Ciclo automatico de empujones
    # Cada 4 segundos aplica un empujon
    auto_pushes = [
        (4.0, "¡EMPUJON AUTOMATICO FRONTAL!", 60.0, 0.0),
        (8.0, "¡EMPUJON AUTOMATICO TRASERO!", -60.0, 0.0),
        (12.0, "¡EMPUJON AUTOMATICO LATERAL IZQUIERDO!", 0.0, 45.0),
        (16.0, "¡EMPUJON AUTOMATICO LATERAL DERECHO!", 0.0, -45.0),
    ]
    CYCLE_DURATION = 16.0

    decimation = 4
    control_step = 0
    target_pos = default_pos.copy()
    last_auto_cycle_idx = -1

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        t0 = time.time()
        sim_time = 0.0

        while viewer.is_running():
            wall = time.time() - t0
            if wall - sim_time > 0.05:
                sim_time = wall - 0.05

            while sim_time < wall:
                current_sim_t = data.time

                # 1. Empujones automaticos periodicos (despues de 3s de warmup inicial)
                if current_sim_t > 3.0:
                    phase_t = (current_sim_t - 3.0) % CYCLE_DURATION
                    # Verificar si coincide con algun evento del ciclo
                    for push_t, label, fx, fy in auto_pushes:
                        if abs(phase_t - (push_t - 4.0)) < 0.005:
                            cycle_idx = int(current_sim_t / 4.0)
                            if cycle_idx != last_auto_cycle_idx:
                                last_auto_cycle_idx = cycle_idx
                                trigger_push(label, fx, fy)

                # 2. Aplicar fuerza en el torso si hay un empuje activo
                if current_sim_t < force_until_time:
                    data.xfrc_applied[torso_id] = manual_force
                else:
                    data.xfrc_applied[torso_id] = np.zeros(6)

                # 3. Control del robot por IA
                if control_step % decimation == 0:
                    obs = get_walker_obs()
                    action = walker(obs)
                    target_pos = default_pos + action * action_scales
                    last_action[:] = action

                # 4. Enviar objetivos a los actuadores
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
