"""
AURA FARMER v2 - Con gravedad y controlador de balance
El robot farmee aura SIN flotar. Usa un PD controller en las piernas
que lee la inclinacion del pelvis y corrige tobillos/caderas para no caerse.
"""

import mujoco
import mujoco.viewer
import time
import math
import numpy as np


def get_idx(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return i if i != -1 else None


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def ease(t):
    return 0.5 * (1 - math.cos(math.pi * max(0.0, min(1.0, t))))


def quat_to_pitch_roll(quat):
    """Extrae pitch (inclinacion adelante/atras) y roll (inclinacion lateral) de un cuaternion."""
    w, x, y, z = quat
    # Pitch = rotacion alrededor del eje Y (adelante/atras)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # Roll = rotacion alrededor del eje X (izquierda/derecha)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    return pitch, roll


def main():
    model = mujoco.MjModel.from_xml_path('mujoco_menagerie/unitree_g1/scene.xml')
    data = mujoco.MjData(model)

    # Todos los actuadores
    names = [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint', 'left_elbow_joint',
        'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint', 'right_elbow_joint',
        'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
    ]
    idx = {n: get_idx(model, n) for n in names}

    # Freejoint
    root_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
    qpos_adr = model.jnt_qposadr[root_jnt]
    dof_adr = model.jnt_dofadr[root_jnt]

    # ====================================================================
    #  GANANCIAS DEL CONTROLADOR DE BALANCE (PD)
    # ====================================================================
    # Pitch (adelante/atras): corregir con cadera y tobillo
    KP_PITCH = 8.0    # Ganancia proporcional
    KD_PITCH = 2.0    # Ganancia derivativa (amortiguamiento)
    # Roll (lateral): corregir con cadera y tobillo laterales
    KP_ROLL = 6.0
    KD_ROLL = 1.5

    # Pose base de las piernas (ligeramente flexionadas para tener rango de correccion)
    LEGS_BASE = {
        'left_hip_pitch_joint': -0.15, 'left_hip_roll_joint': 0.0, 'left_hip_yaw_joint': 0,
        'left_knee_joint': 0.3, 'left_ankle_pitch_joint': -0.15, 'left_ankle_roll_joint': 0,
        'right_hip_pitch_joint': -0.15, 'right_hip_roll_joint': 0.0, 'right_hip_yaw_joint': 0,
        'right_knee_joint': 0.3, 'right_ankle_pitch_joint': -0.15, 'right_ankle_roll_joint': 0,
    }

    # ====================================================================
    #  POSES DE AURA (solo torso y brazos, las piernas las maneja el balance)
    # ====================================================================
    WAIST_N = {'waist_yaw_joint': 0, 'waist_roll_joint': 0, 'waist_pitch_joint': 0}

    ARMS_STAND = {
        **WAIST_N,
        'left_shoulder_pitch_joint': 0.2, 'left_shoulder_roll_joint': 0.2,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 1.28,
        'left_wrist_roll_joint': 0, 'left_wrist_pitch_joint': 0, 'left_wrist_yaw_joint': 0,
        'right_shoulder_pitch_joint': 0.2, 'right_shoulder_roll_joint': -0.2,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 1.28,
        'right_wrist_roll_joint': 0, 'right_wrist_pitch_joint': 0, 'right_wrist_yaw_joint': 0,
    }

    ARMS_TPOSE = {
        **WAIST_N,
        'left_shoulder_pitch_joint': 0.0, 'left_shoulder_roll_joint': 1.5,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 0.0,
        'left_wrist_roll_joint': 0, 'left_wrist_pitch_joint': 0, 'left_wrist_yaw_joint': 0,
        'right_shoulder_pitch_joint': 0.0, 'right_shoulder_roll_joint': -1.5,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 0.0,
        'right_wrist_roll_joint': 0, 'right_wrist_pitch_joint': 0, 'right_wrist_yaw_joint': 0,
    }

    ARMS_FLEX = {
        **WAIST_N,
        'left_shoulder_pitch_joint': -0.3, 'left_shoulder_roll_joint': 1.4,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 2.0,
        'left_wrist_roll_joint': 0, 'left_wrist_pitch_joint': 0, 'left_wrist_yaw_joint': 0,
        'right_shoulder_pitch_joint': -0.3, 'right_shoulder_roll_joint': -1.4,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 2.0,
        'right_wrist_roll_joint': 0, 'right_wrist_pitch_joint': 0, 'right_wrist_yaw_joint': 0,
    }

    ARMS_JOJO = {
        'waist_yaw_joint': -0.4, 'waist_roll_joint': 0.15, 'waist_pitch_joint': 0.05,
        'left_shoulder_pitch_joint': -1.5, 'left_shoulder_roll_joint': 0.3,
        'left_shoulder_yaw_joint': 0.5, 'left_elbow_joint': 0.3,
        'left_wrist_roll_joint': 0, 'left_wrist_pitch_joint': 0, 'left_wrist_yaw_joint': 0,
        'right_shoulder_pitch_joint': -0.8, 'right_shoulder_roll_joint': 0.5,
        'right_shoulder_yaw_joint': 1.5, 'right_elbow_joint': 2.0,
        'right_wrist_roll_joint': 0, 'right_wrist_pitch_joint': 0, 'right_wrist_yaw_joint': 0,
    }

    ARMS_DAB = {
        'waist_yaw_joint': 0.3, 'waist_roll_joint': 0, 'waist_pitch_joint': 0,
        'left_shoulder_pitch_joint': -1.8, 'left_shoulder_roll_joint': 1.0,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 0.0,
        'left_wrist_roll_joint': 0, 'left_wrist_pitch_joint': 0, 'left_wrist_yaw_joint': 0,
        'right_shoulder_pitch_joint': -2.5, 'right_shoulder_roll_joint': -0.3,
        'right_shoulder_yaw_joint': 1.0, 'right_elbow_joint': 2.0,
        'right_wrist_roll_joint': 0, 'right_wrist_pitch_joint': 0, 'right_wrist_yaw_joint': 0,
    }

    ARMS_LEAN = {
        'waist_yaw_joint': 0, 'waist_roll_joint': 0, 'waist_pitch_joint': -0.3,
        'left_shoulder_pitch_joint': 0.5, 'left_shoulder_roll_joint': 1.5,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 0.8,
        'left_wrist_roll_joint': 0, 'left_wrist_pitch_joint': 0, 'left_wrist_yaw_joint': 0,
        'right_shoulder_pitch_joint': 0.5, 'right_shoulder_roll_joint': -1.5,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 0.8,
        'right_wrist_roll_joint': 0, 'right_wrist_pitch_joint': 0, 'right_wrist_yaw_joint': 0,
    }

    ARMS_SALUTE = {
        'waist_yaw_joint': 0, 'waist_roll_joint': 0, 'waist_pitch_joint': 0.05,
        'left_shoulder_pitch_joint': 0.3, 'left_shoulder_roll_joint': 0.1,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 0.2,
        'left_wrist_roll_joint': 0, 'left_wrist_pitch_joint': 0, 'left_wrist_yaw_joint': 0,
        'right_shoulder_pitch_joint': -2.2, 'right_shoulder_roll_joint': -0.6,
        'right_shoulder_yaw_joint': 1.2, 'right_elbow_joint': 2.0,
        'right_wrist_roll_joint': 0, 'right_wrist_pitch_joint': 0, 'right_wrist_yaw_joint': 0,
    }

    ARMS_VICTORY = {
        **WAIST_N,
        'left_shoulder_pitch_joint': -2.8, 'left_shoulder_roll_joint': 0.8,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 0.1,
        'left_wrist_roll_joint': 0, 'left_wrist_pitch_joint': 0, 'left_wrist_yaw_joint': 0,
        'right_shoulder_pitch_joint': -2.8, 'right_shoulder_roll_joint': -0.8,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 0.1,
        'right_wrist_roll_joint': 0, 'right_wrist_pitch_joint': 0, 'right_wrist_yaw_joint': 0,
    }

    ARMS_VILLAIN = {
        'waist_yaw_joint': 0, 'waist_roll_joint': 0, 'waist_pitch_joint': -0.2,
        'left_shoulder_pitch_joint': -0.5, 'left_shoulder_roll_joint': -0.3,
        'left_shoulder_yaw_joint': 1.5, 'left_elbow_joint': 2.0,
        'left_wrist_roll_joint': 0, 'left_wrist_pitch_joint': 0, 'left_wrist_yaw_joint': 0,
        'right_shoulder_pitch_joint': -0.5, 'right_shoulder_roll_joint': 0.3,
        'right_shoulder_yaw_joint': -1.5, 'right_elbow_joint': 2.0,
        'right_wrist_roll_joint': 0, 'right_wrist_pitch_joint': 0, 'right_wrist_yaw_joint': 0,
    }

    # ====================================================================
    #  SECUENCIA
    # ====================================================================
    # (pose_from, pose_to, duracion_transicion, duracion_hold)
    sequence = [
        (ARMS_STAND,   ARMS_TPOSE,   1.2, 1.5),
        (ARMS_TPOSE,   ARMS_FLEX,    0.8, 2.0),
        (ARMS_FLEX,    ARMS_JOJO,    1.0, 2.5),
        (ARMS_JOJO,    ARMS_DAB,     0.7, 1.5),
        (ARMS_DAB,     ARMS_LEAN,    1.0, 2.0),
        (ARMS_LEAN,    ARMS_SALUTE,  0.8, 2.0),
        (ARMS_SALUTE,  ARMS_VICTORY, 0.6, 2.0),
        (ARMS_VICTORY, ARMS_VILLAIN, 1.0, 2.5),
        (ARMS_VILLAIN, ARMS_STAND,   1.2, 1.0),
    ]

    phase_times = []
    t_accum = 0.0
    for (p_from, p_to, t_trans, t_hold) in sequence:
        phase_times.append((t_accum, t_trans, t_hold, p_from, p_to))
        t_accum += t_trans + t_hold
    TOTAL_CYCLE = t_accum

    def set_ctrl_safe(name, value):
        if name in idx and idx[name] is not None:
            data.ctrl[idx[name]] = value

    def apply_upper_interpolated(pose_a, pose_b, progress):
        p = ease(progress)
        for name in pose_a:
            set_ctrl_safe(name, lerp(pose_a[name], pose_b[name], p))

    def apply_upper_direct(pose):
        for name, val in pose.items():
            set_ctrl_safe(name, val)

    print("=== AURA FARMER v2 - CON GRAVEDAD ===")
    print(f"Controlador de balance PD activo en piernas")
    print(f"  KP_pitch={KP_PITCH}  KD_pitch={KD_PITCH}")
    print(f"  KP_roll={KP_ROLL}   KD_roll={KD_ROLL}")
    print(f"Poses: {len(sequence)} | Ciclo: {TOTAL_CYCLE:.1f}s")
    print("Cerra la ventana para salir.")

    # Inicializar con keyframe oficial (parado)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Error integral para el controlador (se acumula lentamente)
    pitch_integral = 0.0
    roll_integral = 0.0
    KI_PITCH = 0.5
    KI_ROLL = 0.3

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cycle_start = time.time()

        while viewer.is_running():
            step_start = time.time()
            t = (step_start - cycle_start) % TOTAL_CYCLE

            # ============================================================
            # CONTROLADOR DE BALANCE (PID en las piernas)
            # ============================================================
            # Leer orientacion actual del pelvis
            quat = data.qpos[qpos_adr + 3: qpos_adr + 7]
            pitch, roll = quat_to_pitch_roll(quat)

            # Leer velocidades angulares del pelvis
            pitch_vel = data.qvel[dof_adr + 4]  # Velocidad angular en Y
            roll_vel = data.qvel[dof_adr + 3]   # Velocidad angular en X

            # Acumular error integral (con limite anti-windup)
            pitch_integral += pitch * model.opt.timestep
            roll_integral += roll * model.opt.timestep
            pitch_integral = max(-0.5, min(0.5, pitch_integral))
            roll_integral = max(-0.3, min(0.3, roll_integral))

            # Calcular correcciones PID
            pitch_corr = -(KP_PITCH * pitch + KD_PITCH * pitch_vel + KI_PITCH * pitch_integral)
            roll_corr = -(KP_ROLL * roll + KD_ROLL * roll_vel + KI_ROLL * roll_integral)

            # Distribuir la correccion entre cadera y tobillo
            # Cadera recibe mas peso porque tiene mas torque (139 Nm vs 50 Nm en el tobillo)
            hip_pitch_cmd = LEGS_BASE['left_hip_pitch_joint'] + pitch_corr * 0.6
            ankle_pitch_cmd = LEGS_BASE['left_ankle_pitch_joint'] + pitch_corr * 0.4
            hip_roll_cmd = LEGS_BASE['left_hip_roll_joint'] + roll_corr * 0.5
            ankle_roll_cmd = LEGS_BASE['left_ankle_roll_joint'] + roll_corr * 0.5

            # Aplicar a ambas piernas (simetricamente)
            set_ctrl_safe('left_hip_pitch_joint', hip_pitch_cmd)
            set_ctrl_safe('right_hip_pitch_joint', hip_pitch_cmd)
            set_ctrl_safe('left_ankle_pitch_joint', ankle_pitch_cmd)
            set_ctrl_safe('right_ankle_pitch_joint', ankle_pitch_cmd)

            set_ctrl_safe('left_hip_roll_joint', hip_roll_cmd)
            set_ctrl_safe('right_hip_roll_joint', -hip_roll_cmd)  # Signo opuesto por simetria
            set_ctrl_safe('left_ankle_roll_joint', ankle_roll_cmd)
            set_ctrl_safe('right_ankle_roll_joint', -ankle_roll_cmd)

            # Rodillas fijas ligeramente flexionadas (absorben vibraciones)
            set_ctrl_safe('left_knee_joint', LEGS_BASE['left_knee_joint'])
            set_ctrl_safe('right_knee_joint', LEGS_BASE['right_knee_joint'])

            # Hip yaw neutro
            set_ctrl_safe('left_hip_yaw_joint', 0)
            set_ctrl_safe('right_hip_yaw_joint', 0)

            # ============================================================
            # POSES DE AURA (torso y brazos)
            # ============================================================
            for (t_start, t_trans, t_hold, p_from, p_to) in phase_times:
                t_end = t_start + t_trans + t_hold
                if t < t_end:
                    elapsed = t - t_start
                    if elapsed < t_trans:
                        apply_upper_interpolated(p_from, p_to, elapsed / t_trans)
                    else:
                        apply_upper_direct(p_to)
                    break

            # Avanzar simulacion
            mujoco.mj_step(model, data)
            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == '__main__':
    main()
