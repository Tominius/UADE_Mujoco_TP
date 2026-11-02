"""
Backflip realista del Unitree G1 en MuJoCo.

Datos del robot (sacados directamente de g1.xml):
  - Masa total: ~35 kg (sumando todas las inertials del XML)
  - Altura parada: 0.793 m (posicion inicial del pelvis en el XML)
  - Keyframe "stand": qpos y ctrl oficiales del modelo

Limites de articulaciones relevantes (en radianes, del XML):
  - left/right_hip_pitch:  [-2.53, 2.88]
  - left/right_knee:       [-0.087, 2.88]
  - left/right_ankle_pitch:[-0.87, 0.52]
  - left/right_shoulder_pitch: [-3.09, 2.67]
  - left/right_elbow:      [-1.05, 2.09]
  - waist_pitch:           [-0.52, 0.52]

Fisica del salto:
  - Para saltar ~0.6m de altura se necesita v0 = sqrt(2*g*h) = sqrt(2*9.81*0.6) = 3.43 m/s
  - Tiempo de vuelo total: 2*v0/g = 0.70 s
  - Velocidad angular necesaria para 1 vuelta en 0.70s: 2*pi/0.70 = ~9.0 rad/s
"""

import mujoco
import mujoco.viewer
import time
import math
import numpy as np


def get_ctrl_index(model, name):
    """Obtiene el indice del actuador por nombre, o None si no existe."""
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return idx if idx != -1 else None


def set_ctrl(data, idx_map, name, value):
    """Setea un control si el actuador existe."""
    if name in idx_map and idx_map[name] is not None:
        data.ctrl[idx_map[name]] = value


def lerp(a, b, t):
    """Interpolacion lineal entre a y b con t en [0, 1]."""
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def main():
    model = mujoco.MjModel.from_xml_path('mujoco_menagerie/unitree_g1/scene.xml')
    data = mujoco.MjData(model)

    # --- Obtener indices de todos los actuadores ---
    actuator_names = [
        # Piernas
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        # Torso
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
        # Brazos
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint', 'left_elbow_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint', 'right_elbow_joint',
    ]
    idx = {name: get_ctrl_index(model, name) for name in actuator_names}

    # --- Datos del freejoint (raiz del robot) ---
    root_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_base_joint')
    qpos_adr = model.jnt_qposadr[root_jnt]
    dof_adr = model.jnt_dofadr[root_jnt]

    # --- Posiciones articulares clave (en radianes) ---
    # Sacadas del keyframe "stand" del XML y de los limites de cada joint

    # POSE PARADO (del keyframe oficial "stand" en g1.xml linea 347)
    STAND = {
        'left_hip_pitch_joint': 0, 'left_hip_roll_joint': 0, 'left_hip_yaw_joint': 0,
        'left_knee_joint': 0, 'left_ankle_pitch_joint': 0, 'left_ankle_roll_joint': 0,
        'right_hip_pitch_joint': 0, 'right_hip_roll_joint': 0, 'right_hip_yaw_joint': 0,
        'right_knee_joint': 0, 'right_ankle_pitch_joint': 0, 'right_ankle_roll_joint': 0,
        'waist_yaw_joint': 0, 'waist_roll_joint': 0, 'waist_pitch_joint': 0,
        'left_shoulder_pitch_joint': 0.2, 'left_shoulder_roll_joint': 0.2,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 1.28,
        'right_shoulder_pitch_joint': 0.2, 'right_shoulder_roll_joint': -0.2,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 1.28,
    }

    # POSE AGACHADO (flexion profunda de rodillas y caderas, respetando limites)
    CROUCH = {
        'left_hip_pitch_joint': -1.2, 'left_hip_roll_joint': 0, 'left_hip_yaw_joint': 0,
        'left_knee_joint': 2.4, 'left_ankle_pitch_joint': -0.7, 'left_ankle_roll_joint': 0,
        'right_hip_pitch_joint': -1.2, 'right_hip_roll_joint': 0, 'right_hip_yaw_joint': 0,
        'right_knee_joint': 2.4, 'right_ankle_pitch_joint': -0.7, 'right_ankle_roll_joint': 0,
        'waist_yaw_joint': 0, 'waist_roll_joint': 0, 'waist_pitch_joint': 0.3,
        # Brazos hacia atras para tomar impulso
        'left_shoulder_pitch_joint': 1.5, 'left_shoulder_roll_joint': 0.3,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 0.5,
        'right_shoulder_pitch_joint': 1.5, 'right_shoulder_roll_joint': -0.3,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 0.5,
    }

    # POSE TUCK (bolita en el aire - para conservar momento angular y girar rapido)
    TUCK = {
        'left_hip_pitch_joint': -2.0, 'left_hip_roll_joint': 0, 'left_hip_yaw_joint': 0,
        'left_knee_joint': 2.8, 'left_ankle_pitch_joint': -0.5, 'left_ankle_roll_joint': 0,
        'right_hip_pitch_joint': -2.0, 'right_hip_roll_joint': 0, 'right_hip_yaw_joint': 0,
        'right_knee_joint': 2.8, 'right_ankle_pitch_joint': -0.5, 'right_ankle_roll_joint': 0,
        'waist_yaw_joint': 0, 'waist_roll_joint': 0, 'waist_pitch_joint': -0.4,
        # Brazos abrazando las rodillas
        'left_shoulder_pitch_joint': -2.5, 'left_shoulder_roll_joint': 0.5,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 2.0,
        'right_shoulder_pitch_joint': -2.5, 'right_shoulder_roll_joint': -0.5,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 2.0,
    }

    # POSE EXTENSION (abrirse antes de aterrizar)
    EXTEND = {
        'left_hip_pitch_joint': -0.3, 'left_hip_roll_joint': 0, 'left_hip_yaw_joint': 0,
        'left_knee_joint': 0.4, 'left_ankle_pitch_joint': -0.3, 'left_ankle_roll_joint': 0,
        'right_hip_pitch_joint': -0.3, 'right_hip_roll_joint': 0, 'right_hip_yaw_joint': 0,
        'right_knee_joint': 0.4, 'right_ankle_pitch_joint': -0.3, 'right_ankle_roll_joint': 0,
        'waist_yaw_joint': 0, 'waist_roll_joint': 0, 'waist_pitch_joint': 0,
        # Brazos arriba para estabilizar
        'left_shoulder_pitch_joint': -1.0, 'left_shoulder_roll_joint': 1.2,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 0.3,
        'right_shoulder_pitch_joint': -1.0, 'right_shoulder_roll_joint': -1.2,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 0.3,
    }

    # POSE ATERRIZAJE (absorber impacto con rodillas flexionadas)
    LAND = {
        'left_hip_pitch_joint': -0.8, 'left_hip_roll_joint': 0, 'left_hip_yaw_joint': 0,
        'left_knee_joint': 1.6, 'left_ankle_pitch_joint': -0.5, 'left_ankle_roll_joint': 0,
        'right_hip_pitch_joint': -0.8, 'right_hip_roll_joint': 0, 'right_hip_yaw_joint': 0,
        'right_knee_joint': 1.6, 'right_ankle_pitch_joint': -0.5, 'right_ankle_roll_joint': 0,
        'waist_yaw_joint': 0, 'waist_roll_joint': 0, 'waist_pitch_joint': 0.2,
        # Brazos al frente para balance
        'left_shoulder_pitch_joint': -0.5, 'left_shoulder_roll_joint': 0.8,
        'left_shoulder_yaw_joint': 0, 'left_elbow_joint': 0.8,
        'right_shoulder_pitch_joint': -0.5, 'right_shoulder_roll_joint': -0.8,
        'right_shoulder_yaw_joint': 0, 'right_elbow_joint': 0.8,
    }

    def apply_pose(pose_a, pose_b, progress):
        """Interpola entre dos poses y aplica a los controles."""
        for name in pose_a:
            val = lerp(pose_a[name], pose_b[name], progress)
            set_ctrl(data, idx, name, val)

    def apply_pose_direct(pose):
        """Aplica una pose directamente."""
        for name, val in pose.items():
            set_ctrl(data, idx, name, val)

    # --- Constantes fisicas del salto ---
    # v0 = sqrt(2*g*h) para saltar h metros
    JUMP_HEIGHT = 0.55  # metros (realista para un robot de 35kg con piernas cortas)
    G = 9.81
    V0_UP = math.sqrt(2 * G * JUMP_HEIGHT)  # ~3.28 m/s
    FLIGHT_TIME = 2 * V0_UP / G              # ~0.67 s
    OMEGA_BACK = 2 * math.pi / FLIGHT_TIME   # ~9.4 rad/s (1 vuelta completa)

    # --- Duraciones de cada fase (en segundos) ---
    T_STAND   = 1.0   # Pararse derecho
    T_CROUCH  = 0.8   # Agacharse tomando impulso
    T_LAUNCH  = 0.0   # Instante del salto (instantaneo)
    T_TUCK    = FLIGHT_TIME * 0.5   # Primera mitad del vuelo: bolita
    T_EXTEND  = FLIGHT_TIME * 0.5   # Segunda mitad: abrirse
    T_LAND    = 1.5   # Absorber impacto y estabilizar
    T_RECOVER = 2.0   # Volver a pararse
    T_PAUSE   = 1.5   # Pausa antes de repetir

    TOTAL_CYCLE = T_STAND + T_CROUCH + T_TUCK + T_EXTEND + T_LAND + T_RECOVER + T_PAUSE

    print("=== BACKFLIP REALISTA - Unitree G1 ===")
    print(f"Masa total:     ~35 kg")
    print(f"Altura salto:   {JUMP_HEIGHT} m")
    print(f"Vel. inicial:   {V0_UP:.2f} m/s")
    print(f"Tiempo vuelo:   {FLIGHT_TIME:.2f} s")
    print(f"Vel. angular:   {OMEGA_BACK:.1f} rad/s ({OMEGA_BACK * 180/math.pi:.0f} deg/s)")
    print(f"Ciclo total:    {TOTAL_CYCLE:.1f} s")
    print("Cerrra la ventana para salir.")

    # Inicializar con la pose de parado (keyframe oficial)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cycle_start = time.time()
        launched = False

        while viewer.is_running():
            step_start = time.time()
            t = step_start - cycle_start

            # ---- FASE 1: PARADO (estabilizar antes de empezar) ----
            if t < T_STAND:
                apply_pose_direct(STAND)
                launched = False

            # ---- FASE 2: AGACHARSE (tomar impulso) ----
            elif t < T_STAND + T_CROUCH:
                progress = (t - T_STAND) / T_CROUCH
                # Curva ease-in-out para movimiento suave
                progress = 0.5 * (1 - math.cos(math.pi * progress))
                apply_pose(STAND, CROUCH, progress)

            # ---- FASE 3: DESPEGUE + TUCK (primera mitad del vuelo) ----
            elif t < T_STAND + T_CROUCH + T_TUCK:
                if not launched:
                    # Impulso instantaneo: velocidad vertical + rotacion hacia atras
                    data.qvel[dof_adr + 2] = V0_UP           # Velocidad hacia arriba
                    data.qvel[dof_adr + 4] = -OMEGA_BACK     # Rotacion en Y (backflip)
                    # Brazos hacia adelante y arriba (impulso como un gimnasta)
                    set_ctrl(data, idx, 'left_shoulder_pitch_joint', -2.8)
                    set_ctrl(data, idx, 'right_shoulder_pitch_joint', -2.8)
                    launched = True

                progress = (t - T_STAND - T_CROUCH) / T_TUCK
                progress = min(1.0, progress * 1.5)  # Tuck rapido
                apply_pose(CROUCH, TUCK, progress)

            # ---- FASE 4: EXTENSION (segunda mitad del vuelo, preparar aterrizaje) ----
            elif t < T_STAND + T_CROUCH + T_TUCK + T_EXTEND:
                progress = (t - T_STAND - T_CROUCH - T_TUCK) / T_EXTEND
                progress = 0.5 * (1 - math.cos(math.pi * progress))
                apply_pose(TUCK, EXTEND, progress)

            # ---- FASE 5: ATERRIZAJE (absorber impacto) ----
            elif t < T_STAND + T_CROUCH + T_TUCK + T_EXTEND + T_LAND:
                progress = (t - T_STAND - T_CROUCH - T_TUCK - T_EXTEND) / T_LAND
                progress = 0.5 * (1 - math.cos(math.pi * progress))
                apply_pose(EXTEND, LAND, progress)

            # ---- FASE 6: RECUPERACION (levantarse) ----
            elif t < T_STAND + T_CROUCH + T_TUCK + T_EXTEND + T_LAND + T_RECOVER:
                progress = (t - T_STAND - T_CROUCH - T_TUCK - T_EXTEND - T_LAND) / T_RECOVER
                progress = 0.5 * (1 - math.cos(math.pi * progress))
                apply_pose(LAND, STAND, progress)

            # ---- FASE 7: PAUSA Y REINICIO ----
            elif t < TOTAL_CYCLE:
                apply_pose_direct(STAND)
            else:
                # Reiniciar todo el ciclo
                mujoco.mj_resetDataKeyframe(model, data, 0)
                cycle_start = time.time()
                launched = False

            # Avanzar simulacion
            mujoco.mj_step(model, data)
            viewer.sync()

            # Mantener tiempo real
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == '__main__':
    main()
