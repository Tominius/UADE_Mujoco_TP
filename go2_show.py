"""
Go2 Show - Coreografía completa de habilidades para el perro robot Unitree Go2.
Incluye:
1. Pararse y estiramiento
2. Flexiones (Push-ups)
3. Baile de cuerpo en 3D (Twisting / Roll-Pitch)
4. Trote / Caminata fluida
5. Sentarse y dar/sacudir la patita (Wave paw)
6. Echarse a descansar (Lay down) y levantarse
"""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).resolve().parent / "unitree_mujoco" / "unitree_robots" / "go2" / "scene.xml"

def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t

def ease_in_out(t):
    t = max(0.0, min(1.0, t))
    return 0.5 * (1.0 - math.cos(math.pi * t))

def main():
    print(f"Cargando modelo Go2 desde {MODEL_PATH}...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Controlador PD
    KP = 90.0
    KD = 5.0

    # 4 patas: [FL, FR, RL, RR], cada una [hip, thigh, calf]
    # Índices:
    # FL: [0, 1, 2]
    # FR: [3, 4, 5]
    # RL: [6, 7, 8]
    # RR: [9, 10, 11]

    # --- DEFINICIÓN DE POSES CLAVE ---
    q_stand = np.array([
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5
    ], dtype=np.float64)

    q_crouch = np.array([
        0.0, 1.15, -2.3,
        0.0, 1.15, -2.3,
        0.0, 1.15, -2.3,
        0.0, 1.15, -2.3
    ], dtype=np.float64)

    q_high = np.array([
        0.0, 0.45, -0.9,
        0.0, 0.45, -0.9,
        0.0, 0.45, -0.9,
        0.0, 0.45, -0.9
    ], dtype=np.float64)

    q_sit = np.array([
        0.0, 0.5, -1.0,   # FL (delantera alta)
        0.0, 0.5, -1.0,   # FR (delantera alta)
        0.0, 1.25, -2.45, # RL (trasera plegada)
        0.0, 1.25, -2.45  # RR (trasera plegada)
    ], dtype=np.float64)

    q_lay = np.array([
        0.0, 1.4, -2.6,
        0.0, 1.4, -2.6,
        0.0, 1.4, -2.6,
        0.0, 1.4, -2.6
    ], dtype=np.float64)

    # Duraciones de cada fase en segundos
    PHASES = [
        ("PARARSE Y ESTIRAMIENTO", 3.0),
        ("FLEXIONES (PUSH-UPS)", 4.0),
        ("BAILE DE CUERPO (TWIST / ROLL)", 5.0),
        ("TROTE / CAMINATA", 5.0),
        ("SENTARSE Y SALUDAR CON LA PATA", 6.0),
        ("ECHARSE A DESCANSAR", 4.0),
    ]
    TOTAL_SHOW_TIME = sum(dur for _, dur in PHASES)

    print("\n" + "="*50)
    print("       UNITREE GO2 - GRAN SHOW DE HABILIDADES")
    print("="*50)
    for name, dur in PHASES:
        print(f"  * {name} ({dur}s)")
    print(f"  Ciclo total: {TOTAL_SHOW_TIME}s (repite automáticamente)")
    print("  Cerrá la ventana para salir.")
    print("="*50 + "\n")

    current_phase_name = ""

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        sim_time = 0.0

        while viewer.is_running():
            wall = time.time() - t0
            if wall - sim_time > 0.05:
                sim_time = wall - 0.05

            while sim_time < wall:
                t = data.time % TOTAL_SHOW_TIME
                target = q_stand.copy()

                # --- MÁQUINA DE ESTADOS DEL SHOW ---
                if t < 3.0:
                    # 1. PARARSE Y ESTIRAMIENTO
                    phase_name = "1. Pararse y Estiramiento"
                    tau_phase = t / 3.0
                    pitch_wave = math.sin(tau_phase * 2 * math.pi) * 0.2
                    # Inclinación adelante/atrás
                    target[1] = q_stand[1] + pitch_wave
                    target[4] = q_stand[4] + pitch_wave
                    target[7] = q_stand[7] - pitch_wave
                    target[10] = q_stand[10] - pitch_wave

                elif t < 7.0:
                    # 2. FLEXIONES (PUSH-UPS)
                    phase_name = "2. Flexiones (Push-ups)"
                    tau_phase = (t - 3.0)
                    pushup = (math.sin(tau_phase * 2 * math.pi * 0.8) + 1.0) * 0.5
                    target = lerp(q_high, q_crouch, pushup)

                elif t < 12.0:
                    # 3. BAILE DE CUERPO 3D (TWIST / CIRCLE)
                    phase_name = "3. Baile de Cuerpo (Twist 3D)"
                    tau_phase = (t - 7.0) * 2 * math.pi * 0.75
                    roll = math.sin(tau_phase) * 0.22
                    pitch = math.cos(tau_phase) * 0.22

                    # Izquierda vs Derecha (Roll) y Adelante vs Atrás (Pitch)
                    target[1] = q_stand[1] + pitch + roll   # FL
                    target[4] = q_stand[4] + pitch - roll   # FR
                    target[7] = q_stand[7] - pitch + roll   # RL
                    target[10] = q_stand[10] - pitch - roll # RR

                elif t < 17.0:
                    # 4. TROTE / CAMINATA
                    phase_name = "4. Trote / Caminata"
                    walk_t = t - 12.0
                    freq = 2.0 * math.pi * 1.8
                    phase_A = walk_t * freq
                    phase_B = walk_t * freq + math.pi
                    amp_hip = 0.26
                    amp_knee = 0.38

                    target[1] = q_stand[1] + math.sin(phase_A) * amp_hip
                    target[2] = q_stand[2] + max(0.0, math.cos(phase_A)) * amp_knee
                    target[10] = q_stand[10] + math.sin(phase_A) * amp_hip
                    target[11] = q_stand[11] + max(0.0, math.cos(phase_A)) * amp_knee

                    target[4] = q_stand[4] + math.sin(phase_B) * amp_hip
                    target[5] = q_stand[5] + max(0.0, math.cos(phase_B)) * amp_knee
                    target[7] = q_stand[7] + math.sin(phase_B) * amp_hip
                    target[8] = q_stand[8] + max(0.0, math.cos(phase_B)) * amp_knee

                elif t < 23.0:
                    # 5. SENTARSE Y SALUDAR CON LA PATA
                    phase_name = "5. Sentarse y Dar la Pata"
                    sit_t = t - 17.0

                    if sit_t < 1.5:
                        # Transición suave a sentarse
                        trans = ease_in_out(sit_t / 1.5)
                        target = lerp(q_stand, q_sit, trans)
                    else:
                        # Ya sentado -> levanta y sacude pata delantera izquierda (FL)
                        target = q_sit.copy()
                        wave_t = (sit_t - 1.5) * 2 * math.pi * 2.0  # 2 Hz sacudida
                        target[0] = 0.15                             # Abducción hacia afuera
                        target[1] = -0.3 + math.sin(wave_t) * 0.25   # Muslo arriba y agitando
                        target[2] = -0.7 + math.cos(wave_t) * 0.2    # Rodilla doblada

                else:
                    # 6. ECHARSE A DESCANSAR (LAY DOWN)
                    phase_name = "6. Echarse a Descansar"
                    lay_t = t - 23.0
                    if lay_t < 1.5:
                        trans = ease_in_out(lay_t / 1.5)
                        target = lerp(q_sit, q_lay, trans)
                    elif lay_t < 3.0:
                        target = q_lay.copy()
                    else:
                        # Transición de vuelta a parado
                        trans = ease_in_out((lay_t - 3.0) / 1.0)
                        target = lerp(q_lay, q_stand, trans)

                # Mostrar fase en consola cuando cambia
                if phase_name != current_phase_name:
                    current_phase_name = phase_name
                    print(f"[{data.time:.1f}s] Ejecutando: {current_phase_name}")

                # Torques PD
                q_cur = data.qpos[7:19]
                v_cur = data.qvel[6:18]
                torques = KP * (target - q_cur) - KD * v_cur
                torques = np.clip(torques, -35.0, 35.0)

                data.ctrl[:] = torques
                mujoco.mj_step(model, data)
                sim_time += model.opt.timestep

            viewer.sync()

if __name__ == "__main__":
    main()
