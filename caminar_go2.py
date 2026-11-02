"""
Caminar Go2 - Generador de marcha trot gait con control PD de torque.
El robot camina hacia adelante de forma continua y estable.
"""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Modelo del Unitree Go2
MODEL_PATH = Path(__file__).resolve().parent / "models" / "unitree_go2" / "scene.xml"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).resolve().parent / "mujoco_menagerie" / "unitree_go2" / "scene.xml"

def main():
    print(f"Cargando modelo Go2 desde {MODEL_PATH}...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    # Iniciar en pose home
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Ganancias del controlador PD de torque
    # Los actuadores del Go2 son motores de torque puro (<motor>),
    # por lo que el PD en software es indispensable para darles rigidez.
    KP = 85.0
    KD = 4.5

    # Pose base de parado (4 patas: FL, FR, RL, RR)
    # Cada pata tiene 3 joints: [hip, thigh, calf]
    base_q = np.array([
        0.0, 0.8, -1.5,  # FL (Front Left)
        0.0, 0.8, -1.5,  # FR (Front Right)
        0.0, 0.8, -1.5,  # RL (Rear Left)
        0.0, 0.8, -1.5   # RR (Rear Right)
    ], dtype=np.float64)

    print("\n" + "="*45)
    print("  UNITREE GO2 - TROTE Y CAMINATA ESTABLE")
    print("="*45)
    print("  Controlador: Generador de Marcha + PD Torque")
    print("  Cerrá la ventana para salir.")
    print("="*45 + "\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        sim_time = 0.0

        while viewer.is_running():
            wall = time.time() - t0
            if wall - sim_time > 0.05:
                sim_time = wall - 0.05

            while sim_time < wall:
                step_t = data.time
                target = base_q.copy()

                # Esperar 0.8s iniciales para estabilizar
                if step_t > 0.8:
                    walk_t = step_t - 0.8
                    freq = 2.0 * math.pi * 1.8  # Frecuencia del trote (1.8 Hz)

                    # Pares diagonales:
                    # Par A: Front Left (FL) + Rear Right (RR)
                    phase_A = walk_t * freq
                    # Par B: Front Right (FR) + Rear Left (RL) - Desfasados 180° (pi)
                    phase_B = walk_t * freq + math.pi

                    amp_hip = 0.26
                    amp_knee = 0.38

                    # Par A (FL: 0,1,2 | RR: 9,10,11)
                    target[1] = base_q[1] + math.sin(phase_A) * amp_hip
                    target[2] = base_q[2] + max(0.0, math.cos(phase_A)) * amp_knee

                    target[10] = base_q[10] + math.sin(phase_A) * amp_hip
                    target[11] = base_q[11] + max(0.0, math.cos(phase_A)) * amp_knee

                    # Par B (FR: 3,4,5 | RL: 6,7,8)
                    target[4] = base_q[4] + math.sin(phase_B) * amp_hip
                    target[5] = base_q[5] + max(0.0, math.cos(phase_B)) * amp_knee

                    target[7] = base_q[7] + math.sin(phase_B) * amp_hip
                    target[8] = base_q[8] + max(0.0, math.cos(phase_B)) * amp_knee

                # Calcular Torques PD
                q_cur = data.qpos[7:19]
                v_cur = data.qvel[6:18]
                torques = KP * (target - q_cur) - KD * v_cur
                torques = np.clip(torques, -30.0, 30.0)

                data.ctrl[:] = torques
                mujoco.mj_step(model, data)
                sim_time += model.opt.timestep

            viewer.sync()

if __name__ == "__main__":
    main()
