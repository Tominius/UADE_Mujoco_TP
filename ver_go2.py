import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

# Cargar el modelo del Go2 directamente desde el repo oficial que descargamos
REPO_DIR = Path(__file__).resolve().parent / "unitree_mujoco"
MODEL_PATH = REPO_DIR / "unitree_robots" / "go2" / "scene.xml"

def main():
    if not MODEL_PATH.exists():
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return

    print("Cargando modelo del Unitree Go2...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Posiciones objetivo para que el perro se pare (Stand up)
    # Son 12 motores (3 por pata: cadera_x, cadera_y, rodilla)
    stand_up_pos = np.array([
        0.005, 0.608, -1.217,  # Pata delantera derecha
       -0.005, 0.608, -1.217,  # Pata delantera izquierda
        0.005, 0.608, -1.217,  # Pata trasera derecha
       -0.005, 0.608, -1.217   # Pata trasera izquierda
    ])

    print("Iniciando simulador...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        
        while viewer.is_running():
            step_start = time.perf_counter()
            
            # Asignar la posición objetivo a todos los motores usando PD control simple
            for i in range(12):
                data.ctrl[i] = stand_up_pos[i]

            mujoco.mj_step(model, data)
            viewer.sync()

            # Mantener el simulador a la velocidad correcta
            time_until_next = model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    main()
