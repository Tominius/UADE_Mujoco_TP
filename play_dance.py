"""
Reproductor Local de G1 Moves (Animaciones CSV)
Lee los archivos CSV del repositorio Unitree-G1MuJoCo-Viser y los reproduce
en el visor local de MuJoCo mediante cinemática (sobrescribiendo qpos).
"""

import sys
import time
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Ruta al repo de Viser
VISER_REPO = Path(__file__).resolve().parent / "Unitree-G1MuJoCo-Viser"
CSV_DIR = VISER_REPO / "g1_moves" / "g1_moves_outputs" / "web_csv"
# Usamos el modelo especifico del repo Viser que coincida perfectamente con los CSV
MODEL_PATH = VISER_REPO / "assets" / "proto_g1" / "g1_holo_compat_web.xml"

def list_dances():
    if not CSV_DIR.exists():
        print(f"Error: No se encontro el directorio {CSV_DIR}")
        return []
    
    csv_files = list(CSV_DIR.glob("*.csv"))
    dances = [f.stem for f in csv_files]
    return dances

def main():
    parser = argparse.ArgumentParser(description="G1 Dance Player (CSV)")
    parser.add_argument("dance_name", nargs="?", default="dance_B_DadDance_web",
                        help="Nombre del baile a reproducir (sin .csv)")
    parser.add_argument("--list", action="store_true", help="Listar todos los bailes disponibles")
    parser.add_argument("--fps", type=float, default=60.0, help="FPS de reproduccion (default: 60)")
    
    args = parser.parse_args()

    if args.list:
        print("\n--- Bailes Disponibles ---")
        for d in sorted(list_dances()):
            print(f" - {d}")
        print("\nPara correr uno: python play_dance.py <nombre_del_baile>")
        return

    csv_path = CSV_DIR / f"{args.dance_name}.csv"
    if not csv_path.exists():
        # Intentar buscar coincidencia parcial
        matches = [f for f in list_dances() if args.dance_name.lower() in f.lower()]
        if len(matches) == 1:
            csv_path = CSV_DIR / f"{matches[0]}.csv"
            print(f"Autocompletado a: {matches[0]}")
        else:
            print(f"Error: No se encontro {csv_path.name}.")
            print("Usa 'python play_dance.py --list' para ver las opciones.")
            return

    print(f"\nCargando animacion: {csv_path.name}...")
    try:
        # Cargar trayectoria (N frames x 36 valores)
        trajectory = np.loadtxt(csv_path, delimiter=",")
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        return

    num_frames = trajectory.shape[0]
    duration = num_frames / args.fps
    print(f"Frames: {num_frames} | Duracion: {duration:.2f}s | FPS: {args.fps}")

    # Cargar modelo
    if not MODEL_PATH.exists():
        print(f"Error: No se encontro el modelo {MODEL_PATH}")
        return
        
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print("\nAbriendo visor de MuJoCo...")
    print("El baile se repetira en loop infinito. Cerra la ventana para salir.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        
        while viewer.is_running():
            wall = time.time() - t0
            
            # Calcular que frame deberiamos mostrar segun el tiempo real
            # Usamos modulo para que sea un loop infinito
            current_time = wall % duration
            frame_idx = int(current_time * args.fps)
            
            # Asegurarnos de no exceder los limites por redondeo
            if frame_idx >= num_frames:
                frame_idx = num_frames - 1
                
            # Modo cinematico: sobrescribir qpos directamente
            # La trayectoria CSV tiene 36 valores (3 pos root, 4 quat root, 29 joints)
            data.qpos[:] = trajectory[frame_idx]
            
            # Calcular forward kinematics para actualizar geometrias sin simular fisica
            mujoco.mj_forward(model, data)
            
            viewer.sync()
            
            # Pequeño sleep para no freir la CPU
            time.sleep(0.005)

if __name__ == "__main__":
    main()
