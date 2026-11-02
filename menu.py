"""
PANEL PRINCIPAL DE DEMOSTRACIONES (SELECCIONADAS PARA EL PROFESOR)
1. Perro Go2 - Gran Show de Habilidades
2. Perro Go2 - Trote Dinamico Estable
3. Humanoide G1 - Equilibrio y Recuperacion ante Empujones (Push Recovery)
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEMOS = {
    "1": {
        "title": "Perro Go2 - Gran Show de Habilidades",
        "desc": "Rutina completa: flexiones (push-ups), torsion 3D de columna, trote, sentarse, dar la patita y descanso.",
        "cmd": [sys.executable, str(BASE_DIR / "go2_show.py")],
        "tech": "Control por Torque PD en tiempo real (Kp=90, Kd=5) a 500 Hz sobre motores de torque puro."
    },
    "2": {
        "title": "Perro Go2 - Trote Dinamico Estable",
        "desc": "Caminata y trote continuo hacia adelante con marcha diagonal simetrica.",
        "cmd": [sys.executable, str(BASE_DIR / "caminar_go2.py")],
        "tech": "Generador de marcha senoidal Trot Gait en contrafase (180 deg) por pares diagonales a 1.8 Hz."
    },
    "3": {
        "title": "Humanoide G1 - Equilibrio y Recuperacion ante Empujones (Push Recovery)",
        "desc": "El robot humanoide recibe empujones ciclicos (frontales, traseros y laterales) o manuales por teclado, absorbe el impacto y recupera el equilibrio sin caerse.",
        "cmd": [sys.executable, str(BASE_DIR / "g1_empujon_equilibrio.py")],
        "tech": "Deep Reinforcement Learning (walker.onnx a 200 Hz). Rechazo de perturbaciones (Disturbance Rejection)."
    }
}

def print_header():
    print("=" * 72)
    print("       PANEL DE DEMOSTRACIONES SELECCIONADAS - ROBOTICA (MUJOCO)")
    print("                Unitree Go2 (Cuadrupedo) & Unitree G1 (Humanoide)")
    print("=" * 72)
    print()

def main():
    while True:
        print_header()
        for key, info in DEMOS.items():
            print(f"  [{key}] {info['title']}")
            print(f"      -> {info['desc']}")
            print(f"      [Tecnica: {info['tech']}]")
            print()
        print("  [0] Salir")
        print()
        print("=" * 72)
        
        choice = input("Selecciona una opcion (1, 2, 3 o 0 para salir): ").strip()
        
        if choice == "0":
            print("\nHasta luego!")
            break
            
        if choice in DEMOS:
            demo = DEMOS[choice]
            print(f"\n>>> Lanzando: {demo['title']}...")
            print(">>> Cierra la ventana 3D de MuJoCo cuando termines para volver a este menu.\n")
            try:
                subprocess.run(demo["cmd"])
            except KeyboardInterrupt:
                print("\nInterrumpido por el usuario.")
            except Exception as e:
                print(f"\nError al ejecutar la demo: {e}")
            print("\n" + "-" * 72 + "\n")
        else:
            print("\nOpcion no valida. Por favor ingresa 1, 2 o 3.\n")

if __name__ == "__main__":
    main()
