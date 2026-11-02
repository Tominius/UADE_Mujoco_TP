# Simulación y Control de Robots en MuJoCo: Unitree Go2 & G1

Proyecto universitario de robótica avanzada, simulación física y control dinámico implementado en **MuJoCo** para los robots **Unitree Go2** (cuadrúpedo) y **Unitree G1** (humanoide bipedal).

---

## 🚀 Inicio Rápido (Quickstart)

### 1. Clonar el Repositorio
```bash
git clone https://github.com/Tominius/UADE_Mujoco_TP.git
cd UADE_Mujoco_TP
```

### 2. Instalar Dependencias
Se recomienda Python 3.10 o superior:
```bash
pip install -r requirements.txt
```

### 3. Ejecutar las Demostraciones
- **En Windows:** Hacé doble clic en `INICIAR_DEMOS.bat`
- **O desde la terminal:**
  ```bash
  python menu.py
  ```

---

## 📋 Las 3 Demostraciones

El menú interactivo permite seleccionar y ejecutar cualquiera de las 3 funciones principales:

### `[1]` Perro Go2 - Gran Show de Habilidades (`go2_show.py`)
Rutina continua que recorre 6 comportamientos dinámicos:
1. **Pararse y Estiramiento:** Inclinaciones controladas en cabeceo (*pitch*).
2. **Flexiones (Push-ups):** Flexión y extensión rítmica de codos y rodillas.
3. **Baile de Cuerpo 3D:** Rotación tridimensional del torso combinando balanceo lateral (*roll*) y cabeceo (*pitch*).
4. **Trote Dinámico:** Caminata en línea recta con zancada simétrica.
5. **Sentarse y Saludar:** Apoyo sobre el tren trasero, elevación de pata delantera y sacudida en el aire.
6. **Echarse a Descansar:** Plegado completo en el suelo y vuelta a posición bípeda/cuadrúpeda.

* **Fundamento Teórico:** Los actuadores del Go2 son motores de torque directo (`<motor>`). Se diseñó un controlador PD en tiempo real ($\tau = K_p(q^* - q) - K_d\dot{q}$) con $K_p=90, K_d=5$ corriendo a 500 Hz para otorgar rigidez elástica contra la gravedad.

---

### `[2]` Perro Go2 - Trote Dinámico Estable (`caminar_go2.py`)
Caminata y trote continuo hacia adelante con marcha de trote simétrica (*Trot Gait*).

* **Fundamento Teórico:** Generador de marcha por osciladores sinusoidales en contrafase ($180^\circ$). Las patas se coordinan en pares diagonales (Front-Left + Rear-Right y Front-Right + Rear-Left) a 1.8 Hz, levantando la rodilla durante la fase positiva de la cadera.

---

### `[3]` Humanoide G1 - Equilibrio y Push Recovery (`g1_empujon_equilibrio.py`)
El robot humanoide de 29 grados de libertad mantiene el equilibrio bajo gravedad real mediante una red neuronal de **Deep Reinforcement Learning** (`walker.onnx`) entrenada por PPO a 200 Hz.

* **Empujones Automáticos:** Cada 4 segundos el simulador le aplica impactos de 45 N a 65 N en el torso (frontal, trasero y laterales). El robot reacomoda tobillos y rodillas absorbiendo la perturbación sin caerse.
* **Controles Interactivos en Vivo:**
  * `W` / `Arriba`: Empujón frontal (+60 N).
  * `S` / `Abajo`: Empujón trasero (-60 N).
  * `A` / `Izquierda`: Empujón lateral izquierdo (+45 N).
  * `D` / `Derecha`: Empujón lateral derecho (-45 N).
  * `Espacio`: Empujón en dirección aleatoria.
  * `Ctrl` + `Clic Derecho` + Arrastrar con el mouse: Jalar al robot interactivamente.

---

## 📁 Estructura del Repositorio

```
├── INICIAR_DEMOS.bat          # Lanzador 1-click para Windows
├── menu.py                    # Menú interactivo en terminal
├── go2_show.py                # Demo 1: Show de habilidades Go2
├── caminar_go2.py             # Demo 2: Trote Go2
├── g1_empujon_equilibrio.py   # Demo 3: Push recovery G1 con IA
├── requirements.txt           # Dependencias de Python
├── README.md                  # Documentación
└── models/                    # Modelos MJCF, mallas y pesos ONNX
    ├── unitree_go2/           # Modelo del cuadrúpedo Go2
    └── unitree_g1/            # Modelo del humanoide G1 + walker.onnx
```

---

## 👥 Historial y Ramas

- **Rama `main`:** Versión limpia, optimizada y autocontenida con las 3 funciones principales para evaluación y presentación.
- **Rama `v1-completo` (Tag `v1.0-completo`):** Contiene el historial completo de iteraciones preliminares (experimentos de mocap de baile, backflip, bridge DDS, etc.).
