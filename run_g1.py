import mujoco
import mujoco.viewer
import time

def main():
    # Cargar el modelo del Unitree G1 desde la carpeta de mujoco_menagerie
    # Usamos scene.xml porque incluye el robot y el suelo
    model_path = 'mujoco_menagerie/unitree_g1/scene.xml'
    
    print(f"Cargando modelo desde: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Iniciar el visor de MuJoCo
    print("Iniciando la simulación. Presiona ESC en la ventana para salir.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Bucle principal de simulación
        while viewer.is_running():
            step_start = time.time()

            # Avanzar un paso en la simulación física
            # Acá es donde en el futuro puedes agregar tu código para mover los motores (data.ctrl)
            mujoco.mj_step(model, data)

            # Sincronizar el estado con el visor 3D
            viewer.sync()

            # Sincronización básica del tiempo para que se vea a tiempo real
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()
