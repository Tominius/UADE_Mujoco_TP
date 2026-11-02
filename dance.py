import mujoco
import mujoco.viewer
import time
import math

def main():
    model_path = 'mujoco_menagerie/unitree_g1/scene.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Actuadores que vamos a mover para el "Six Seven"
    actuators = [
        'left_shoulder_roll_joint',
        'right_shoulder_roll_joint',
        'left_shoulder_pitch_joint',
        'right_shoulder_pitch_joint',
        'left_hip_roll_joint',
        'right_hip_roll_joint',
        'left_knee_joint',
        'right_knee_joint',
        'waist_yaw_joint'
    ]
    
    # Buscamos en qué índice está cada actuador
    ctrl_indices = {}
    for name in actuators:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx != -1:
            ctrl_indices[name] = idx

    # Buscamos la articulación raíz (el cuerpo entero) para mantenerlo en el aire y que no se caiga
    root_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    qpos_adr = model.jnt_qposadr[root_jnt_id]
    dof_adr = model.jnt_dofadr[root_jnt_id]

    print("¡Iniciando el famoso baile Six Seven!")
    print("Abre otra terminal y ejecuta: python dance.py")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            t = step_start - start_time

            # 1. GRAVEDAD ACTIVADA
            # Como ya no forzamos su posición, el robot va a sentir la gravedad.
            # (¡Alerta de spoiler: al no tener un controlador de equilibrio, se va a caer de boca!)

            # 2. GENERAR LOS MOVIMIENTOS DEL "SIX SEVEN"
            # Movimiento de los hombros (arriba y abajo tipo aleteo)
            if 'left_shoulder_roll_joint' in ctrl_indices:
                data.ctrl[ctrl_indices['left_shoulder_roll_joint']] = math.sin(t * 4) * 1.5
            if 'right_shoulder_roll_joint' in ctrl_indices:
                data.ctrl[ctrl_indices['right_shoulder_roll_joint']] = math.sin(t * 4) * -1.5
            
            # Movimiento de brazos adelante y atrás
            if 'left_shoulder_pitch_joint' in ctrl_indices:
                data.ctrl[ctrl_indices['left_shoulder_pitch_joint']] = math.cos(t * 4) * 1.0
            if 'right_shoulder_pitch_joint' in ctrl_indices:
                data.ctrl[ctrl_indices['right_shoulder_pitch_joint']] = math.cos(t * 4 + math.pi) * 1.0
                
            # Apertura de piernas rítmica
            if 'left_hip_roll_joint' in ctrl_indices:
                data.ctrl[ctrl_indices['left_hip_roll_joint']] = math.sin(t * 8) * 0.3
            if 'right_hip_roll_joint' in ctrl_indices:
                data.ctrl[ctrl_indices['right_hip_roll_joint']] = math.sin(t * 8) * -0.3
                
            # Flexión de rodillas
            if 'left_knee_joint' in ctrl_indices:
                data.ctrl[ctrl_indices['left_knee_joint']] = abs(math.sin(t * 2)) * 1.5
            if 'right_knee_joint' in ctrl_indices:
                data.ctrl[ctrl_indices['right_knee_joint']] = abs(math.sin(t * 2 + math.pi)) * 1.5

            # Rotación de cadera (el toque mágico)
            if 'waist_yaw_joint' in ctrl_indices:
                data.ctrl[ctrl_indices['waist_yaw_joint']] = math.sin(t * 6) * 0.8

            # Avanzar la simulación
            mujoco.mj_step(model, data)
            viewer.sync()

            # Mantener la simulación a velocidad real
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()
