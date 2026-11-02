import mujoco
import numpy as np
import math

model_path = "mujoco_menagerie/unitree_go2/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
model.opt.timestep = 0.002
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

kp = 80.0
kd = 4.0

# Base joint positions for standing:
# 4 legs: [FL, FR, RL, RR], each [hip, thigh, calf]
# FL: [0, 1, 2], FR: [3, 4, 5], RL: [6, 7, 8], RR: [9, 10, 11]
# Standard stand: thigh ~ 0.8, calf ~ -1.5, hip ~ 0
base_q = np.array([
    0.0, 0.8, -1.5,  # FL
    0.0, 0.8, -1.5,  # FR
    0.0, 0.8, -1.5,  # RL
    0.0, 0.8, -1.5   # RR
])

print("Testing Trot Gait with PD...")

fell = False
for step in range(2500): # 5 seconds
    t = step * model.opt.timestep
    
    # Warmup 1 sec
    target = base_q.copy()
    if t > 1.0:
        walk_t = t - 1.0
        freq = 2.0 * math.pi * 1.8 # 1.8 Hz trot
        phase_A = walk_t * freq             # FL + RR
        phase_B = walk_t * freq + math.pi   # FR + RL
        
        amp_hip = 0.25
        amp_knee = 0.35
        
        # Leg FL (0,1,2) - Phase A
        target[1] = base_q[1] + math.sin(phase_A) * amp_hip
        target[2] = base_q[2] + max(0.0, math.cos(phase_A)) * amp_knee
        
        # Leg RR (9,10,11) - Phase A
        target[10] = base_q[10] + math.sin(phase_A) * amp_hip
        target[11] = base_q[11] + max(0.0, math.cos(phase_A)) * amp_knee
        
        # Leg FR (3,4,5) - Phase B
        target[4] = base_q[4] + math.sin(phase_B) * amp_hip
        target[5] = base_q[5] + max(0.0, math.cos(phase_B)) * amp_knee
        
        # Leg RL (6,7,8) - Phase B
        target[7] = base_q[7] + math.sin(phase_B) * amp_hip
        target[8] = base_q[8] + max(0.0, math.cos(phase_B)) * amp_knee
        
    q_cur = data.qpos[7:19]
    v_cur = data.qvel[6:18]
    torques = kp * (target - q_cur) - kd * v_cur
    torques = np.clip(torques, -30.0, 30.0)
    
    data.ctrl[:] = torques
    mujoco.mj_step(model, data)
    
    if data.qpos[2] < 0.15:
        print(f"Fell at t={t:.2f}s, z={data.qpos[2]:.3f}")
        fell = True
        break

if not fell:
    print(f"Survived trot! Final x={data.qpos[0]:.3f}, y={data.qpos[1]:.3f}, z={data.qpos[2]:.3f}")
