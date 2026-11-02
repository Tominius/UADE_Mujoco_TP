import mujoco
import numpy as np

model_path = "mujoco_menagerie/unitree_go2/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
model.opt.timestep = 0.002
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

kp = 90.0
kd = 5.0

# Base stand
base_q = np.array([
    0.0, 0.8, -1.5,  # FL
    0.0, 0.8, -1.5,  # FR
    0.0, 0.8, -1.5,  # RL
    0.0, 0.8, -1.5   # RR
])

# Sit pose
sit_q = np.array([
    0.0, 0.4, -0.9,  # FL (front high)
    0.0, 0.4, -0.9,  # FR (front high)
    0.0, 1.2, -2.4,  # RL (back low)
    0.0, 1.2, -2.4   # RR (back low)
])

# Test transition to sit
print("Testing Sit...")
for step in range(1500):
    t = step * model.opt.timestep
    alpha = min(1.0, t / 1.5)
    target = (1 - alpha) * base_q + alpha * sit_q
    
    q_cur = data.qpos[7:19]
    v_cur = data.qvel[6:18]
    torques = kp * (target - q_cur) - kd * v_cur
    torques = np.clip(torques, -35.0, 35.0)
    
    data.ctrl[:] = torques
    mujoco.mj_step(model, data)

print(f"Sit completed: z={data.qpos[2]:.3f}, pitch/roll quat={data.qpos[3:7]}")
