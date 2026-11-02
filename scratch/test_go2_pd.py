import mujoco
import numpy as np

model_path = "mujoco_menagerie/unitree_go2/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Reset to keyframe home
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print(f"Initial z: {data.qpos[2]:.3f}")

kp = 60.0
kd = 5.0
target_qpos = data.qpos[7:19].copy()

# Run simulation for 2 seconds (1000 steps with dt=0.002)
for step in range(1000):
    # PD controller for all 12 motors
    # Joint values in qpos[7:19], qvel in qvel[6:18]
    q_cur = data.qpos[7:19]
    v_cur = data.qvel[6:18]
    
    torques = kp * (target_qpos - q_cur) - kd * v_cur
    # Clamp to motor limits
    torques = np.clip(torques, -23.7, 23.7)
    
    data.ctrl[:] = torques
    mujoco.mj_step(model, data)

print(f"Final z after 2s: {data.qpos[2]:.3f}")
if data.qpos[2] > 0.25:
    print("SUCCESS: Go2 is standing strong!")
else:
    print("FAILED: Go2 collapsed.")
