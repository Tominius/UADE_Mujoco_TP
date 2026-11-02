import numpy as np
import mujoco
from pathlib import Path
import json

VISER_REPO = Path("Unitree-G1MuJoCo-Viser")
csv_path = VISER_REPO / "g1_moves/g1_moves_outputs/web_csv/dance_B_DadDance_web.csv"
trajectory = np.loadtxt(csv_path, delimiter=",")
model = mujoco.MjModel.from_xml_path(str(VISER_REPO / "assets/proto_g1/g1_holo_compat_web.xml"))
model.opt.timestep = 0.005
data = mujoco.MjData(model)

# Set initial state
data.qpos[:] = trajectory[0]
mujoco.mj_forward(model, data)

print("Testing PD...")
frame = 0
fps = 60.0

with open(VISER_REPO.parent / "g1-manipulation-challenge/model_config.json") as f:
    config = json.load(f)

for step in range(1000):
    current_time = step * model.opt.timestep
    frame = min(int(current_time * fps), len(trajectory) - 1)
    
    target_qpos = trajectory[frame]
    
    # Simple PD ctrl mapping
    for i, name in enumerate(config["joint_names"]):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        # qpos offset is 7
        data.ctrl[act_id] = target_qpos[7 + i]
        
    mujoco.mj_step(model, data)
    
    if data.qpos[2] < 0.4:
        print(f"Fell at step {step} (time {current_time:.2f}s)")
        break

if data.qpos[2] >= 0.4:
    print("Survived 1000 steps!")
