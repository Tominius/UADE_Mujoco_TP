import json
import numpy as np
from pathlib import Path
import mujoco
import onnxruntime as ort

ch_dir = Path("g1-manipulation-challenge")
with open(ch_dir / "model_config.json") as f:
    config = json.load(f)
joint_names = config["joint_names"]
num_joints = len(joint_names)
default_pos = np.zeros(num_joints, dtype=np.float32)
for name, val in config["default_joint_pos"].items():
    if name in joint_names:
        default_pos[joint_names.index(name)] = val

action_scales = np.array([config["action_scales"][n] for n in joint_names], dtype=np.float32)

def set_armature(model, joint_names):
    ARM_5020 = 0.00360972
    ARM_7520_14 = 0.01017752
    ARM_7520_22 = 0.02510192
    ARM_4010 = 0.00425000
    ARM_2x5020 = 0.00721945
    for i, name in enumerate(joint_names):
        dof = 6 + i
        if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
            model.dof_armature[dof] = ARM_5020
        elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            model.dof_armature[dof] = ARM_7520_14
        elif "hip_roll" in name or "knee" in name:
            model.dof_armature[dof] = ARM_7520_22
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            model.dof_armature[dof] = ARM_4010
        elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
            model.dof_armature[dof] = ARM_2x5020
        else:
            model.dof_armature[dof] = ARM_5020

def quat_apply_inverse(quat, vec):
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)

model = mujoco.MjModel.from_xml_path(str(ch_dir / "scene.xml"))
model.opt.timestep = 0.005
set_armature(model, joint_names)
data = mujoco.MjData(model)

data.qpos[0] = -0.6
data.qpos[2] = 0.76
data.qpos[3:7] = [1, 0, 0, 0]
for name, val in config["default_joint_pos"].items():
    if name in joint_names:
        data.qpos[7 + joint_names.index(name)] = val
mujoco.mj_forward(model, data)

walker = ort.InferenceSession(str(ch_dir / "walker.onnx"), providers=["CPUExecutionProvider"])
in_name = walker.get_inputs()[0].name
out_name = walker.get_outputs()[0].name

last_action = np.zeros(num_joints, dtype=np.float32)
qpos_idx = {n: 7 + i for i, n in enumerate(joint_names)}
qvel_idx = {n: 6 + i for i, n in enumerate(joint_names)}

def get_obs():
    quat = data.qpos[3:7].copy()
    lin_vel_world = data.qvel[:3].copy()
    ang_vel_body = data.qvel[3:6].copy()
    lin_vel_body = quat_apply_inverse(quat, lin_vel_world)
    proj_grav = quat_apply_inverse(quat, np.array([0, 0, -1.0]))
    jpos = np.array([data.qpos[qpos_idx[n]] - default_pos[i] for i, n in enumerate(joint_names)], dtype=np.float32)
    jvel = np.array([data.qvel[qvel_idx[n]] for n in joint_names], dtype=np.float32)
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([lin_vel_body, ang_vel_body, proj_grav, jpos, jvel, last_action, cmd]).astype(np.float32)

torso_id = 16 # torso_link_rev_1_0
pelvis_id = 1

# Warmup 2 seconds (400 steps)
target_pos = default_pos.copy()
for step in range(400):
    if step % 4 == 0:
        obs = get_obs()
        action = walker.run([out_name], {in_name: obs.reshape(1, -1)})[0][0]
        target_pos = default_pos + action * action_scales
        last_action[:] = action
    for i, name in enumerate(joint_names):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id >= 0:
            data.ctrl[act_id] = target_pos[i]
    mujoco.mj_step(model, data)

print(f"Warmup done, G1 height: {data.qpos[2]:.3f}")

# Now test pushes!
pushes = [
    ("Empujon Frontal (+X)", np.array([60.0, 0.0, 0.0, 0, 0, 0])),
    ("Empujon Trasero (-X)", np.array([-60.0, 0.0, 0.0, 0, 0, 0])),
    ("Empujon Lateral Izq (+Y)", np.array([0.0, 45.0, 0.0, 0, 0, 0])),
    ("Empujon Lateral Der (-Y)", np.array([0.0, -45.0, 0.0, 0, 0, 0])),
]

for push_name, force_vec in pushes:
    print(f"\nProbando: {push_name}...")
    # Apply force for 0.15 seconds (30 steps)
    for step in range(30):
        data.xfrc_applied[torso_id] = force_vec
        if step % 4 == 0:
            obs = get_obs()
            action = walker.run([out_name], {in_name: obs.reshape(1, -1)})[0][0]
            target_pos = default_pos + action * action_scales
            last_action[:] = action
        for i, name in enumerate(joint_names):
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id >= 0:
                data.ctrl[act_id] = target_pos[i]
        mujoco.mj_step(model, data)
    
    # Remove force
    data.xfrc_applied[torso_id] = np.zeros(6)
    
    # Let it recover for 1.5 seconds (300 steps)
    fell = False
    for step in range(300):
        if step % 4 == 0:
            obs = get_obs()
            action = walker.run([out_name], {in_name: obs.reshape(1, -1)})[0][0]
            target_pos = default_pos + action * action_scales
            last_action[:] = action
        for i, name in enumerate(joint_names):
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id >= 0:
                data.ctrl[act_id] = target_pos[i]
        mujoco.mj_step(model, data)
        if data.qpos[2] < 0.5:
            fell = True
            break
            
    if fell:
        print(f"FALLO: El robot cayo con {push_name}")
    else:
        print(f"EXITO: {push_name} absorbido! Altura final: {data.qpos[2]:.3f}")
