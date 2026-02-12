import numpy as np
import robosuite as suite
import imageio
from utils import create_random_tabletop_env


# create environment instance (SERVER SAFE)
env = suite.make(
    env_name="Lift",
    robots="Jaco",
    has_renderer=False,              # ❌ no window
    has_offscreen_renderer=True,     # ✅ offscreen rendering
    use_camera_obs=True,
    camera_names="frontview",
    camera_heights=512,
    camera_widths=512,
)

obs = env.reset()

frames = []

for i in range(300):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)

    # Access force-torque sensor data directly from MuJoCo simulation
    force_torque = env.sim.data.sensordata  # 6D: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
    force = force_torque[0:3]
    torque = force_torque[3:6]

    # Print sensor data every 50 steps
    if i % 50 == 0:
        n_contacts = env.sim.data.ncon
        print(f"\nStep {i}:")
        print(f"  Force (N):  [{force[0]:7.3f}, {force[1]:7.3f}, {force[2]:7.3f}] | mag={np.linalg.norm(force):7.3f}")
        print(f"  Torque (Nm):[{torque[0]:7.3f}, {torque[1]:7.3f}, {torque[2]:7.3f}] | mag={np.linalg.norm(torque):7.3f}")
        print(f"  Active contacts: {n_contacts}")

        # Show some contact details if any
        if n_contacts > 0:
            for j in range(min(2, n_contacts)):
                contact = env.sim.data.contact[j]
                geom1 = env.sim.model.geom_id2name(contact.geom1)
                geom2 = env.sim.model.geom_id2name(contact.geom2)
                if 'gripper' in geom1.lower() or 'gripper' in geom2.lower():
                    print(f"    → Gripper contact: {geom1} <-> {geom2}")

    # grab rendered frame
    frame = obs["frontview_image"]
    frame = np.flip(frame, axis=0)
    frames.append(frame)

env.close()

# save video
imageio.mimsave("lift_hello_world.mp4", frames, fps=20)

print("Saved video to lift_hello_world.mp4 🎉")