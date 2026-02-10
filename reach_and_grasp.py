import numpy as np
import robosuite as suite
import imageio


# Create environment with OSC_POSE controller for precise end-effector control
env = suite.make(
    env_name="Lift",
    robots="Jaco",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="frontview",
    camera_heights=512,
    camera_widths=512,
    horizon=500,
    control_freq=20,
)

obs = env.reset()
frames = []

# Storage for sensor data analysis
sensor_log = {
    'step': [],
    'phase': [],
    'contact': [],
    'force_mag': [],
    'torque_mag': [],
    'force_x': [], 'force_y': [], 'force_z': [],
    'torque_x': [], 'torque_y': [], 'torque_z': [],
    'gripper_width': [],
    'cube_height': [],
}

# Get initial positions
cube_pos = obs["cube_pos"]
eef_pos = obs["robot0_eef_pos"]

print(f"Initial cube position: {cube_pos}")
print(f"Initial end-effector position: {eef_pos}")
print(f"Distance to cube: {np.linalg.norm(cube_pos - eef_pos):.3f}m\n")

# Control gains for smooth movement
kp_pos = 1.0  # Position gain
kp_gripper = 1.0  # Gripper gain

# Phases of the grasp
phase = "reach"  # reach -> descend -> grasp -> lift -> done
grasp_counter = 0
lift_counter = 0

# Track first contact for snapshot
first_contact_detected = False
first_contact_frame = None
last_no_contact_frame = None

for i in range(400):
    # Get current state
    cube_pos = obs["cube_pos"]
    eef_pos = obs["robot0_eef_pos"]
    gripper_to_cube = obs["gripper_to_cube_pos"]
    distance = np.linalg.norm(gripper_to_cube)

    # Get force-torque sensor data
    force_torque = env.sim.data.sensordata
    force = force_torque[0:3]
    torque = force_torque[3:6]
    force_mag = np.linalg.norm(force)

    # Initialize action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    # For OSC_POSE: first 3 are position deltas, next 3 are rotation, last is gripper
    action = np.zeros(env.action_spec[0].shape)

    if phase == "reach":
        # Move toward cube with offset (approach from above)
        target_offset = np.array([0.0, 0.0, 0.05])  # 5cm above cube
        target_pos = cube_pos + target_offset
        pos_error = target_pos - eef_pos

        # Position control (first 3 dimensions)
        action[0:3] = kp_pos * pos_error

        # Keep gripper open wide
        action[-1] = -1.0  # Open gripper

        # Check if reached position
        if distance < 0.08:  # Within 8cm
            phase = "descend"
            print(f"Step {i}: Above cube, descending to grasp")

    elif phase == "descend":
        # Descend BELOW cube center so gripper fingers are at cube height
        target_offset = np.array([0.0, 0.0, -0.04])  # 4cm BELOW cube center
        target_pos = cube_pos + target_offset
        pos_error = target_pos - eef_pos

        action[0:3] = kp_pos * pos_error
        action[-1] = -1.0  # Keep gripper open

        grasp_counter += 1

        # Once descended low enough, start closing gripper
        if grasp_counter > 50 and distance < 0.055:
            phase = "grasp"
            grasp_counter = 0
            print(f"Step {i}: Gripper positioned at height {eef_pos[2]:.3f}m (cube at {cube_pos[2]:.3f}m), closing")

    elif phase == "grasp":
        # Hold position and close gripper - DON'T MOVE YET!
        target_offset = np.array([0.0, 0.0, -0.04])  # Stay low
        target_pos = cube_pos + target_offset
        pos_error = target_pos - eef_pos

        action[0:3] = kp_pos * pos_error * 0.2  # Very gentle, just hold

        # Close gripper HARD
        action[-1] = 1.0  # Close gripper

        grasp_counter += 1

        # Wait for gripper to FULLY close - must be well into negative territory
        if gripper_width < -0.2 or grasp_counter > 150:  # Wait until VERY closed OR 7.5 seconds
            phase = "lift"
            print(f"Step {i}: Gripper tightly closed (width={gripper_width:.3f}), NOW lifting cube")

    elif phase == "lift":
        # Lift the cube up
        target_offset = np.array([0.0, 0.0, 0.15])  # 15cm above initial cube position
        target_pos = cube_pos + target_offset
        pos_error = target_pos - eef_pos

        action[0:3] = kp_pos * pos_error
        action[2] += 0.3  # Additional upward bias

        # Keep gripper closed
        action[-1] = 1.0

        lift_counter += 1

        if lift_counter > 100:  # Hold lifted for a while
            phase = "done"
            print(f"Step {i}: Grasp complete!")

    elif phase == "done":
        # Hold position
        action[0:3] = 0.0
        action[-1] = 1.0  # Keep gripper closed

    # Clip action to reasonable range
    action[0:3] = np.clip(action[0:3], -0.5, 0.5)

    # Step simulation
    obs, reward, done, info = env.step(action)

    # Get gripper state
    gripper_qpos = obs["robot0_gripper_qpos"]
    gripper_width = np.mean(gripper_qpos)  # Average gripper position

    # Check for gripper-cube contacts
    gripper_cube_contact = False
    for j in range(env.sim.data.ncon):
        contact = env.sim.data.contact[j]
        geom1 = env.sim.model.geom_id2name(contact.geom1)
        geom2 = env.sim.model.geom_id2name(contact.geom2)
        if ('gripper' in geom1.lower() and 'cube' in geom2.lower()) or \
           ('cube' in geom1.lower() and 'gripper' in geom2.lower()):
            gripper_cube_contact = True
            break

    # Capture first contact moment
    if not first_contact_detected and gripper_cube_contact:
        first_contact_detected = True
        first_contact_frame = obs["frontview_image"].copy()
        first_contact_step = i
        print(f"\n⚡ FIRST CONTACT DETECTED at step {i}!")
        print(f"   Force: {force_mag:.2f}N, Torque: {np.linalg.norm(torque):.3f}Nm")
        print(f"   Gripper width: {gripper_width:.3f}\n")
    elif not gripper_cube_contact:
        # Keep updating the last no-contact frame
        last_no_contact_frame = obs["frontview_image"].copy()

    # Log sensor data
    sensor_log['step'].append(i)
    sensor_log['phase'].append(phase)
    sensor_log['contact'].append(1 if gripper_cube_contact else 0)
    sensor_log['force_mag'].append(force_mag)
    sensor_log['torque_mag'].append(np.linalg.norm(torque))
    sensor_log['force_x'].append(force[0])
    sensor_log['force_y'].append(force[1])
    sensor_log['force_z'].append(force[2])
    sensor_log['torque_x'].append(torque[0])
    sensor_log['torque_y'].append(torque[1])
    sensor_log['torque_z'].append(torque[2])
    sensor_log['gripper_width'].append(gripper_width)
    sensor_log['cube_height'].append(cube_pos[2])

    # Print status every 20 steps
    if i % 20 == 0:
        contact_str = "CONTACT" if gripper_cube_contact else "no-contact"
        print(f"Step {i:3d} | Phase: {phase:8s} | Dist: {distance:.3f}m | "
              f"Force: {force_mag:6.2f}N | Torque: {np.linalg.norm(torque):6.3f}Nm | "
              f"Gripper: {gripper_width:6.3f} | Cube Z: {cube_pos[2]:.3f}m | {contact_str}")

    # Grab rendered frame
    frame = obs["frontview_image"]
    frame = np.flip(frame, axis=0)
    frames.append(frame)

env.close()

# Save video
imageio.mimsave("reach_and_grasp.mp4", frames, fps=20)

# Save contact moment snapshots
if first_contact_frame is not None:
    # Flip frames for correct orientation (same as video)
    first_contact_img = np.flip(first_contact_frame, axis=0)
    imageio.imwrite("first_contact.png", first_contact_img)
    print("\n📸 Saved first contact snapshot to: first_contact.png")

if last_no_contact_frame is not None:
    last_no_contact_img = np.flip(last_no_contact_frame, axis=0)
    imageio.imwrite("before_contact.png", last_no_contact_img)
    print("📸 Saved before contact snapshot to: before_contact.png")

# Analyze sensor data changes
print("\n" + "="*80)
print("SENSOR DATA ANALYSIS")
print("="*80)

# Convert to numpy arrays for easier analysis
steps = np.array(sensor_log['step'])
contacts = np.array(sensor_log['contact'])
force_mags = np.array(sensor_log['force_mag'])
torque_mags = np.array(sensor_log['torque_mag'])
cube_heights = np.array(sensor_log['cube_height'])
gripper_widths = np.array(sensor_log['gripper_width'])

# Find key moments
first_contact_idx = np.where(contacts == 1)[0][0] if np.any(contacts) else -1
lift_start_idx = np.where(cube_heights > cube_heights[0] + 0.01)[0][0] if np.any(cube_heights > cube_heights[0] + 0.01) else -1

print(f"\n1. BEFORE CONTACT (Steps 0-{first_contact_idx if first_contact_idx > 0 else 'N/A'}):")
if first_contact_idx > 0:
    before_contact = slice(0, first_contact_idx)
    print(f"   Force:  {force_mags[before_contact].mean():.2f} ± {force_mags[before_contact].std():.2f} N")
    print(f"   Torque: {torque_mags[before_contact].mean():.3f} ± {torque_mags[before_contact].std():.3f} Nm")
    print(f"   Cube height: {cube_heights[before_contact].mean():.3f}m (stable)")

print(f"\n2. DURING CONTACT (Steps {first_contact_idx}-{lift_start_idx if lift_start_idx > 0 else 'end'}):")
if first_contact_idx > 0:
    if lift_start_idx > first_contact_idx:
        during_contact = slice(first_contact_idx, lift_start_idx)
    else:
        during_contact = slice(first_contact_idx, len(steps))

    print(f"   Force:  {force_mags[during_contact].mean():.2f} ± {force_mags[during_contact].std():.2f} N")
    print(f"   Torque: {torque_mags[during_contact].mean():.3f} ± {torque_mags[during_contact].std():.3f} Nm")
    print(f"   Gripper closed from {gripper_widths[first_contact_idx]:.3f} to {gripper_widths[during_contact][-1]:.3f}")
    print(f"   Contact duration: {len(steps[during_contact])} steps ({len(steps[during_contact])/20:.1f}s)")

print(f"\n3. DURING LIFT (Steps {lift_start_idx if lift_start_idx > 0 else 'N/A'}-end):")
if lift_start_idx > 0:
    during_lift = slice(lift_start_idx, len(steps))
    print(f"   Force:  {force_mags[during_lift].mean():.2f} ± {force_mags[during_lift].std():.2f} N")
    print(f"   Torque: {torque_mags[during_lift].mean():.3f} ± {torque_mags[during_lift].std():.3f} Nm")
    print(f"   Cube lifted from {cube_heights[lift_start_idx]:.3f}m to {cube_heights[-1]:.3f}m")
    print(f"   Lift height: {cube_heights[-1] - cube_heights[0]:.3f}m")

# Force components analysis
print(f"\n4. FORCE VECTOR CHANGES:")
print(f"   Before contact: Fz={np.array(sensor_log['force_z'][:first_contact_idx]).mean():.2f}N (gravity)")
if lift_start_idx > 0:
    print(f"   During lift:    Fz={np.array(sensor_log['force_z'][lift_start_idx:]).mean():.2f}N")
    force_change = np.array(sensor_log['force_z'][lift_start_idx:]).mean() - np.array(sensor_log['force_z'][:first_contact_idx]).mean()
    print(f"   Force change: {force_change:+.2f}N (negative = reduced gravity load)")

# Contact statistics
contact_percent = (contacts.sum() / len(contacts)) * 100
print(f"\n5. CONTACT STATISTICS:")
print(f"   Total steps with contact: {contacts.sum()}/{len(contacts)} ({contact_percent:.1f}%)")
print(f"   First contact at step: {first_contact_idx}")
print(f"   Maintained contact: {'Yes ✓' if contacts[lift_start_idx:].all() else 'No ✗'}")

print("\n" + "="*80)
print(f"✅ Saved video to reach_and_grasp.mp4")
print(f"✅ Final cube height: {cube_heights[-1]:.3f}m (initial: {cube_heights[0]:.3f}m)")
print("="*80)
