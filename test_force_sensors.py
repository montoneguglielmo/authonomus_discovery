import numpy as np
import robosuite as suite

# Create environment
env = suite.make(
    env_name="Lift",
    robots="Jaco",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

obs = env.reset()

# Check if the robot has force-torque sensors
print("Robot information:")
print(f"  Robot name: {env.robots[0].name}")
print(f"  Robot model: {env.robots[0].robot_model.naming_prefix}")

# Check MuJoCo model for sensors
print("\nMuJoCo sensors available:")
sim = env.sim
if hasattr(sim.model, 'sensor_names'):
    for i, sensor_name in enumerate(sim.model.sensor_names):
        print(f"  {i}: {sensor_name}")
else:
    print("  No sensor_names attribute found")

# Check for force/torque data in MuJoCo
print(f"\nMuJoCo sensor data shape: {sim.data.sensordata.shape if hasattr(sim.data, 'sensordata') else 'N/A'}")

# Take a few steps and check for contact forces
print("\nTaking 10 steps and checking for contact information...")
for i in range(10):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)

# Check contact data
if hasattr(sim.data, 'contact'):
    n_contacts = sim.data.ncon
    print(f"\nNumber of active contacts: {n_contacts}")

    if n_contacts > 0:
        print("Contact details:")
        for i in range(min(5, n_contacts)):  # Show first 5 contacts
            contact = sim.data.contact[i]
            print(f"  Contact {i}:")
            print(f"    Geom1: {sim.model.geom_id2name(contact.geom1)}")
            print(f"    Geom2: {sim.model.geom_id2name(contact.geom2)}")
            print(f"    Distance: {contact.dist:.6f}")
            # Contact force is in contact.frame
            print(f"    Normal force: {np.linalg.norm(contact.frame[:3]):.6f}")

# Check for force-torque in sensor data
if hasattr(sim.data, 'sensordata') and len(sim.data.sensordata) > 0:
    print(f"\nForce-Torque Sensor data:")
    print(f"  Force (x,y,z):  {sim.data.sensordata[0:3]}")
    print(f"  Torque (x,y,z): {sim.data.sensordata[3:6]}")
    print(f"  Force magnitude: {np.linalg.norm(sim.data.sensordata[0:3]):.6f} N")
    print(f"  Torque magnitude: {np.linalg.norm(sim.data.sensordata[3:6]):.6f} Nm")

env.close()
print("\nSummary:")
print("✓ MuJoCo contact data is available for collision detection")
print("✓ You can check sim.data.ncon and sim.data.contact for touch detection")
