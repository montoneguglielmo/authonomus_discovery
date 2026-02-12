"""
Utility functions for creating and managing robosuite environments.
"""
import numpy as np
import random
import robosuite as suite
from robosuite.models.objects import (
    BoxObject, CylinderObject, BallObject,
    CapsuleObject
)
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite import load_composite_controller_config
from custom_env import RandomObjectsEnv
import math


# ========= Quaternion to Axis-Angle =========
def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den



def create_random_objects(
    min_objects=1,
    max_objects=5,
    ):
    """
    Creates a collection of random robosuite objects and a placement sampler for positioning them on a table.

    This function generates a random number of objects (between min_objects and max_objects) with
    randomized properties including shape, size, color, and density. Objects are selected from
    four types: boxes, cylinders, balls, and capsules. Each object is assigned random physical
    properties suitable for physics simulation.

    Args:
        min_objects (int, optional): Minimum number of objects to create. Defaults to 1.
        max_objects (int, optional): Maximum number of objects to create. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - selected_objects (list): List of robosuite MujocoObject instances with randomized
              properties (size, RGBA color, density between 500-3000 kg/m³).
            - placement_initializer (UniformRandomSampler): Sampler configured to place objects
              randomly on a table surface (0.5m x 0.5m area centered at origin, height 0.8m).

    Note:
        Object sizes are randomized within physically realistic ranges:
        - Boxes: 0.02-0.05m (width/depth), 0.02-0.08m (height)
        - Cylinders: 0.015-0.03m (radius), 0.04-0.08m (height)
        - Balls: 0.015-0.035m (radius)
        - Capsules: 0.015-0.025m (radius), 0.04-0.08m (height)
    """

    # Randomly select number of objects
    num_objects = random.randint(min_objects, max_objects)

    # Available object types with their creation functions
    object_types = [
        ("box", lambda name: BoxObject(
            name=name,
            size=np.random.uniform([0.02, 0.02, 0.02], [0.05, 0.05, 0.08]),
            rgba=np.random.uniform([0.2, 0.2, 0.2, 1], [0.9, 0.9, 0.9, 1]),
            density=np.random.uniform(500, 3000)  # Random density (kg/m³)
        )),
        ("cylinder", lambda name: CylinderObject(
            name=name,
            size=[np.random.uniform(0.015, 0.03), np.random.uniform(0.04, 0.08)],
            rgba=np.random.uniform([0.2, 0.2, 0.2, 1], [0.9, 0.9, 0.9, 1]),
            density=np.random.uniform(500, 3000)
        )),
        ("ball", lambda name: BallObject(
            name=name,
            size=[np.random.uniform(0.015, 0.035)],
            rgba=np.random.uniform([0.2, 0.2, 0.2, 1], [0.9, 0.9, 0.9, 1]),
            density=np.random.uniform(500, 3000)
        )),
        ("capsule", lambda name: CapsuleObject(
            name=name,
            size=[np.random.uniform(0.015, 0.025), np.random.uniform(0.04, 0.08)],
            rgba=np.random.uniform([0.2, 0.2, 0.2, 1], [0.9, 0.9, 0.9, 1]),
            density=np.random.uniform(500, 3000)
        )),
    ]

    # Randomly select objects
    selected_objects = []
    object_names = []

    for i in range(num_objects):
        obj_type, create_fn = random.choice(object_types)
        obj_name = f"{obj_type}_{i}"
        obj = create_fn(obj_name)
        selected_objects.append(obj)
        object_names.append(obj_name)

    # Define table bounds for object placement (in meters)
    # Table is typically centered at origin with dimensions ~0.8m x 0.8m
    table_x_range = [-0.25, 0.25]  # Left-right on table
    table_y_range = [-0.25, 0.25]  # Front-back on table
    table_z = 0.8  # Table surface height

    # Create placement sampler for random positions on table
    placement_initializer = UniformRandomSampler(
        name="ObjectSampler",
        mujoco_objects=selected_objects,
        x_range=table_x_range,
        y_range=table_y_range,
        rotation=None,  # Random rotation
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=np.array([0, 0, table_z]),
        z_offset=0.0,  # No offset - place directly on table
    )

    return selected_objects, placement_initializer

    


if __name__ == '__main__':
    import imageio


    selected_objects, placement_initializer = create_random_objects()

    # Load default controller config for the robot
    controller_config = load_composite_controller_config(controller=None, robot="Panda")

    # Create custom environment with our objects
    env = RandomObjectsEnv(
        robots="Panda",
        custom_objects=selected_objects,  # Pass our custom objects!
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="frontview",
        camera_heights=512,
        camera_widths=512,
        placement_initializer=placement_initializer,
        controller_configs=controller_config,
    )

    # Reset environment
    print("\nResetting environment...")
    obs = env.reset()

    # Let objects settle on the table (physics settling)
    print("Letting objects settle on table...")
    for _ in range(20):
        zero_action = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(zero_action)

    print("\nObject positions (world frame):")
    for obj in selected_objects:
        body_id = env.sim.model.body_name2id(obj.root_body)
        pos = env.sim.data.body_xpos[body_id]
        print(f"{obj.name}: {pos}")

    # Run random actions for a few steps
    print("\nRunning robot with random actions...")
    frames = []
    num_steps = 200

    for i in range(num_steps):
        # Generate random action
        action = np.random.randn(*env.action_spec[0].shape) * 0.05
        obs, reward, done, info = env.step(action)

        # Grab camera frame if available
        if "frontview_image" in obs:
            frame = obs["frontview_image"]
            frame = np.flip(frame, axis=0)
            frames.append(frame)

        # Print progress every 50 steps
        if (i + 1) % 50 == 0:
            print(f"  Step {i + 1}/{num_steps}")

    env.close()
    print("\nEnvironment closed.")

    # Save video if frames were captured
    if frames:
        output_file = "random_env_test.mp4"
        imageio.mimsave(output_file, frames, fps=20)
        print(f"\n✅ Test complete! Video saved to {output_file}")
    else:
        print("\n✅ Test complete! (No video frames captured)")
