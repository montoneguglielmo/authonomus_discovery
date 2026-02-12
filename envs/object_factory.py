"""
Factory functions for creating random robosuite objects and placement samplers.
"""

import numpy as np
import random
from robosuite.models.objects import (
    BoxObject, CylinderObject, BallObject,
    CapsuleObject
)
from robosuite.utils.placement_samplers import UniformRandomSampler


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
