"""
Factory functions for creating random robosuite objects and placement samplers.
"""

import math
import numpy as np
import random
from robosuite.models.objects import (
    BoxObject, CylinderObject, BallObject,
    CapsuleObject
)
from robosuite.utils.placement_samplers import UniformRandomSampler


def restitution_to_solref(restitution, mass):
    """Convert coefficient of restitution (0-1) to MuJoCo solref (negative/direct format).

    Uses damped spring model: negative solref values are interpreted by MuJoCo
    as direct -stiffness, -damping. The damping is derived from energy conservation:
        b = 2 * sqrt(k * m) * ln(e) / pi
    """
    K = np.random.uniform(2e3, 2e5)
    zeta = np.random.uniform(0.05, 1.5)
    B = 2 * zeta * np.sqrt(K * mass)
    return [-K, -B]


def random_contact_params(density, volume):
    """Generate random friction, solref (from restitution), and solimp parameters."""
    friction = [
        np.random.uniform(0.3, 2.0),    # sliding friction
        np.random.uniform(0.001, 0.02),  # torsional friction
        np.random.uniform(0.0001, 0.002) # rolling friction
    ]
    restitution = np.random.uniform(0.0, 0.9)  # coefficient of restitution
    mass = density * volume
    solref = restitution_to_solref(restitution, mass)
    solimp = [
        np.random.uniform(0.7, 0.95),    # dmin
        np.random.uniform(0.95, 0.9999),   # dmax
        np.random.uniform(0.0005, 0.02)   # width
    ]
    return friction, solref, solimp


def make_box(name):
    """Create a BoxObject with randomized size, density, and contact parameters."""
    size = np.random.uniform([0.02, 0.02, 0.02], [0.05, 0.05, 0.08])
    density = np.random.uniform(500, 3000)
    volume = 8.0 * size[0] * size[1] * size[2]  # box volume = (2w)(2d)(2h)
    friction, solref, solimp = random_contact_params(density, volume)
    return BoxObject(
        name=name, size=size,
        rgba=np.random.uniform([0.2, 0.2, 0.2, 1], [0.9, 0.9, 0.9, 1]),
        density=density, friction=friction, solref=solref, solimp=solimp,
    )


def make_cylinder(name):
    """Create a CylinderObject with randomized size, density, and contact parameters."""
    radius = np.random.uniform(0.015, 0.03)
    half_height = np.random.uniform(0.04, 0.08)
    density = np.random.uniform(500, 3000)
    volume = math.pi * radius ** 2 * (2.0 * half_height)
    friction, solref, solimp = random_contact_params(density, volume)
    return CylinderObject(
        name=name, size=[radius, half_height],
        rgba=np.random.uniform([0.2, 0.2, 0.2, 1], [0.9, 0.9, 0.9, 1]),
        density=density, friction=friction, solref=solref, solimp=solimp,
    )


def make_ball(name):
    """Create a BallObject with randomized size, density, and contact parameters."""
    radius = np.random.uniform(0.015, 0.035)
    density = np.random.uniform(500, 3000)
    volume = (4.0 / 3.0) * math.pi * radius ** 3
    friction, solref, solimp = random_contact_params(density, volume)
    return BallObject(
        name=name, size=[radius],
        rgba=np.random.uniform([0.2, 0.2, 0.2, 1], [0.9, 0.9, 0.9, 1]),
        density=density, friction=friction, solref=solref, solimp=solimp,
    )


def make_capsule(name):
    """Create a CapsuleObject with randomized size, density, and contact parameters."""
    radius = np.random.uniform(0.015, 0.025)
    half_height = np.random.uniform(0.04, 0.08)
    density = np.random.uniform(500, 3000)
    # capsule = cylinder + two hemispheres
    volume = math.pi * radius ** 2 * (2.0 * half_height) + (4.0 / 3.0) * math.pi * radius ** 3
    friction, solref, solimp = random_contact_params(density, volume)
    return CapsuleObject(
        name=name, size=[radius, half_height],
        rgba=np.random.uniform([0.2, 0.2, 0.2, 1], [0.9, 0.9, 0.9, 1]),
        density=density, friction=friction, solref=solref, solimp=solimp,
    )


def extract_object_metadata(obj):
    """Extract physical properties from a robosuite object as a flat dict.

    The object shape is parsed from the object name (e.g. "box_0" -> "box").
    """
    shape = obj.name.rsplit("_", 1)[0]
    return {
        "object_shape": shape,
        "object_size": list(obj.size),
        "object_density": float(obj.density),
        "object_friction": list(obj.friction),
        "object_rgba": list(obj.rgba),
        "object_solref": list(obj.solref),
        "object_solimp": list(obj.solimp),
    }


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
        ("box", make_box),
        ("cylinder", make_cylinder),
        ("ball", make_ball),
        ("capsule", make_capsule),
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
