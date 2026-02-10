"""
Custom robosuite environment that supports random objects on a table.
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor


class RandomObjectsEnv(ManipulationEnv):
    """
    Custom environment with random objects on a table.

    This environment allows you to place custom objects on a table,
    unlike predefined environments (Stack, Lift) which have hardcoded objects.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(1.5, 1.5, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        custom_objects=None,
    ):
        """
        Args:
            custom_objects (list): List of MujocoObject instances to place in the environment
            All other args are standard robosuite ManipulationEnv arguments
        """
        # Store custom objects and placement initializer BEFORE super().__init__()
        # because _load_model() and _setup_observables() are called during parent init
        self.custom_objects = custom_objects if custom_objects is not None else []
        self.placement_initializer = placement_initializer
        self.use_object_obs = use_object_obs

        # Store table specifications
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Simple reward function - can be customized.
        For now, just return 0 (exploration environment).
        """
        return 0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Load arena (table)
        # table_offset z-value should be the height where the table top sits
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=(0, 0, 0.8),  # Standard table height
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Initialize placement sampler with custom objects
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.custom_objects)

        # Task: combine arena, robot, and objects into one model
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.custom_objects,  # Use our custom objects!
        )

    def _setup_references(self):
        """
        Sets up references to important components.
        """
        super()._setup_references()

        # Set references to custom objects
        for i, obj in enumerate(self.custom_objects):
            setattr(self, f"obj_{i}", self.sim.data.get_body_xpos(obj.root_body))

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset placement sampler
        if self.placement_initializer is not None:
            # Sample object placements
            object_placements = self.placement_initializer.sample()

            # Place objects
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.
        """
        observables = super()._setup_observables()

        # Low-level object information
        if self.use_object_obs:
            # Get robot prefix
            pf = self.robots[0].robot_model.naming_prefix

            # Add observables for each object
            for i, obj in enumerate(self.custom_objects):
                obj_name = f"object{i}"

                # Object position
                @sensor(modality=f"{obj_name}_pos")
                def obj_pos(obs_cache, obj_body=obj.root_body):
                    return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(obj_body)])

                # Object orientation
                @sensor(modality=f"{obj_name}_quat")
                def obj_quat(obs_cache, obj_body=obj.root_body):
                    return np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(obj_body)])

                sensors = [obj_pos, obj_quat]
                names = [f"{obj_name}_pos", f"{obj_name}_quat"]

                # Create observables
                for name, s in zip(names, sensors):
                    observables[name] = Observable(
                        name=name,
                        sensor=s,
                        sampling_rate=self.control_freq,
                    )

        return observables

    def _check_success(self):
        """
        Check if task is successfully completed.
        For now, always return False (exploration environment).
        """
        return False

    def visualize(self, vis_settings):
        """
        Do any needed visualization here
        """
        # Do superclass visualizations
        super().visualize(vis_settings=vis_settings)
