import numpy as np
import random


class PickAndLiftV2():

    def __init__(self, action_shape, selected_objects):
        self.action_shape = action_shape
        self.hover_height = 0.05
        self.grasp_threshold = 0.02
        self.hover_threshold = 0.03
        self.lift_height = 0.25
        self.kp_fast = 2.0
        self.kp_slow = 1.0
        self.gripper_min_steps = 40
        self.gripper_fully_closed_threshold = 0.005
        self.gripper_count = 0
        self.phase = "hover"
        self.select_object(selected_objects)

    def next_action(self, obs, env):
        body_id = env.sim.model.body_name2id(self.obj_to_pick.root_body)
        obj_pos = env.sim.data.body_xpos[body_id]
        eef_pos = obs["robot0_eef_pos"]

        action = np.zeros(self.action_shape)

        if self.phase == "hover":
            # Move above the object first to avoid knocking it
            hover_target = obj_pos.copy()
            hover_target[2] += self.hover_height
            pos_error = hover_target - eef_pos
            action[0:3] = self.kp_fast * pos_error
            action[-1] = -1.0  # Gripper open

            xy_dist = np.linalg.norm(pos_error[:2])
            z_dist = abs(pos_error[2])
            if xy_dist < self.hover_threshold and z_dist < self.hover_threshold:
                self.phase = "descend"

        elif self.phase == "descend":
            # Descend to object level but keep XY aligned
            descend_target = obj_pos.copy()
            pos_error = descend_target - eef_pos
            action[0:3] = self.kp_slow * pos_error
            action[-1] = -1.0  # Gripper open

            distance = np.linalg.norm(pos_error)
            if distance < self.grasp_threshold:
                self.grasp_pos = eef_pos.copy()
                gripper_qpos = obs["robot0_gripper_qpos"]
                self.open_gripper_width = np.mean(np.abs(gripper_qpos))
                self.phase = "grasp"

        elif self.phase == "grasp":
            # Hold position and close gripper — don't chase the object
            pos_error = self.grasp_pos - eef_pos
            action[0:3] = self.kp_slow * pos_error
            action[-1] = 1.0
            self.gripper_count += 1

            gripper_qpos = obs["robot0_gripper_qpos"]
            gripper_width = np.mean(np.abs(gripper_qpos))
            gripper_not_fully_closed = gripper_width > self.gripper_fully_closed_threshold
            gripper_has_closed = gripper_width < self.open_gripper_width * 0.5
            has_contact = self._check_gripper_contact(env)

            if (self.gripper_count >= self.gripper_min_steps
                    and has_contact and gripper_not_fully_closed and gripper_has_closed):
                self.lift_target = eef_pos.copy()
                self.lift_target[2] += self.lift_height
                self.phase = "lift"

        elif self.phase == "lift":
            pos_error = self.lift_target - eef_pos
            action[0:3] = self.kp_fast * pos_error
            action[-1] = 1.0

        return action

    def _check_gripper_contact(self, env):
        """Check if any gripper geom is in contact with the target object."""
        obj_name = self.obj_to_pick.name
        for i in range(env.sim.data.ncon):
            contact = env.sim.data.contact[i]
            geom1 = env.sim.model.geom_id2name(contact.geom1)
            geom2 = env.sim.model.geom_id2name(contact.geom2)
            if geom1 is None or geom2 is None:
                continue
            gripper_touch = 'gripper' in geom1.lower() or 'gripper' in geom2.lower()
            object_touch = obj_name in geom1 or obj_name in geom2
            if gripper_touch and object_touch:
                return True
        return False

    def reset(self, selected_objects):
        """Reset state machine and pick a new target object."""
        self.phase = "hover"
        self.gripper_count = 0
        self.select_object(selected_objects)

    def select_object(self, object_list):
        obj = random.choice(object_list)
        self.obj_to_pick = obj
