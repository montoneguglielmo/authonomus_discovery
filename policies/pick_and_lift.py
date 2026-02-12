import numpy as np
import random


class PickAndLift():

    def __init__(self, action_shape, selected_objects):

        self.phase = "reach"
        self.kp_pos = 1.0 # Position gain
        self.kp_gripper = 1.0 # Gripper gain
        self.action_shape = action_shape
        self.gripper_count = 0
        self.final_target_pos = np.array([0.0, 0.0, 1.0])
        self.select_object(selected_objects)


    def next_action(self, obs, env):

        body_id = env.sim.model.body_name2id(self.obj_to_pick.root_body)
        obj_pos = env.sim.data.body_xpos[body_id]

        eef_pos = obs["robot0_eef_pos"]

        action = np.zeros(self.action_shape)
        distance = np.linalg.norm(obj_pos - eef_pos)

        if self.phase == "reach":
            pos_error = obj_pos - eef_pos
            action[0:3] = self.kp_pos * pos_error
            # Keep gripper open wide
            action[-1] = -1.0

            # Check if reached position
            if distance < 0.05:  # Within 5cm
                self.phase = "grasp"

        elif self.phase == "grasp":
            # Close the gripper
            action[-1] = 1.0
            self.gripper_count += 1

            if self.gripper_count > 100:
                self.phase = "lift"

        elif self.phase == "lift":
            pos_error = self.final_target_pos - eef_pos
            action[0:3] = self.kp_pos * pos_error
            action[-1] = 1.0  # Keep gripper close
        return action

    def select_object(self, object_list):
        obj = random.choice(object_list)
        self.obj_to_pick = obj
