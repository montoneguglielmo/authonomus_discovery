import numpy as np


class pickAndLift():
    
    def __init__(action_shape):
        
        self.phase = "reach"
        self.kp_pos = 1.0 # Position gain
        self.kp_gripper = 1.0 # Gripper gain
        self.action_shape = action_shape
        self.gripped_count = 0
        self.final_target_pos = 
    
    def next_action(obj_pos, eef_pos, sensors):
        
        action = np.zeros
        distance = np.linalg.norm(obj_pos, eef_pos)
        
        if self.phase == "reach":
            pos_error = obj_pos - eef_pos
            action[0:3] = kp_pos * pos_error
            # Keep gripper open wide
            action[-1] = -1.0
            
            # Check if reached position
            if distance < 0.05:  # Within 8cm
                self.phase = "grasp"

        elif self.phase == "grasp":
            # Close the gripper
            action[-1] = 1.0
            self.gripper_count += 1
            
            if self.gripper_count > 20:
                self.phase = "lift"
                
        elif self.phase == "lift":
            pos_error = self.final_target_pos - eef_pos
            action[0:3] = self.kp_pos * pos_error
            action[-1] = 1.0  # Keep gripper close
        return action
            
