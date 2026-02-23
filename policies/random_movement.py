import numpy as np


class RandomMovementOnTable():

        def __init__(self, action_shape):
            self.count_step = 0
            self.kp_pos = 1.0
            self.height = 0.7
            self.target_pos = np.array([0.0, 0.0, self.height])
            self.action_shape = action_shape

        def next_action(self, obs, env):
            eef_pos = obs["robot0_eef_pos"]

            action = np.zeros(self.action_shape)
            self.count_step += 1

            if np.mod(self.count_step, 100) == 0:
                self.target_pos = np.array([
                    np.random.uniform(-0.7, 0.7),
                    np.random.uniform(-0.7, 0.7),
                    self.height
                ])
                
                self.kp_pos = np.random.uniform(0.5, 2)

            pos_error = self.target_pos - eef_pos
            action[0:3] = self.kp_pos * pos_error

            return action
