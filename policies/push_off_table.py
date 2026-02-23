import numpy as np
import random


class PushOffTable:
    """Push a randomly selected object off the edge of the table.

    Phase 1 – approach: move the EEF to a position directly behind the object
               (opposite of the chosen push direction), at table-surface height.
    Phase 2 – push: drive the EEF straight through the object toward the table
               edge and beyond, knocking the object off.
    """

    TABLE_HALF_WIDTH = 0.4  # half of table_full_size (0.8, 0.8, 0.05)
    TABLE_HEIGHT = 0.8       # z of the table top surface
    PUSH_HEIGHT = 0.84       # EEF z during push (just above the surface)
    APPROACH_DIST = 0.15     # metres behind the object for the approach point
    APPROACH_TOL = 0.06      # metres — distance considered "arrived"
    EDGE_OVERSHOOT = 0.30    # how far past the edge to target (ensures full push)

    def __init__(self, action_shape, selected_objects):
        self.action_shape = action_shape
        self.kp_pos = 1.5
        self.phase = "approach"
        self._push_target = None

        self.select_object(selected_objects)

        # Pick one of the four cardinal push directions uniformly at random.
        angle = random.choice([0.0, 90.0, 180.0, 270.0])
        rad = np.radians(angle)
        self.push_dir = np.array([np.cos(rad), np.sin(rad), 0.0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_approach_target(self, obj_pos):
        """Return the EEF target for the approach phase."""
        target = obj_pos.copy()
        target -= self.push_dir * self.APPROACH_DIST
        target[2] = self.PUSH_HEIGHT
        return target

    def _compute_push_target(self, eef_pos):
        """Return the EEF target for the push phase (stored once at transition)."""
        target = eef_pos.copy()
        target[2] = self.PUSH_HEIGHT
        reach = self.TABLE_HALF_WIDTH + self.EDGE_OVERSHOOT
        if abs(self.push_dir[0]) > 0.5:   # pushing along X
            target[0] = np.sign(self.push_dir[0]) * reach
        else:                               # pushing along Y
            target[1] = np.sign(self.push_dir[1]) * reach
        return target

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def next_action(self, obs, env):
        body_id = env.sim.model.body_name2id(self.obj_to_push.root_body)
        obj_pos = env.sim.data.body_xpos[body_id].copy()
        eef_pos = obs["robot0_eef_pos"]

        action = np.zeros(self.action_shape)
        action[-1] = -1.0  # gripper open throughout (palm / fingers push the object)

        if self.phase == "approach":
            target = self._compute_approach_target(obj_pos)
            pos_error = target - eef_pos
            action[0:3] = self.kp_pos * pos_error

            if np.linalg.norm(pos_error) < self.APPROACH_TOL:
                # Store the push target once so it is fixed for the whole push.
                self._push_target = self._compute_push_target(eef_pos)
                self.phase = "push"

        elif self.phase == "push":
            pos_error = self._push_target - eef_pos
            action[0:3] = self.kp_pos * pos_error

        return action

    def select_object(self, object_list):
        self.obj_to_push = random.choice(object_list)
