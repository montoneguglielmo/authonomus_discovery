"""AlignedBoxGrasp policy.

Extends SensorAwarePickAndLift (v6) with gripper yaw alignment:
during the hover phase the gripper rotates so that its X-axis
(the index-finger ↔ pinky-finger axis) aligns with the longest
horizontal dimension of the box top face before descending.

For non-box objects (cylinders, balls, capsules) the yaw correction
is skipped and the policy behaves identically to v6.

State machine
─────────────
  align    Move to object XY at SAFE_Z; gripper open
  hover    Lower to obj + HOVER_HEIGHT; gripper open;
             ALSO rotate EEF yaw to align with box long axis
  descend  Lower to obj + DESCEND_OFFSET; gripper open
  close    Hold position, close gripper; monitor contact + object z
  lift     Move EEF upward; keep gripper closed
"""

import random
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects.primitive.box import BoxObject


# ── Constants ─────────────────────────────────────────────────────────────────

SAFE_Z         = 1.10    # m – EEF height during the align transit
HOVER_HEIGHT   = 0.07    # m above object centre
DESCEND_OFFSET = 0.0     # m above object centre for the close setpoint
MIN_PART_CLOSE = 0.70    # gripper must reach < 70 % of open width
OBJ_Z_RISE     = 0.005   # m – object z increase signals genuine lifting force
LIFT_HEIGHT    = 0.35    # m upward from grasp EEF position

YAW_THRESHOLD  = 0.05    # rad (~3°) – alignment considered done below this
KP_ROT         = 0.5     # proportional gain for yaw correction
ALIGN_RATIO    = 1.05    # skip alignment when hx/hy ratio < this (square face)


class AlignedBoxGrasp:
    """Pick-and-lift with gripper yaw alignment for box objects."""

    def __init__(self, action_shape, selected_objects):
        self.action_shape = action_shape

        # ── gains ──────────────────────────────────────────────────────────────
        self.kp_fast = 2.5
        self.kp_slow = 1.0
        self.kp_grip = 1.5

        # ── thresholds ─────────────────────────────────────────────────────────
        self.align_xy_thr      = 0.03
        self.align_z_thr       = 0.04
        self.hover_threshold   = 0.025
        self.descend_threshold = 0.022

        # ── timing ─────────────────────────────────────────────────────────────
        self.min_close_steps = 120
        self.max_close_steps = 240

        self._init_state(selected_objects)

    # ── public API ─────────────────────────────────────────────────────────────

    def next_action(self, obs, env):
        action  = np.zeros(self.action_shape)
        eef_pos = obs["robot0_eef_pos"]

        body_id = env.sim.model.body_name2id(self.obj_to_pick.root_body)
        obj_pos = env.sim.data.body_xpos[body_id].copy()

        has_contact = self._check_gripper_contact(env)

        if self.phase == "align":
            self._align(action, eef_pos, obj_pos)
        elif self.phase == "hover":
            self._hover(action, eef_pos, obj_pos, obs, env)
        elif self.phase == "descend":
            self._descend(action, eef_pos, obj_pos)
        elif self.phase == "close":
            self._close(action, eef_pos, obs, obj_pos, has_contact)
        elif self.phase == "lift":
            self._lift(action, eef_pos)

        return action

    def reset(self, selected_objects):
        self._init_state(selected_objects)

    # ── internals ──────────────────────────────────────────────────────────────

    def _init_state(self, selected_objects):
        self.phase       = "align"
        self.close_steps = 0
        self.grasp_pos   = None
        self.lift_target = None
        self.open_width  = None
        self.obj_z_init  = None
        self.select_object(selected_objects)

    def select_object(self, object_list):
        self.obj_to_pick = random.choice(object_list)

    def _check_gripper_contact(self, env):
        obj_name = self.obj_to_pick.name
        for i in range(env.sim.data.ncon):
            c  = env.sim.data.contact[i]
            g1 = env.sim.model.geom_id2name(c.geom1)
            g2 = env.sim.model.geom_id2name(c.geom2)
            if g1 is None or g2 is None:
                continue
            if (('gripper' in g1.lower() or 'gripper' in g2.lower())
                    and (obj_name in g1 or obj_name in g2)):
                return True
        return False

    def _compute_yaw_error(self, obs, env):
        """Return signed yaw error (rad) to align gripper X with box long axis.

        Returns 0.0 for non-box objects or when the top face is square.
        Positive = gripper needs to rotate CCW (world Z+) to align.
        The error is clamped to (-π/2, π/2] exploiting 180° symmetry
        (index and pinky are interchangeable).
        """
        if not isinstance(self.obj_to_pick, BoxObject):
            return 0.0

        half_sizes = np.array(self.obj_to_pick.size)   # [hx, hy, hz]
        hx, hy = half_sizes[0], half_sizes[1]

        # Square top face — no preferred orientation
        if min(hx, hy) < 1e-9:
            return 0.0
        if max(hx, hy) / min(hx, hy) < ALIGN_RATIO:
            return 0.0

        # Box rotation in world frame
        body_id = env.sim.model.body_name2id(self.obj_to_pick.root_body)
        mj_q    = env.sim.data.body_xquat[body_id].copy()          # [w,x,y,z]
        rs_q    = np.array([mj_q[1], mj_q[2], mj_q[3], mj_q[0]]) # [x,y,z,w]
        box_rot = T.quat2mat(rs_q)   # 3×3 rotation matrix

        # Longest horizontal axis of the box (in world XY)
        target_3d = box_rot[:, 0] if hx >= hy else box_rot[:, 1]
        target = target_3d[:2].copy()
        t_norm = np.linalg.norm(target)
        if t_norm < 1e-6:
            return 0.0
        target /= t_norm

        # Current gripper X-axis (index→pinky direction) in world XY
        eef_rot = T.quat2mat(obs["robot0_eef_quat"])   # [x,y,z,w] robosuite convention
        current_3d = eef_rot[:, 0]
        current = current_3d[:2].copy()
        c_norm = np.linalg.norm(current)
        if c_norm < 1e-6:
            return 0.0
        current /= c_norm

        # Signed angle from current to target
        cross = current[0] * target[1] - current[1] * target[0]
        dot   = np.dot(current, target)
        angle = np.arctan2(cross, dot)   # ∈ (-π, π]

        # Exploit 180° symmetry: index and pinky are symmetric,
        # so rotating by π gives an equivalent grip.
        if abs(angle) > np.pi / 2:
            angle -= np.sign(angle) * np.pi

        return angle   # ∈ (-π/2, π/2]

    # ── phase methods ──────────────────────────────────────────────────────────

    def _align(self, action, eef_pos, obj_pos):
        """Transit to object XY at SAFE_Z; gripper fully open."""
        target      = np.array([obj_pos[0], obj_pos[1], SAFE_Z])
        pe          = target - eef_pos
        action[0:3] = self.kp_fast * pe
        action[-1]  = -1.0  # open

        if np.linalg.norm(pe[:2]) < self.align_xy_thr and abs(pe[2]) < self.align_z_thr:
            self.phase = "hover"

    def _hover(self, action, eef_pos, obj_pos, obs, env):
        """Lower to HOVER_HEIGHT above object; align yaw; gripper open.

        The phase only exits when both the position error and the yaw error
        are within their respective thresholds.
        """
        target      = obj_pos.copy()
        target[2]  += HOVER_HEIGHT
        pe          = target - eef_pos
        action[0:3] = self.kp_fast * pe
        action[-1]  = -1.0  # open

        # Yaw alignment: action[5] is the world-Z (yaw) delta
        yaw_err  = self._compute_yaw_error(obs, env)
        action[5] = KP_ROT * yaw_err

        if np.linalg.norm(pe) < self.hover_threshold and abs(yaw_err) < YAW_THRESHOLD:
            self.phase = "descend"

    def _descend(self, action, eef_pos, obj_pos):
        """Approach to DESCEND_OFFSET above centre; gripper open."""
        target      = obj_pos.copy()
        target[2]  += DESCEND_OFFSET
        pe          = target - eef_pos
        action[0:3] = self.kp_slow * pe
        action[-1]  = -1.0  # open

        if np.linalg.norm(pe) < self.descend_threshold:
            self.grasp_pos   = eef_pos.copy()
            self.close_steps = 0
            self.open_width  = None
            self.obj_z_init  = obj_pos[2]
            self.phase       = "close"

    def _close(self, action, eef_pos, obs, obj_pos, has_contact):
        """Close the gripper gently; lift when contact is confirmed."""
        pe          = self.grasp_pos - eef_pos
        action[0:3] = self.kp_slow * pe
        action[-1]  = self.kp_grip
        self.close_steps += 1

        gw = float(np.mean(np.abs(obs["robot0_gripper_qpos"])))
        if self.open_width is None:
            self.open_width = max(gw, 1e-6)

        partially_closed = gw < self.open_width * MIN_PART_CLOSE

        # ── Signal 1: object being lifted by fingers (object z rises) ─────────
        if has_contact and self.obj_z_init is not None:
            if obj_pos[2] - self.obj_z_init > OBJ_Z_RISE:
                self._begin_lift(eef_pos)
                return

        # ── Signal 2: grip equilibrated with object in it ─────────────────────
        if self.close_steps >= self.min_close_steps:
            if has_contact and partially_closed:
                self._begin_lift(eef_pos)
                return

        # ── Signal 3: timed fallback — only lift if object is in contact ──────
        if self.close_steps >= self.max_close_steps:
            if has_contact:
                self._begin_lift(eef_pos)

    def _begin_lift(self, eef_pos):
        self.lift_target    = eef_pos.copy()
        self.lift_target[2] += LIFT_HEIGHT
        self.phase = "lift"

    def _lift(self, action, eef_pos):
        pe          = self.lift_target - eef_pos
        action[0:3] = self.kp_slow * pe  # gentle lift to keep grip intact
        action[-1]  = 1.0   # full close during lift
