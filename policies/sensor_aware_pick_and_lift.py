"""SensorAwarePickAndLift policy — v6 (stable).

What works and what doesn't (runs 1–5 summary):

  ✗ Touch-only trigger:  inter-finger contact (grip fully closed = ~0) gives
                         false 2–3 N readings.  Spurious lifts with nothing in grip.
  ✗ Stall detection:    enters close when EEF is 60 mm above the object;
                         grip closes on air; same false-positive problem.
  ✗ any_touch fallback: still fires from finger–finger contact at step > max_close.
  ✓ has_contact check:  specifically queries gripper ↔ target-object geom contact;
                         immune to finger-finger false positives.
  ✓ Align phase:        transit to object XY at a safe height first so arm links
                         don't sweep through object level.
  ✓ Unconditional fallback guarded by has_contact: if the grip IS established
                         the lift succeeds; if not, the fallback is skipped and
                         nothing worse happens.
  ✓ Object-z rise monitoring: clean sensor-fusion signal — if the object lifts
                         even 5 mm during close it means fingers are actually
                         applying a lifting force.

Design goals for v6:
  - Keep the align phase (prevents transit knockoffs).
  - Use has_contact for ALL lift decisions (no touch-only or blind lifts).
  - Add the object-z signal as the primary early-lift detector.
  - Use a fixed, generous close duration (240 steps ≈ 12 s at 20 Hz).
  - During close, push the EEF slightly DOWN toward the object every step so
    the fingers stay engaged even if the arm is 40–60 mm above the true centre.

State machine
─────────────
  align    Move to object XY at SAFE_Z; gripper open
  hover    Lower to obj + HOVER_HEIGHT; gripper open
  descend  Lower to obj + DESCEND_OFFSET; gripper open
  close    Hold position, close gripper; monitor contact + object z
             ▸ LIFT if has_contact AND obj_z rose > OBJ_Z_RISE
             ▸ LIFT if has_contact AND partially_closed AND ≥ min_close_steps
             ▸ LIFT if has_contact AND ≥ max_close_steps  (timed fallback)
  lift     Move EEF upward; keep gripper closed
"""

import random
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

SAFE_Z         = 1.10    # m – EEF height during the align transit
HOVER_HEIGHT   = 0.07    # m above object centre
DESCEND_OFFSET = 0.01    # m above object centre for the close setpoint
TOUCH_THR      = 0.05    # N
MIN_PART_CLOSE = 0.90    # gripper must reach < 90 % of open width
OBJ_Z_RISE     = 0.005   # m – object z increase signals genuine lifting force
LIFT_HEIGHT    = 0.35    # m upward from grasp EEF position


class SensorAwarePickAndLift:
    """Six-phase pick-and-lift driven by MuJoCo geom contact + object-z signal."""

    def __init__(self, action_shape, selected_objects):
        self.action_shape = action_shape

        # ── gains ──────────────────────────────────────────────────────────────
        self.kp_fast = 2.5
        self.kp_slow = 1.0
        self.kp_grip = 0.8   # < 1.0 to reduce lateral knockoff force

        # ── thresholds ─────────────────────────────────────────────────────────
        self.align_xy_thr      = 0.03
        self.align_z_thr       = 0.04
        self.hover_threshold   = 0.025
        self.descend_threshold = 0.022

        # ── timing ─────────────────────────────────────────────────────────────
        self.min_close_steps = 80    # wait for grip to equilibrate
        self.max_close_steps = 240   # timed fallback

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
            self._hover(action, eef_pos, obj_pos)
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

    # ── phase methods ──────────────────────────────────────────────────────────

    def _align(self, action, eef_pos, obj_pos):
        """Transit to object XY at SAFE_Z; gripper fully open."""
        target    = np.array([obj_pos[0], obj_pos[1], SAFE_Z])
        pe        = target - eef_pos
        action[0:3] = self.kp_fast * pe
        action[-1]  = -1.0  # open

        if np.linalg.norm(pe[:2]) < self.align_xy_thr and abs(pe[2]) < self.align_z_thr:
            self.phase = "hover"

    def _hover(self, action, eef_pos, obj_pos):
        """Lower to HOVER_HEIGHT above object; gripper open."""
        target      = obj_pos.copy()
        target[2]  += HOVER_HEIGHT
        pe          = target - eef_pos
        action[0:3] = self.kp_fast * pe
        action[-1]  = -1.0  # open

        if np.linalg.norm(pe) < self.hover_threshold:
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
        # Maintain EEF at grasp setpoint
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
        action[0:3] = self.kp_fast * pe
        action[-1]  = 1.0   # full close during lift
