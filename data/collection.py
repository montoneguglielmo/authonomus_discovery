"""
Shared data collection loop used by both policy-driven and teleoperation scripts.
"""

import time
import cv2
import numpy as np
from utils.transforms import quat2axisangle


_camera_window_initialized = False


def _show_camera_feeds(obs):
    """Display wrist camera in an OpenCV window."""
    global _camera_window_initialized
    wrist = np.rot90(obs["robot0_eye_in_hand_image"], 2)
    wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
    if not _camera_window_initialized:
        cv2.namedWindow("Wrist Camera", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("Wrist Camera", 1300, 0)
        _camera_window_initialized = True
    cv2.imshow("Wrist Camera", wrist_bgr)
    cv2.waitKey(1)


class EpisodeReset(Exception):
    """Raised when the teleop user requests an episode reset (e.g. Ctrl+Q)."""
    pass


class KeyboardActionSource:
    """Wraps robosuite Keyboard device to match the action_source(obs, env) callable interface."""

    def __init__(self, device, env):
        self.device = device
        self.env = env

    def __call__(self, obs, env):
        input_ac_dict = self.device.input2action()

        if input_ac_dict is None:
            raise EpisodeReset()

        robot = env.robots[0]
        controller_input_type = robot.part_controllers[robot.arms[0]].input_type

        action_dict = {}
        for arm in robot.arms:
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            action_dict[f"{arm}_gripper"] = input_ac_dict[f"{arm}_gripper"]

        return robot.create_action_vector(action_dict)


def collect_episode(env, action_source, writer, task_id=0, num_steps=500,
                    settle_steps=20, control_freq=20, render=False):
    """
    Run one episode of data collection.

    Args:
        env: robosuite environment (already created)
        action_source: callable(obs, env) -> action (works for both policies and teleop wrapper)
        writer: EpisodeParquetWriter instance
        task_id: task index for the parquet file
        num_steps: steps per episode
        settle_steps: zero-action steps to let physics settle
        control_freq: Hz for the control loop
        render: whether to call env.render() each step (needed for teleop GUI)
    """
    obs = env.reset()

    # Let objects settle on the table
    print("Letting objects settle on table...")
    for _ in range(settle_steps):
        zero_action = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(zero_action)
        if render:
            env.render()
            _show_camera_feeds(obs)

    target_dt = 1.0 / control_freq

    try:
        for i in range(num_steps):
            step_start = time.time()

            action = action_source(obs, env)
            obs, reward, done, info = env.step(action)

            if render:
                env.render()
                _show_camera_feeds(obs)

            touch = env._get_touch_sensor_data() if hasattr(env, "_get_touch_sensor_data") else np.zeros(3)
            state = np.concatenate([
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
                touch,                              # [thumb, index, pinky] normal force (N)
            ])

            writer.add_step(
                state=state,
                action=np.array(action),
                task_index=task_id,
                frame_agent=np.rot90(obs["frontview_image"], 2),
                frame_wrist=np.rot90(obs["robot0_eye_in_hand_image"], 2)
            )

            if (i + 1) % 50 == 0:
                print(f"  Step {i + 1}/{num_steps}")

            elapsed = time.time() - step_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    except EpisodeReset:
        print("Episode ended early by user reset.")

    if render:
        global _camera_window_initialized
        _camera_window_initialized = False
        cv2.destroyAllWindows()

    writer.save_episode()
