"""
Verify that per-finger touch sensors work correctly in JacoThreeFingerTouchGripper.

Run from the project root:
    python sandbox/test_touch_sensors.py
"""
import numpy as np
from robosuite import load_composite_controller_config

from envs.object_factory import create_random_objects
from envs.random_objects_env import RandomObjectsEnv  # also imports models.grippers
from utils.transforms import quat2axisangle


def main():
    selected_objects, placement_initializer = create_random_objects(
        min_objects=1, max_objects=1
    )
    controller_config = load_composite_controller_config(controller=None, robot="Jaco")

    env = RandomObjectsEnv(
        robots="Jaco",
        gripper_types="JacoThreeFingerTouchGripper",
        custom_objects=selected_objects,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        placement_initializer=placement_initializer,
        controller_configs=controller_config,
    )

    env.reset()

    # --- Verify sensor registration ---
    print("All MuJoCo sensors in model:")
    for name in env.sim.model.sensor_names:
        print(f"  {name}")

    touch_keys = ["touch_thumb", "touch_index", "touch_pinky"]
    print("\nLooking for touch sensors by suffix:")
    for key in touch_keys:
        matched = [n for n in env.sim.model.sensor_names if n.endswith(key)]
        if matched:
            print(f"  OK  '{key}' -> '{matched[0]}'")
        else:
            print(f"  MISSING  '{key}' not found in sensor_names!")

    # --- Run steps with gripper closing and read touch data ---
    print("\nRunning 100 steps with gripper closing...")
    action = np.zeros(env.action_spec[0].shape)
    action[-1] = 1.0  # close gripper

    for _ in range(100):
        env.step(action)

    touch = env._get_touch_sensor_data()
    print(f"\nTouch sensor readings after closing gripper:")
    print(f"  Thumb:  {touch[0]:.4f} N")
    print(f"  Index:  {touch[1]:.4f} N")
    print(f"  Pinky:  {touch[2]:.4f} N")

    # --- Verify state vector shape ---
    obs, _, _, _ = env.step(action)
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
        touch,
    ])
    print(f"\nState vector shape: {state.shape}")
    expected_len = 3 + 3 + len(obs["robot0_gripper_qpos"]) + 3
    assert state.shape[0] == expected_len, (
        f"Unexpected state dim {state.shape[0]}, expected {expected_len}"
    )
    print("All checks passed.")

    env.close()


if __name__ == "__main__":
    main()
