"""Collect episodes via keyboard teleoperation."""

import argparse
from robosuite import load_composite_controller_config
from robosuite.devices import Keyboard

from envs.object_factory import create_random_objects
from envs.random_objects_env import RandomObjectsEnv
from data.saving import EpisodeParquetWriter
from data.collection import collect_episode, KeyboardActionSource


def main():
    parser = argparse.ArgumentParser(description="Collect episodes via keyboard teleoperation")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory for output data")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--steps_per_episode", type=int, default=500)
    args = parser.parse_args()

    selected_objects, placement_initializer = create_random_objects()

    controller_config = load_composite_controller_config(controller=None, robot="Jaco")

    # Teleop needs on-screen renderer for the operator to see the scene
    env = RandomObjectsEnv(
        robots="Jaco",
        custom_objects=selected_objects,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview", "robot0_eye_in_hand"],
        camera_heights=512,
        camera_widths=512,
        placement_initializer=placement_initializer,
        controller_configs=controller_config,
        renderer="mjviewer",
    )

    device = Keyboard(env=env, pos_sensitivity=1.0, rot_sensitivity=1.0)
    device.start_control()
    action_source = KeyboardActionSource(device, env)

    writer = EpisodeParquetWriter(args.output_dir)

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        print("Use keyboard to control the robot. Press 'q' to end episode.")
        device.start_control()
        collect_episode(env, action_source, writer,
                        num_steps=args.steps_per_episode, render=True,
                        enforce_timing=True)

    env.close()
    print("\nDone. Environment closed.")


if __name__ == "__main__":
    main()
