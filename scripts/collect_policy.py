"""Collect episodes using an automated policy (no GUI needed)."""

import argparse
from robosuite import load_composite_controller_config

from envs.object_factory import create_random_objects
from envs.random_objects_env import RandomObjectsEnv
from policies.random_movement import RandomMovementOnTable
from policies.pick_and_lift import PickAndLift
from data.saving import EpisodeParquetWriter
from data.collection import collect_episode


def main():
    parser = argparse.ArgumentParser(description="Collect episodes with automated policies")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory for output data")
    parser.add_argument("--policy", choices=["random", "pick_and_lift"], default="random")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--steps_per_episode", type=int, default=500)
    args = parser.parse_args()

    selected_objects, placement_initializer = create_random_objects()

    controller_config = load_composite_controller_config(controller=None, robot="Jaco")

    env = RandomObjectsEnv(
        robots="Jaco",
        custom_objects=selected_objects,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview", "robot0_eye_in_hand"],
        camera_heights=512,
        camera_widths=512,
        placement_initializer=placement_initializer,
        controller_configs=controller_config,
    )

    if args.policy == "random":
        policy = RandomMovementOnTable(action_shape=env.action_spec[0].shape)
    else:
        policy = PickAndLift(action_shape=env.action_spec[0].shape, selected_objects=selected_objects)

    writer = EpisodeParquetWriter(args.output_dir)

    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        collect_episode(env, policy.next_action, writer, num_steps=args.steps_per_episode)

    env.close()
    print("\nDone. Environment closed.")


if __name__ == "__main__":
    main()
