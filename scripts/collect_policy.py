"""Collect episodes using an automated policy (no GUI needed)."""

import argparse
import random
from robosuite import load_composite_controller_config

from envs.object_factory import create_random_objects
from envs.random_objects_env import RandomObjectsEnv
from policies.random_movement import RandomMovementOnTable
from policies.pick_and_lift import PickAndLift
from data.saving import EpisodeParquetWriter
from data.collection import collect_episode

POLICY_TASK_IDS = {
    "random": 0,
    "pick_and_lift": 1,
}

AVAILABLE_POLICIES = list(POLICY_TASK_IDS.keys())


def create_env(selected_objects, placement_initializer, controller_config):
    return RandomObjectsEnv(
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


def create_policy(policy_name, action_shape, selected_objects):
    if policy_name == "random":
        return RandomMovementOnTable(action_shape=action_shape)
    elif policy_name == "pick_and_lift":
        return PickAndLift(action_shape=action_shape, selected_objects=selected_objects)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def main():
    parser = argparse.ArgumentParser(description="Collect episodes with automated policies")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for output data")
    parser.add_argument("--policy", choices=["random", "pick_and_lift", "all"],
                        default="all")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--steps_per_episode", type=int, default=500)
    args = parser.parse_args()

    controller_config = load_composite_controller_config(controller=None, robot="Jaco")
    writer = EpisodeParquetWriter(args.output_dir)

    if args.policy == "all":
        for ep in range(args.num_episodes):
            selected_objects, placement_initializer = create_random_objects()
            env = create_env(selected_objects, placement_initializer, controller_config)

            policy_name = random.choice(AVAILABLE_POLICIES)
            policy = create_policy(policy_name, env.action_spec[0].shape, selected_objects)
            task_id = POLICY_TASK_IDS[policy_name]

            print(f"\n=== Episode {ep + 1}/{args.num_episodes} | "
                  f"Policy: {policy_name} | Objects: {len(selected_objects)} ===")

            collect_episode(env, policy.next_action, writer,
                            task_id=task_id, num_steps=args.steps_per_episode)
            env.close()
    else:
        selected_objects, placement_initializer = create_random_objects()
        env = create_env(selected_objects, placement_initializer, controller_config)
        policy = create_policy(args.policy, env.action_spec[0].shape, selected_objects)
        task_id = POLICY_TASK_IDS[args.policy]

        for ep in range(args.num_episodes):
            print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
            collect_episode(env, policy.next_action, writer,
                            task_id=task_id, num_steps=args.steps_per_episode)

        env.close()

    print("\nDone. Environment closed.")


if __name__ == "__main__":
    main()
