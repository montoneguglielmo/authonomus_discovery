"""Collect a test/evaluation dataset: one random object per episode, random policy."""

import argparse
import random
from robosuite import load_composite_controller_config

from envs.object_factory import create_random_objects, extract_object_metadata
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
        gripper_types="JacoThreeFingerTouchGripper",
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
    parser = argparse.ArgumentParser(
        description="Collect test dataset: 1 random object, random policy per episode")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for output data")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--steps_per_episode", type=int, default=500)
    args = parser.parse_args()

    controller_config = load_composite_controller_config(controller=None, robot="Jaco")
    writer = EpisodeParquetWriter(args.output_dir)

    for ep in range(args.num_episodes):
        # Single random object per episode
        selected_objects, placement_initializer = create_random_objects(
            min_objects=1, max_objects=1)
        env = create_env(selected_objects, placement_initializer, controller_config)

        # Random policy
        policy_name = random.choice(AVAILABLE_POLICIES)
        policy = create_policy(policy_name, env.action_spec[0].shape, selected_objects)
        task_id = POLICY_TASK_IDS[policy_name]

        # Store object metadata + policy name for this episode
        obj = selected_objects[0]
        metadata = extract_object_metadata(obj)
        metadata["policy_name"] = policy_name
        writer.set_episode_metadata(metadata)

        print(f"\n=== Episode {ep + 1}/{args.num_episodes} | "
              f"Policy: {policy_name} | Object: {obj.name} ===")

        collect_episode(env, policy.next_action, writer,
                        task_id=task_id, num_steps=args.steps_per_episode)
        env.close()

    print("\nDone. Environment closed.")


if __name__ == "__main__":
    main()
