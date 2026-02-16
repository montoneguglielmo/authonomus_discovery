"""Demo script to visualize and record any policy on the RandomObjectsEnv."""

import os
import sys
import argparse
import numpy as np
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robosuite import load_composite_controller_config

from envs.object_factory import create_random_objects
from envs.random_objects_env import RandomObjectsEnv
from policies.random_movement import RandomMovementOnTable
from policies.pick_and_lift import PickAndLift
from policies.pick_and_lift_v2 import PickAndLiftV2

POLICIES = {
    "random": lambda shape, objs: RandomMovementOnTable(action_shape=shape),
    "pick_and_lift": lambda shape, objs: PickAndLift(action_shape=shape, selected_objects=objs),
    "pick_and_lift_v2": lambda shape, objs: PickAndLiftV2(action_shape=shape, selected_objects=objs),
}


def main():
    parser = argparse.ArgumentParser(description="Visualize a policy and save a video")
    parser.add_argument("--policy", choices=list(POLICIES.keys()), default="pick_and_lift_v2",
                        help="Policy to run")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="Number of simulation steps")
    parser.add_argument("--num_objects", type=int, default=1,
                        help="Number of random objects on the table")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: <policy>.mp4)")
    parser.add_argument("--camera", type=str, default="frontview",
                        help="Camera name for rendering")
    args = parser.parse_args()

    output_path = args.output or f"{args.policy}.mp4"

    # Create environment
    selected_objects, placement_initializer = create_random_objects(
        min_objects=args.num_objects, max_objects=args.num_objects)
    controller_config = load_composite_controller_config(controller=None, robot="Jaco")

    env = RandomObjectsEnv(
        robots="Jaco",
        custom_objects=selected_objects,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[args.camera],
        camera_heights=512,
        camera_widths=512,
        placement_initializer=placement_initializer,
        controller_configs=controller_config,
    )

    obs = env.reset()
    action_shape = env.action_spec[0].shape

    # Create policy
    policy = POLICIES[args.policy](action_shape, selected_objects)

    print(f"Policy: {args.policy} | Objects: {len(selected_objects)} | Steps: {args.num_steps}")

    # Run and record
    frames = []
    for i in range(args.num_steps):
        action = policy.next_action(obs, env)
        obs, reward, done, _ = env.step(action)
        frame = obs[f"{args.camera}_image"]
        frames.append(np.flip(frame, axis=0))

    env.close()
    imageio.mimsave(output_path, frames, fps=20)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
