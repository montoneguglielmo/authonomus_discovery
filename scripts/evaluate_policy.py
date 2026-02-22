"""Evaluate the performance of a policy over N episodes."""

import os
import sys
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robosuite import load_composite_controller_config

from envs.object_factory import create_random_objects
from envs.random_objects_env import RandomObjectsEnv
from policies.random_movement import RandomMovementOnTable
from policies.pick_and_lift import PickAndLift
from policies.pick_and_lift_v2 import PickAndLiftV2
from data.saving import save_video

TABLE_HEIGHT = 0.8  # z-position of table surface in RandomObjectsEnv

POLICIES = {
    "random":           lambda shape, objs: RandomMovementOnTable(action_shape=shape),
    "pick_and_lift":    lambda shape, objs: PickAndLift(action_shape=shape, selected_objects=objs),
    "pick_and_lift_v2": lambda shape, objs: PickAndLiftV2(action_shape=shape, selected_objects=objs),
}


def create_env(selected_objects, placement_initializer, controller_config, record=False):
    return RandomObjectsEnv(
        robots="Jaco",
        gripper_types="JacoThreeFingerTouchGripper",
        custom_objects=selected_objects,
        has_renderer=False,
        has_offscreen_renderer=record,
        use_camera_obs=record,
        camera_names=["frontview"] if record else [],
        camera_heights=512 if record else 1,
        camera_widths=512 if record else 1,
        placement_initializer=placement_initializer,
        controller_configs=controller_config,
    )


def get_object_positions(env, selected_objects):
    """Return {obj.name: np.array([x, y, z])} for each object."""
    positions = {}
    for obj in selected_objects:
        body_id = env.sim.model.body_name2id(obj.root_body)
        positions[obj.name] = env.sim.data.body_xpos[body_id].copy()
    return positions


# --- Metrics ---

def metric_object_above_table(object_positions, threshold):
    """True if any object centre is more than `threshold` metres above the table."""
    return any(pos[2] > TABLE_HEIGHT + threshold for pos in object_positions.values())


METRICS = {
    "object_above_table": metric_object_above_table,
}


# --- Logging ---

def save_episode_log(log_rows, path):
    if not log_rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)


# --- Episode runner ---

CONTROL_FREQ = 20  # Hz — matches robosuite default and data/collection.py


def run_episode(env, policy, num_steps, selected_objects,
                settle_steps=20, record=False, log_every_n=CONTROL_FREQ):
    """Reset env, settle, then run policy for num_steps.

    Returns:
        frames:    list of RGB numpy arrays (empty when record=False)
        log_rows:  list of dicts sampled every log_every_n steps (1 Hz by default)
    """
    obs = env.reset()
    frames = []
    log_rows = []

    zero_action = np.zeros(env.action_spec[0].shape)
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(zero_action)

    for step in range(num_steps):
        action = policy.next_action(obs, env)
        obs, _, done, _ = env.step(action)

        if record:
            frame = obs.get("frontview_image")
            if frame is not None:
                frames.append(np.flip(frame, axis=0))

        if step % log_every_n == 0:
            touch = (env._get_touch_sensor_data()
                     if hasattr(env, "_get_touch_sensor_data") else np.zeros(3))
            obj_positions = get_object_positions(env, selected_objects)
            row = {
                "step":         step,
                "time_s":       step / CONTROL_FREQ,
                "eef_x":        obs["robot0_eef_pos"][0],
                "eef_y":        obs["robot0_eef_pos"][1],
                "eef_z":        obs["robot0_eef_pos"][2],
                "eef_qx":       obs["robot0_eef_quat"][0],
                "eef_qy":       obs["robot0_eef_quat"][1],
                "eef_qz":       obs["robot0_eef_quat"][2],
                "eef_qw":       obs["robot0_eef_quat"][3],
                "gripper_qpos": float(obs["robot0_gripper_qpos"][0]),
                "touch_thumb":  float(touch[0]),
                "touch_index":  float(touch[1]),
                "touch_pinky":  float(touch[2]),
            }
            for name, pos in obj_positions.items():
                row[f"obj_{name}_x"] = float(pos[0])
                row[f"obj_{name}_y"] = float(pos[1])
                row[f"obj_{name}_z"] = float(pos[2])
            log_rows.append(row)

        if done:
            break

    return frames, log_rows


# --- Main evaluation loop ---

def evaluate(policy_name, num_episodes, steps_per_episode, metric_name,
             lift_threshold, video_dir):
    controller_config = load_composite_controller_config(controller=None, robot="Jaco")
    metric_fn = METRICS[metric_name]
    record = video_dir is not None

    if record:
        success_dir = os.path.join(video_dir, "success")
        failure_dir = os.path.join(video_dir, "failure")
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(failure_dir, exist_ok=True)

    successes = 0
    episode_results = []

    for ep in range(num_episodes):
        selected_objects, placement_initializer = create_random_objects(
            min_objects=1, max_objects=1)
        env = create_env(selected_objects, placement_initializer, controller_config,
                         record=record)

        action_shape = env.action_spec[0].shape
        policy = POLICIES[policy_name](action_shape, selected_objects)

        frames, log_rows = run_episode(env, policy, steps_per_episode,
                                       selected_objects=selected_objects, record=record)

        object_positions = get_object_positions(env, selected_objects)
        success = metric_fn(object_positions, threshold=lift_threshold)
        if success:
            successes += 1

        episode_results.append({"episode": ep, "success": success,
                                 "object_positions": object_positions})

        pos_str = "  ".join(
            f"{name}=[{pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:+.2f}]"
            for name, pos in object_positions.items()
        )
        print(f"Episode {ep + 1:>{len(str(num_episodes))}}/{num_episodes}:  "
              f"success={str(success):<5}  {pos_str}")

        if record:
            dest_dir = success_dir if success else failure_dir
            save_video(frames, filename=f"episode_{ep:06d}.mp4", fps=20, save_dir=dest_dir)
            save_episode_log(log_rows, os.path.join(dest_dir, f"episode_{ep:06d}.csv"))

        env.close()

    success_rate = successes / num_episodes
    print(f"\n=== Evaluation Results ===")
    print(f"Policy:       {policy_name}")
    print(f"Metric:       {metric_name}  (threshold={lift_threshold}m above table)")
    print(f"Episodes:     {num_episodes}")
    print(f"Success rate: {success_rate:.1%}  ({successes}/{num_episodes})")
    if record:
        print(f"Videos + logs saved: {video_dir}/{{success,failure}}/")

    return episode_results, success_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy over N episodes")
    parser.add_argument("--policy", choices=list(POLICIES.keys()),
                        default="pick_and_lift_v2",
                        help="Policy to evaluate")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes (default: 10)")
    parser.add_argument("--steps_per_episode", type=int, default=500,
                        help="Simulation steps per episode (default: 500)")
    parser.add_argument("--metric", choices=list(METRICS.keys()),
                        default="object_above_table",
                        help="Metric to compute (default: object_above_table)")
    parser.add_argument("--lift_threshold", type=float, default=0.15,
                        help="Metres above table surface to count as lifted (default: 0.15)")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Directory to save episode videos organised into "
                             "success/ and failure/ subdirectories")
    args = parser.parse_args()

    evaluate(
        policy_name=args.policy,
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
        metric_name=args.metric,
        lift_threshold=args.lift_threshold,
        video_dir=args.video_dir,
    )


if __name__ == "__main__":
    main()
