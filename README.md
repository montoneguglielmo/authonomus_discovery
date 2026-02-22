# Autonomous Discovery

A robotics data collection framework for gathering manipulation trajectories in simulation. Built on [robosuite](https://robosuite.ai/) (MuJoCo), it supports both automated policy rollouts and keyboard teleoperation, saving episodes in a [LeRobot](https://github.com/huggingface/lerobot)-compatible Parquet format.

## Project Structure

```
scripts/          # Entry points for data collection
  collect_policy.py    # Collect training data with automated policies (headless)
  collect_test.py      # Collect test data: 1 object per episode + metadata
  collect_teleop.py    # Collect via keyboard teleoperation
  evaluate_policy.py   # Evaluate a policy over N episodes with metrics and optional video/log output

envs/             # Custom robosuite environments
  random_objects_env.py  # Table env with randomized objects
  object_factory.py      # Generates random boxes, cylinders, balls, capsules

policies/         # Manipulation policies
  random_movement.py     # Random exploration on the table
  pick_and_lift.py       # State-machine pick-and-lift
  pick_and_lift_v2.py    # Improved pick-and-lift with hover, contact detection

data/             # Data collection and persistence
  collection.py          # Episode collection loop (policy + teleop)
  saving.py              # Parquet writer with video recording

utils/            # Helpers
  transforms.py          # Quaternion to axis-angle conversion

sandbox/          # Experimental scripts and demos
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install robosuite numpy pandas pyarrow opencv-python imageio imageio-ffmpeg pillow
```

## Usage

### Collect with an automated policy

```bash
# Full randomization (default): each episode gets random objects and a random policy
python scripts/collect_policy.py \
  --output_dir data/output \
  --num_episodes 10 \
  --steps_per_episode 500

# Use a specific policy (objects are created once, positions randomize each episode)
python scripts/collect_policy.py \
  --output_dir data/output \
  --policy pick_and_lift \
  --num_episodes 10 \
  --steps_per_episode 500
```

Available `--policy` options: `all` (default), `random`, `pick_and_lift`. In `all` mode, both the object scene and the policy are randomized per episode, and the `task_index` column in the parquet records which policy was used (0 = random, 1 = pick_and_lift).

### Collect a test dataset

Each episode places a single random object on the table, picks a random policy, and records full object metadata (shape, size, density, friction, etc.) alongside the trajectory.

```bash
python scripts/collect_test.py \
  --output_dir data/test \
  --num_episodes 20 \
  --steps_per_episode 500
```

In addition to the standard columns, the test parquet files include: `object_shape`, `object_size`, `object_density`, `object_friction`, `object_rgba`, `object_solref`, `object_solimp`, and `policy_name`.

### Evaluate a policy

Run a policy for N episodes and compute success metrics. Optionally save per-episode videos and CSV logs, organised into `success/` and `failure/` subdirectories.

```bash
# Headless evaluation (metrics only)
python scripts/evaluate_policy.py \
  --policy pick_and_lift_v2 \
  --num_episodes 20

# With video and log output
python scripts/evaluate_policy.py \
  --policy pick_and_lift_v2 \
  --num_episodes 20 \
  --video_dir eval_out/pick_and_lift_v2
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--policy` | `pick_and_lift_v2` | `random`, `pick_and_lift`, `pick_and_lift_v2` |
| `--num_episodes` | `10` | Number of episodes |
| `--steps_per_episode` | `500` | Simulation steps per episode |
| `--metric` | `object_above_table` | Metric to compute |
| `--lift_threshold` | `0.15` | Metres above table surface to count as lifted |
| `--video_dir` | *(none)* | If set, saves videos + CSV logs to `<dir>/success/` and `<dir>/failure/` |

When `--video_dir` is provided, each episode produces:
- `episode_NNNNNN.mp4` — frontview video at 20 fps
- `episode_NNNNNN.csv` — robot state + touch sensors + object positions sampled at 1 Hz

CSV columns: `step`, `time_s`, `eef_x/y/z`, `eef_qx/qy/qz/qw`, `gripper_qpos`, `touch_thumb/index/pinky`, `obj_<name>_x/y/z`.

### Collect via keyboard teleoperation

```bash
python scripts/collect_teleop.py \
  --output_dir data/output \
  --num_episodes 5 \
  --steps_per_episode 500
```

## Data Format

Each episode is saved as a Parquet file under `{output_dir}/data/chunk-{id}/episode_*.parquet` with:

| Column | Description |
|---|---|
| `observation.state` | EEF position (3), axis-angle orientation (3), gripper qpos (1) |
| `action` | Full action vector |
| `timestamp` | Frame timestamp |
| `frame_index` | Index within episode |
| `episode_index` | Episode ID |
| `task_index` | Task type ID |

Dual-camera video streams (frontview + wrist) are saved as MP4 at 30 fps alongside the Parquet files.

## Environment

The `RandomObjectsEnv` places 1-5 randomly generated objects (varying shape, size, color, density) on a table. It uses the Jaco arm with dual 512x512 cameras (front view and eye-in-hand wrist camera), running at 20 Hz control frequency.
