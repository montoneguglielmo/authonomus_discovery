# Autonomous Discovery

A robotics data collection framework for gathering manipulation trajectories in simulation. Built on [robosuite](https://robosuite.ai/) (MuJoCo), it supports both automated policy rollouts and keyboard teleoperation, saving episodes in a [LeRobot](https://github.com/huggingface/lerobot)-compatible Parquet format.

## Project Structure

```
scripts/          # Entry points for data collection
  collect_policy.py    # Collect with automated policies (headless)
  collect_teleop.py    # Collect via keyboard teleoperation

envs/             # Custom robosuite environments
  random_objects_env.py  # Table env with randomized objects
  object_factory.py      # Generates random boxes, cylinders, balls, capsules

policies/         # Manipulation policies
  random_movement.py     # Random exploration on the table
  pick_and_lift.py       # State-machine pick-and-lift

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
python scripts/collect_policy.py \
  --output_dir data/output \
  --policy pick_and_lift \
  --num_episodes 10 \
  --steps_per_episode 500
```

Available policies: `random`, `pick_and_lift`.

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
