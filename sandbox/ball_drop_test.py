"""
Test script: drops 10 randomly generated balls onto a table and saves each as a video.
Ball properties (size, density, friction, solref, solimp) are randomized via make_ball
from the object factory.

Usage:
    python sandbox/ball_drop_test.py
    python sandbox/ball_drop_test.py -o my_output_dir --duration 4.0
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import sys
import argparse
import mujoco
import numpy as np
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.object_factory import make_ball


MJCF_TEMPLATE = """
<mujoco model="ball_drop">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 -1 2" dir="0 0.5 -1" diffuse="1 1 1"/>
    <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" rgba="0.8 0.8 0.8 1"
          friction="1 0.005 0.0001"/>

    <body name="table" pos="0 0 0.4">
      <geom name="table_top" type="box" size="0.3 0.3 0.02" rgba="0.6 0.4 0.2 1"
            friction="1 0.005 0.0001"/>
    </body>

    <body name="ball" pos="0 0 {drop_height}">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="{radius}"
            density="{density}"
            friction="{friction}"
            solref="{solref}"
            solimp="{solimp}"
            rgba="{rgba}"/>
    </body>
  </worldbody>
</mujoco>
"""

WIDTH = 640
HEIGHT = 480
FPS = 30
NUM_SIMULATIONS = 10
TABLE_TOP_Z = 0.42  # table center (0.4) + half-height (0.02)
DROP_OFFSET = 0.20  # 20cm above table


def simulate_ball(ball, output_path, duration):
    """Run one drop simulation and save the video."""
    radius = ball.size[0]
    drop_height = TABLE_TOP_Z + DROP_OFFSET + radius

    xml = MJCF_TEMPLATE.format(
        drop_height=drop_height,
        radius=radius,
        density=ball.density,
        friction=" ".join(f"{v}" for v in ball.friction),
        solref=" ".join(f"{v}" for v in ball.solref),
        solimp=" ".join(f"{v}" for v in ball.solimp),
        rgba=" ".join(f"{v}" for v in ball.rgba),
    )

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.5]
    camera.distance = 1.5
    camera.azimuth = 135
    camera.elevation = -25

    steps_per_frame = int(1.0 / (model.opt.timestep * FPS))
    total_steps = int(duration / model.opt.timestep)
    frames = []

    for step in range(total_steps):
        mujoco.mj_step(model, data)
        if step % steps_per_frame == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render().copy())

    renderer.close()
    imageio.mimsave(output_path, frames, fps=FPS)


def main():
    parser = argparse.ArgumentParser(description="Drop 10 random balls and save videos")
    parser.add_argument("-o", "--output-dir", type=str, default="sandbox/ball_drop_videos",
                        help="Output directory for videos (default: sandbox/ball_drop_videos)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Simulation duration in seconds (default: 3.0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(NUM_SIMULATIONS):
        ball = make_ball(f"ball_{i}")

        print(f"\n[{i+1}/{NUM_SIMULATIONS}] ball_{i}")
        print(f"  radius:      {ball.size[0]:.4f} m")
        print(f"  density:     {ball.density:.0f} kg/m^3")
        print(f"  friction:    {[f'{v:.4f}' for v in ball.friction]}")
        print(f"  solref:      {[f'{v:.2f}' for v in ball.solref]}")
        print(f"  solimp:      {[f'{v:.4f}' for v in ball.solimp]}")

        output_path = os.path.join(args.output_dir, f"ball_{i}.mp4")
        simulate_ball(ball, output_path, args.duration)
        print(f"  saved:       {output_path}")

    print(f"\nDone! {NUM_SIMULATIONS} videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
