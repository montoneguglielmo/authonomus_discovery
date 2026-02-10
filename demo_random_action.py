import time
import imageio

from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *

MAX_FR = 25  # max frame rate for running simluation

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "single-robot":
            options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True, exclude_single_arm=True)
        else:
            options["robots"] = []

            # Have user choose two robots
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=False, use_humanoids=True))
    # If a humanoid environment has been chosen, choose humanoid robots
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    else:
        options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names="frontview",
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
    )
    env.reset()
    #env.viewer.set_camera(camera_id=0)
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # do visualization
    frames = []
    for i in range(200):
        start = time.time()
        action = np.random.randn(*env.action_spec[0].shape)
        obs, reward, done, _ = env.step(action)
        frame = obs["frontview_image"]
        frame = np.flip(frame, axis=0)
        frames.append(frame)

        # limit frame rate if necessary
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
    
    env.close()
    imageio.mimsave("lift_hello_world.mp4", frames, fps=20)

    print("Saved video to lift_hello_world.mp4 🎉")