if __name__ == '__main__':
    import imageio
    import cv2
    import time
    from custom_env import RandomObjectsEnv
    from utils import create_random_objects, quat2axisangle
    from robosuite import load_composite_controller_config
    from policies import PickAndLift, RandomMovementOnTable
    import numpy as np
    from saving import EpisodeParquetWriter


    selected_objects, placement_initializer = create_random_objects()

    # Load default controller config for the robot
    controller_config = load_composite_controller_config(controller=None, robot="Jaco")

    # Create custom environment with our objects
    env = RandomObjectsEnv(
        robots="Jaco",
        custom_objects=selected_objects,  # Pass our custom objects!
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview", "robot0_eye_in_hand"],
        camera_heights=512,
        camera_widths=512,
        placement_initializer=placement_initializer,
        controller_configs=controller_config,
    )
    
    #policy = PickAndLift(action_shape=env.action_spec[0].shape, selected_objects=selected_objects)
    policy = RandomMovementOnTable(action_shape=env.action_spec[0].shape)
    
    
    # Reset environment
    print("\nResetting environment...")
    obs = env.reset()

    # Let objects settle on the table (physics settling)
    print("Letting objects settle on table...")
    for _ in range(20):
        zero_action = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(zero_action)

    # Run random actions for a few steps
    print("\nRunning robot with random actions...")
    frames = []
    num_steps = 500

   
    writer = EpisodeParquetWriter('../to_kill')
    task_id = 0
    
    target_dt = 1.0 / 20  # 20 Hz
    for i in range(num_steps):
        step_start = time.time()

        action = policy.next_action(obs, env)
        obs, reward, done, info = env.step(action)
        print(obs.keys())

        state = np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"]
            ])
        
        writer.add_step(state=state,
                        action=np.array(action),
                        task_index=task_id,
                        frame_agent=np.rot90(obs["frontview_image"], 2),
                        frame_wrist=np.rot90(obs["robot0_eye_in_hand_image"], 2)
                        )

        # Grab camera frame if available
        if "frontview_image" in obs:
            frame = obs["frontview_image"]
            frame = np.flip(frame, axis=0).copy()
            frames.append(frame)

        # Print progress every 50 steps
        if (i + 1) % 50 == 0:
            print(f"  Step {i + 1}/{num_steps}")

        # Enforce 20 Hz
        elapsed = time.time() - step_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    env.close()
    print("\nEnvironment closed.")

    # Save video if frames were captured
    if frames:
        output_file = "random_env_test.mp4"
        imageio.mimsave(output_file, frames, fps=20)
        print(f"\n✅ Test complete! Video saved to {output_file}")
    else:
        print("\n✅ Test complete! (No video frames captured)")
        
    
    # Write the episode to disk    
    writer.save_episode()
                