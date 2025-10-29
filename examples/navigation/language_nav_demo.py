"""
Language-Conditioned Navigation Demo

This script demonstrates how to:
1. Load a BEHAVIOR task with natural language description
2. Extract navigation goals from the task (ground truth object positions)
3. Plan a path through multiple waypoints
4. Execute navigation without manipulation or perception

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/language_nav_demo.py

Author: Auto-generated for path planning research
"""

import os
import sys
import yaml
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import omnigibson as og
from omnigibson.object_states import Pose
from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES

# Import our navigation utilities
from utils.navigation_utils import (
    parse_task_to_navigation_goals,
    plan_multipoint_path,
    compute_navigation_metrics,
    visualize_navigation_plan
)


def main():
    """Main demo function."""

    print("\n" + "="*70)
    print("LANGUAGE-CONDITIONED NAVIGATION DEMO")
    print("="*70)

    # ========================================================================
    # STEP 1: Load Configuration
    # ========================================================================
    print("\n[1/6] Loading navigation-only configuration...")

    config_path = os.path.join(
        os.path.dirname(__file__), "../../configs/navigation_only_holonomic.yaml"
    )

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # You can override the task here
    # config['task']['activity_name'] = 'setting_table'
    # config['task']['activity_definition_id'] = 0

    print(f"  Task: {config['task']['type']}")
    print(f"  Scene: {config['scene']['scene_model']}")
    print(f"  Robot: {config['robots'][0]['type']}")
    print(f"  Perception: {'DISABLED' if not config['robots'][0]['obs_modalities'] else 'ENABLED'}")

    # ========================================================================
    # STEP 2: Create Environment
    # ========================================================================
    print("\n[2/6] Creating OmniGibson environment...")
    print("  (This may take a minute on first run...)")

    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]
    task = env.task

    print(f"  ✓ Environment loaded")
    print(f"  ✓ Scene contains {len(scene.objects)} objects")
    print(f"  ✓ Robot: {robot.name}")

    # ========================================================================
    # STEP 3: Extract Navigation Goals
    # ========================================================================
    print("\n[3/6] Getting navigation goal...")

    # For PointNavigationTask, we have a single goal position
    # Access the goal position from the task
    if hasattr(task, 'goal_pos') and task.goal_pos is not None:
        goal_pos = task.goal_pos
        print(f"  Goal position: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")

        # Create a simple navigation goal structure
        nav_goals = [{
            'name': 'navigation_goal',
            'category': 'point',
            'position': np.array(goal_pos),
            'object': None
        }]
    else:
        print("  No pre-set goal, using random navigable point...")
        floor, random_pos = scene.get_random_point(floor=0, robot=robot)
        nav_goals = [{
            'name': 'random_goal',
            'category': 'point',
            'position': np.array([random_pos[0], random_pos[1], 0.0]),
            'object': None
        }]
        print(f"  Random goal: [{random_pos[0]:.2f}, {random_pos[1]:.2f}]")

    print(f"\n  Total navigation goals: {len(nav_goals)}")

    # ========================================================================
    # STEP 4: Plan Multi-Waypoint Path
    # ========================================================================
    print("\n[4/6] Planning navigation path...")

    if len(nav_goals) == 0:
        print("  ⚠ No navigation goals found! Exiting...")
        env.close()
        return

    # Extract waypoint positions
    waypoint_positions = [goal['position'][:2] for goal in nav_goals]  # Take x, y only

    # Plan path through waypoints
    planned_path, total_distance = plan_multipoint_path(
        robot=robot,
        waypoints=waypoint_positions,
        scene=scene
    )

    print(f"  ✓ Path planned!")
    print(f"    Total waypoints: {len(planned_path)}")
    print(f"    Geodesic distance: {total_distance:.2f} meters")

    # Visualize the plan
    visualize_navigation_plan(
        scene=scene,
        robot=robot,
        waypoints=planned_path,
        goal_objects=[g['object'] for g in nav_goals if g['object'] is not None]
    )

    # ========================================================================
    # STEP 5: Execute Navigation (Simulation)
    # ========================================================================
    print("\n[5/6] Simulating navigation...")
    print("  (Running for 100 steps - you can increase this)")

    env.reset()

    # Simple navigation: move toward first waypoint
    # In practice, you would implement a controller or use a learned policy
    num_steps = 100
    path_taken = []

    for step in range(num_steps):
        # Get current robot position and orientation
        robot_pos, robot_quat = robot.states[Pose].get_value()
        robot_pos_np = robot_pos.cpu().numpy() if hasattr(robot_pos, 'cpu') else robot_pos
        robot_quat_np = robot_quat.cpu().numpy() if hasattr(robot_quat, 'cpu') else robot_quat
        path_taken.append(robot_pos_np[:2])

        # Simple proportional controller toward first goal
        if len(waypoint_positions) > 0:
            target = waypoint_positions[0]
            direction = target - robot_pos_np[:2]
            distance = np.linalg.norm(direction)

            if distance > 0.1:  # Not at goal yet
                direction = direction / distance  # Normalize

                # For differential drive (Turtlebot): [left_wheel_vel, right_wheel_vel]
                # Simple differential drive controller
                # Forward velocity component
                forward_vel = 0.3 if distance > 0.5 else 0.1 * distance

                # Compute heading error
                robot_yaw = np.arctan2(2.0 * (robot_quat_np[3] * robot_quat_np[2] + robot_quat_np[0] * robot_quat_np[1]),
                                      1.0 - 2.0 * (robot_quat_np[1]**2 + robot_quat_np[2]**2))
                target_yaw = np.arctan2(direction[1], direction[0])
                yaw_error = target_yaw - robot_yaw
                # Normalize to [-pi, pi]
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                # Differential drive: left and right wheel velocities
                turn_vel = 0.3 * yaw_error
                action = np.array([
                    forward_vel - turn_vel,  # left wheel
                    forward_vel + turn_vel   # right wheel
                ])
            else:
                # Reached goal, stop
                action = np.zeros(2)
                print(f"  ✓ Reached goal 1 at step {step}!")
                waypoint_positions.pop(0)  # Remove reached goal
        else:
            # All goals reached
            action = np.zeros(2)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Print progress every 20 steps
        if step % 20 == 0:
            current_pos = robot_pos_np[:2]
            if waypoint_positions:
                dist_to_goal = np.linalg.norm(waypoint_positions[0] - current_pos)
                print(f"  Step {step:3d}: Distance to next goal: {dist_to_goal:.2f}m")

    # ========================================================================
    # STEP 6: Compute Metrics
    # ========================================================================
    print("\n[6/6] Computing navigation metrics...")

    goal_positions_3d = [goal['position'] for goal in nav_goals]
    metrics = compute_navigation_metrics(
        robot=robot,
        goal_positions=goal_positions_3d,
        scene=scene,
        path_taken=path_taken
    )

    print(f"\n  Metrics:")
    print(f"    Goals reached: {metrics['goals_reached']}/{metrics['total_goals']}")
    print(f"    Distance to nearest goal: {metrics['distance_to_nearest_goal']:.2f}m")
    print(f"    Optimal path length: {metrics['optimal_path_length']:.2f}m")
    if 'actual_path_length' in metrics:
        print(f"    Actual path length: {metrics['actual_path_length']:.2f}m")
        print(f"    Path efficiency: {metrics['path_efficiency']:.2%}")
    print(f"    Success: {metrics['success']}")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\n" + "="*70)
    print("Demo complete! Closing environment...")
    env.close()
    print("✓ Environment closed")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
