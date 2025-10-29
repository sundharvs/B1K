"""
Navigation Policy Training Framework

Gym-compatible wrapper for training navigation policies with language conditioning.
Supports reinforcement learning for path planning from natural language tasks.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python training/nav_policy_trainer.py [--algorithm ppo] [--episodes 1000]

Requirements:
    pip install stable-baselines3  # For RL algorithms (optional)

Author: Auto-generated for path planning research
"""

import os
import sys
import yaml
import argparse
import numpy as np
from typing import Dict, Tuple, Any, Optional
import gymnasium as gym
from gymnasium import spaces

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import omnigibson as og
from omnigibson.object_states import Pose
from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES

from utils.navigation_utils import (
    parse_task_to_navigation_goals,
    compute_navigation_metrics,
)


class LanguageNavigationEnv(gym.Env):
    """
    Gym environment for language-conditioned navigation.

    Observation space:
        - robot_position: [x, y] robot position
        - robot_orientation: [theta] robot heading
        - goal_position: [x, y] current navigation goal
        - goal_distance: [d] distance to goal
        - goal_bearing: [theta] angle to goal

    Action space:
        - For holonomic base: [vx, vy, omega]
        - For differential drive: [v_left, v_right]
    """

    def __init__(self, config_path: str, max_steps: int = 500, goal_tolerance: float = 0.5):
        """
        Initialize the navigation environment.

        Args:
            config_path: Path to OmniGibson configuration file
            max_steps: Maximum steps per episode
            goal_tolerance: Distance threshold for reaching a goal (meters)
        """
        super().__init__()

        # Load configuration
        with open(config_path, "r") as f:
            self.og_config = yaml.load(f, Loader=yaml.FullLoader)

        # Environment parameters
        self.max_steps = max_steps
        self.goal_tolerance = goal_tolerance

        # Create OmniGibson environment
        print("Creating OmniGibson environment...")
        self.og_env = og.Environment(configs=self.og_config)
        self.scene = self.og_env.scene
        self.robot = self.og_env.robots[0]
        self.task = self.og_env.task

        # Define observation space
        # [robot_x, robot_y, robot_theta, goal_x, goal_y, goal_distance, goal_bearing]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, -np.inf, -np.inf, 0.0, -np.pi]),
            high=np.array([np.inf, np.inf, np.pi, np.inf, np.inf, np.inf, np.pi]),
            dtype=np.float32
        )

        # Define action space based on robot controller
        robot_type = self.og_config['robots'][0]['type']
        controller_name = self.og_config['robots'][0]['controller_config']['base']['name']

        if 'Holonomic' in controller_name:
            # [vx, vy, omega]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32
            )
        else:
            # Differential drive: [v_left, v_right]
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )

        # Episode tracking
        self.current_step = 0
        self.current_goal_idx = 0
        self.navigation_goals = []
        self.path_taken = []

        print(f"✓ Navigation environment initialized")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space.shape}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed
            options: Additional options (can include 'activity_name' to change task)

        Returns:
            Tuple of (observation, info dict)
        """
        super().reset(seed=seed)

        # Optionally change task
        if options and 'activity_name' in options:
            activity_name = options['activity_name']
            if activity_name in BEHAVIOR_ACTIVITIES:
                self.og_config['task']['activity_name'] = activity_name
                self.og_env.close()
                self.og_env = og.Environment(configs=self.og_config)
                self.scene = self.og_env.scene
                self.robot = self.og_env.robots[0]
                self.task = self.og_env.task

        # Reset OmniGibson environment
        self.og_env.reset()

        # Parse navigation goals
        self.navigation_goals = parse_task_to_navigation_goals(self.task, max_goals=5)

        if not self.navigation_goals:
            # No goals found, sample random goal
            floor, pos = self.scene.get_random_point(floor=0, robot=self.robot)
            self.navigation_goals = [{
                'position': np.array([pos[0], pos[1], 0.0]),
                'name': 'random_goal',
                'category': 'random'
            }]

        # Reset episode tracking
        self.current_step = 0
        self.current_goal_idx = 0
        self.path_taken = []

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Action array matching action_space

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Execute action in OmniGibson
        _, _, terminated, truncated, _ = self.og_env.step(action)

        # Track path
        robot_pos, _ = self.robot.states[Pose].get_value()
        robot_pos_np = robot_pos.cpu().numpy() if hasattr(robot_pos, 'cpu') else robot_pos
        self.path_taken.append(robot_pos_np[:2])

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        self.current_step += 1

        # Check if goal reached
        goal_reached = obs[5] < self.goal_tolerance  # goal_distance < tolerance

        if goal_reached:
            print(f"  ✓ Goal {self.current_goal_idx + 1} reached!")
            self.current_goal_idx += 1

            # Check if all goals reached
            if self.current_goal_idx >= len(self.navigation_goals):
                terminated = True
                reward += 100.0  # Bonus for completing all goals
            else:
                reward += 10.0  # Bonus for reaching a goal

        # Truncate if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Observation array: [robot_x, robot_y, robot_theta, goal_x, goal_y, goal_distance, goal_bearing]
        """
        # Robot pose
        robot_pos, robot_quat = self.robot.states[Pose].get_value()
        robot_pos_np = robot_pos.cpu().numpy() if hasattr(robot_pos, 'cpu') else robot_pos
        robot_quat_np = robot_quat.cpu().numpy() if hasattr(robot_quat, 'cpu') else robot_quat

        # Convert quaternion to yaw angle
        robot_theta = self._quat_to_yaw(robot_quat_np)

        # Current goal position
        if self.current_goal_idx < len(self.navigation_goals):
            goal_pos = self.navigation_goals[self.current_goal_idx]['position'][:2]
        else:
            goal_pos = robot_pos_np[:2]  # All goals reached

        # Distance and bearing to goal
        goal_vec = goal_pos - robot_pos_np[:2]
        goal_distance = np.linalg.norm(goal_vec)
        goal_bearing = np.arctan2(goal_vec[1], goal_vec[0]) - robot_theta

        # Normalize bearing to [-pi, pi]
        goal_bearing = np.arctan2(np.sin(goal_bearing), np.cos(goal_bearing))

        obs = np.array([
            robot_pos_np[0],
            robot_pos_np[1],
            robot_theta,
            goal_pos[0],
            goal_pos[1],
            goal_distance,
            goal_bearing
        ], dtype=np.float32)

        return obs

    def _compute_reward(self) -> float:
        """
        Compute reward for current state.

        Returns:
            Reward value
        """
        obs = self._get_observation()
        goal_distance = obs[5]

        # Distance-based reward (negative distance)
        reward = -goal_distance * 0.1

        # Time penalty to encourage efficiency
        reward -= 0.01

        return reward

    def _get_info(self) -> Dict:
        """
        Get info dict for current state.

        Returns:
            Info dictionary
        """
        goal_positions = [g['position'] for g in self.navigation_goals]
        metrics = compute_navigation_metrics(
            robot=self.robot,
            goal_positions=goal_positions,
            scene=self.scene,
            path_taken=self.path_taken
        )

        info = {
            'current_step': self.current_step,
            'current_goal_idx': self.current_goal_idx,
            'total_goals': len(self.navigation_goals),
            'metrics': metrics,
            'task_name': self.og_config['task']['activity_name']
        }

        return info

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        """
        Convert quaternion to yaw angle.

        Args:
            quat: Quaternion [x, y, z, w]

        Returns:
            Yaw angle in radians
        """
        # Extract yaw from quaternion
        x, y, z, w = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def close(self):
        """Close the environment."""
        self.og_env.close()


def train_random_policy(env: LanguageNavigationEnv, num_episodes: int = 10):
    """
    Train a simple random policy (baseline).

    Args:
        env: Navigation environment
        num_episodes: Number of training episodes
    """
    print(f"\n{'='*70}")
    print("TRAINING RANDOM POLICY (BASELINE)")
    print(f"{'='*70}\n")

    episode_rewards = []
    episode_successes = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Task: {info['task_name']}")
        print(f"  Goals: {info['total_goals']}")

        while not (terminated or truncated):
            # Random action
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        success = info['metrics']['success']
        episode_rewards.append(episode_reward)
        episode_successes.append(success)

        print(f"  Episode reward: {episode_reward:.2f}")
        print(f"  Success: {success}")
        print(f"  Goals reached: {info['metrics']['goals_reached']}/{info['total_goals']}")

    # Print summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Success rate: {np.mean(episode_successes):.2%}")
    print(f"{'='*70}\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Navigation Policy Trainer")
    parser.add_argument("--config", type=str, default="navigation_only_holonomic.yaml",
                       help="Config file to use")
    parser.add_argument("--algorithm", type=str, default="random", choices=["random", "ppo"],
                       help="Training algorithm")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Maximum steps per episode")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("NAVIGATION POLICY TRAINING")
    print("="*70)
    print(f"  Config: {args.config}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print("="*70 + "\n")

    # Create environment
    config_path = os.path.join(
        os.path.dirname(__file__), "../configs", args.config
    )

    env = LanguageNavigationEnv(
        config_path=config_path,
        max_steps=args.max_steps
    )

    try:
        if args.algorithm == "random":
            train_random_policy(env, num_episodes=args.episodes)

        elif args.algorithm == "ppo":
            try:
                from stable_baselines3 import PPO
                from stable_baselines3.common.env_checker import check_env

                print("Checking environment...")
                check_env(env)
                print("✓ Environment is compatible with stable-baselines3\n")

                print("Training PPO agent...")
                model = PPO("MlpPolicy", env, verbose=1)
                model.learn(total_timesteps=args.episodes * args.max_steps)

                print("\n✓ Training complete!")
                print("To save model: model.save('navigation_ppo')")

            except ImportError:
                print("✗ stable-baselines3 not installed.")
                print("Install with: pip install stable-baselines3")
                return

    finally:
        env.close()

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
