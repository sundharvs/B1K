"""
Interactive Navigation Interface

Simple command-line interface for testing language-conditioned navigation.
Allows users to input custom tasks or object names and see navigation plans.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/interactive_nav_interface.py

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

from utils.navigation_utils import (
    parse_task_to_navigation_goals,
    language_to_objects,
    plan_multipoint_path,
    get_navigable_positions_near_object,
    visualize_navigation_plan,
)


class NavigationInterface:
    """Interactive interface for navigation testing."""

    def __init__(self, config_path: str):
        """
        Initialize the navigation interface.

        Args:
            config_path: Path to configuration YAML file
        """
        print("\n" + "="*70)
        print("INTERACTIVE NAVIGATION INTERFACE")
        print("="*70)

        # Load configuration
        print("\nLoading configuration...")
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Create environment
        print("Creating environment (this may take a minute)...")
        self.env = og.Environment(configs=self.config)
        self.scene = self.env.scene
        self.robot = self.env.robots[0]
        self.task = self.env.task

        # Reset
        self.env.reset()

        print(f"✓ Environment loaded")
        print(f"  Scene: {self.config['scene']['scene_model']}")
        print(f"  Robot: {self.robot.name}")
        print(f"  Task: {self.config['task']['activity_name']}")
        print(f"  Objects in scene: {len(self.scene.objects)}")

    def list_available_objects(self):
        """List all objects available in the current scene."""
        print("\n" + "-"*70)
        print("AVAILABLE OBJECTS IN SCENE")
        print("-"*70)

        task_objects = []
        for obj_instance, bddl_entity in self.task.object_scope.items():
            if bddl_entity.exists and "agent" not in obj_instance.lower():
                category = bddl_entity.og_categories[0] if bddl_entity.og_categories else bddl_entity.synset
                task_objects.append((obj_instance, category, bddl_entity))

        for i, (name, category, entity) in enumerate(task_objects, 1):
            obj = entity.wrapped_obj
            pos, _ = obj.states[Pose].get_value()
            pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else pos
            print(f"{i:3d}. {name:30s} ({category:20s}) at [{pos_np[0]:.1f}, {pos_np[1]:.1f}, {pos_np[2]:.1f}]")

        print(f"\nTotal: {len(task_objects)} task-relevant objects")
        print("-"*70)

    def navigate_to_object(self, query: str):
        """
        Plan navigation to object(s) matching the query.

        Args:
            query: Object name, category, or description
        """
        print(f"\n{'='*70}")
        print(f"Navigation Query: '{query}'")
        print("="*70)

        # Find matching objects
        matching_objects = language_to_objects(query, self.task.object_scope, fuzzy_match=True)

        if not matching_objects:
            print(f"✗ No objects found matching '{query}'")
            return

        print(f"\nFound {len(matching_objects)} matching object(s):")
        for obj_name in matching_objects[:5]:  # Show first 5
            bddl_entity = self.task.object_scope[obj_name]
            obj = bddl_entity.wrapped_obj
            pos, _ = obj.states[Pose].get_value()
            pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else pos
            print(f"  - {obj_name} at [{pos_np[0]:.2f}, {pos_np[1]:.2f}, {pos_np[2]:.2f}]")

        # Get positions
        goal_positions = []
        goal_objects = []
        for obj_name in matching_objects[:3]:  # Navigate to first 3
            bddl_entity = self.task.object_scope[obj_name]
            obj = bddl_entity.wrapped_obj
            pos, _ = obj.states[Pose].get_value()
            pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else pos

            # Get navigable positions near the object
            nav_positions = get_navigable_positions_near_object(
                obj=obj,
                scene=self.scene,
                robot=self.robot,
                num_samples=8,
                distance_range=(0.5, 1.5)
            )

            if nav_positions:
                goal_positions.append(nav_positions[0])  # Use closest navigable position
                goal_objects.append(obj)
            else:
                # Fallback to object position
                goal_positions.append(pos_np[:2])
                goal_objects.append(obj)

        # Plan path
        print(f"\nPlanning path to {len(goal_positions)} goal(s)...")
        planned_path, total_distance = plan_multipoint_path(
            robot=self.robot,
            waypoints=goal_positions,
            scene=self.scene
        )

        if planned_path:
            print(f"✓ Path planned!")
            print(f"  Waypoints: {len(planned_path)}")
            print(f"  Total distance: {total_distance:.2f} meters")

            # Visualize
            visualize_navigation_plan(
                scene=self.scene,
                robot=self.robot,
                waypoints=planned_path,
                goal_objects=goal_objects
            )
        else:
            print("✗ Could not plan path to goal(s)")

    def navigate_to_task_goals(self):
        """Plan navigation to all task-relevant objects."""
        print(f"\n{'='*70}")
        print("Navigating to Task Goals")
        print("="*70)

        # Parse task goals
        nav_goals = parse_task_to_navigation_goals(self.task, max_goals=5)

        if not nav_goals:
            print("✗ No navigation goals found in current task")
            return

        print(f"\nTask: {self.config['task']['activity_name']}")
        print(f"Found {len(nav_goals)} navigation goals:")
        for i, goal in enumerate(nav_goals, 1):
            print(f"  {i}. {goal['name']} ({goal['category']}) at {goal['position'][:2]}")

        # Plan path
        waypoint_positions = [goal['position'][:2] for goal in nav_goals]
        planned_path, total_distance = plan_multipoint_path(
            robot=self.robot,
            waypoints=waypoint_positions,
            scene=self.scene
        )

        if planned_path:
            print(f"\n✓ Path planned!")
            print(f"  Waypoints: {len(planned_path)}")
            print(f"  Total distance: {total_distance:.2f} meters")

            visualize_navigation_plan(
                scene=self.scene,
                robot=self.robot,
                waypoints=planned_path,
                goal_objects=[g['object'] for g in nav_goals]
            )
        else:
            print("✗ Could not plan path to goals")

    def change_task(self, task_name: str):
        """
        Load a different BEHAVIOR task.

        Args:
            task_name: Name of the BEHAVIOR activity
        """
        if task_name not in BEHAVIOR_ACTIVITIES:
            print(f"✗ Task '{task_name}' not found in BEHAVIOR activities")
            print(f"Available tasks: {len(BEHAVIOR_ACTIVITIES)}")
            return

        print(f"\nChanging to task: {task_name}")
        print("Reloading environment...")

        # Close current environment
        self.env.close()

        # Update config
        self.config['task']['activity_name'] = task_name

        # Reload
        self.env = og.Environment(configs=self.config)
        self.scene = self.env.scene
        self.robot = self.env.robots[0]
        self.task = self.env.task
        self.env.reset()

        print(f"✓ Task changed to: {task_name}")

    def run_interactive_loop(self):
        """Run the interactive command loop."""
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("="*70)
        print("\nCommands:")
        print("  list                  - List all objects in scene")
        print("  goto <object>         - Navigate to object (e.g., 'goto trash')")
        print("  task                  - Navigate to all task goals")
        print("  change <task_name>    - Load a different BEHAVIOR task")
        print("  help                  - Show this help message")
        print("  quit                  - Exit")
        print("="*70)

        while True:
            try:
                command = input("\n> ").strip().lower()

                if not command:
                    continue

                if command == "quit" or command == "exit":
                    break

                elif command == "list":
                    self.list_available_objects()

                elif command == "task":
                    self.navigate_to_task_goals()

                elif command.startswith("goto "):
                    query = command[5:].strip()
                    self.navigate_to_object(query)

                elif command.startswith("change "):
                    task_name = command[7:].strip()
                    self.change_task(task_name)

                elif command == "help":
                    print("\nCommands:")
                    print("  list                  - List all objects in scene")
                    print("  goto <object>         - Navigate to object")
                    print("  task                  - Navigate to all task goals")
                    print("  change <task_name>    - Load a different BEHAVIOR task")
                    print("  quit                  - Exit")

                else:
                    print(f"✗ Unknown command: '{command}'. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"✗ Error: {e}")

    def close(self):
        """Close the environment."""
        print("\nClosing environment...")
        self.env.close()
        print("✓ Environment closed")


def main():
    """Main function."""
    # Load configuration
    config_path = os.path.join(
        os.path.dirname(__file__), "../../configs/navigation_only_holonomic.yaml"
    )

    # Create interface
    interface = NavigationInterface(config_path)

    # Run interactive loop
    try:
        interface.run_interactive_loop()
    finally:
        interface.close()

    print("\n" + "="*70)
    print("Goodbye!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
