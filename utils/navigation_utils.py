"""
Navigation Utilities for Language-Conditioned Path Planning

This module provides helper functions for extracting navigation goals from BEHAVIOR tasks,
getting ground truth object positions, planning multi-waypoint paths, and computing metrics.

Author: Auto-generated for path planning research
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import omnigibson as og
from omnigibson.object_states import Pose
from omnigibson.tasks.behavior_task import BehaviorTask


def parse_task_to_navigation_goals(task: BehaviorTask, max_goals: int = 10) -> List[Dict]:
    """
    Extract relevant object locations from a BEHAVIOR task for navigation.

    Args:
        task: BehaviorTask instance
        max_goals: Maximum number of goal objects to return

    Returns:
        List of dicts with keys:
            - 'name': str, object instance name (e.g., "apple.n.01_1")
            - 'category': str, object category (e.g., "apple")
            - 'position': np.ndarray, [x, y, z] world position
            - 'synset': str, synset name (e.g., "apple.n.01")
            - 'exists': bool, whether object exists in scene
    """
    navigation_goals = []

    # Iterate through task-relevant objects
    for obj_instance, bddl_entity in task.object_scope.items():
        # Skip if this is the agent
        if "agent" in obj_instance.lower():
            continue

        # Skip if object doesn't exist
        if not bddl_entity.exists:
            continue

        # Get ground truth position
        obj = bddl_entity.wrapped_obj
        position, orientation = obj.states[Pose].get_value()

        # Extract category and synset
        synset = bddl_entity.synset
        categories = bddl_entity.og_categories
        category = categories[0] if categories else synset

        goal_info = {
            'name': obj_instance,
            'category': category,
            'position': position.cpu().numpy() if hasattr(position, 'cpu') else position,
            'synset': synset,
            'exists': True,
            'object': obj
        }

        navigation_goals.append(goal_info)

        if len(navigation_goals) >= max_goals:
            break

    return navigation_goals


def get_ground_truth_object_positions(scene, object_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Get ground truth positions of objects in the scene without perception.

    Args:
        scene: OmniGibson scene instance
        object_names: Optional list of specific object names. If None, returns all objects.

    Returns:
        Dict mapping object names to [x, y, z] positions
    """
    positions = {}

    # Get all objects in scene
    objects = scene.objects if object_names is None else [
        scene.object_registry("name", name) for name in object_names
    ]

    for obj in objects:
        if obj is None:
            continue
        position, _ = obj.states[Pose].get_value()
        positions[obj.name] = position.cpu().numpy() if hasattr(position, 'cpu') else position

    return positions


def plan_multipoint_path(robot, waypoints: List[np.ndarray], scene) -> Tuple[List[np.ndarray], float]:
    """
    Plan a path through multiple waypoints using scene traversability map.

    Args:
        robot: Robot instance
        waypoints: List of [x, y] or [x, y, z] waypoint positions
        scene: Scene instance with traversability map

    Returns:
        Tuple of:
            - List of [x, y] waypoints forming the complete path
            - Total geodesic distance
    """
    if len(waypoints) == 0:
        return [], 0.0

    complete_path = []
    total_distance = 0.0

    # Get current robot position
    robot_pos, _ = robot.states[Pose].get_value()
    robot_pos = robot_pos.cpu().numpy() if hasattr(robot_pos, 'cpu') else robot_pos
    current_pos = robot_pos[:2]  # Take x, y only

    # Plan path through each waypoint sequentially
    for waypoint in waypoints:
        waypoint_2d = waypoint[:2]  # Extract x, y

        # Plan path from current position to waypoint
        path, distance = scene.get_shortest_path(
            floor=0,
            source_world=current_pos,
            target_world=waypoint_2d,
            entire_path=True,
            robot=robot
        )

        if path is not None and len(path) > 0:
            complete_path.extend(path)
            total_distance += distance
            current_pos = waypoint_2d
        else:
            print(f"Warning: Could not find path to waypoint {waypoint_2d}")

    return complete_path, total_distance


def language_to_objects(task_description: str, object_scope: Dict, fuzzy_match: bool = True) -> List[str]:
    """
    Map natural language description to object instances in the task scope.

    Args:
        task_description: Natural language string (e.g., "trash", "refrigerator")
        object_scope: Dict from BehaviorTask.object_scope
        fuzzy_match: If True, use substring matching; if False, require exact match

    Returns:
        List of matching object instance names
    """
    task_description_lower = task_description.lower()
    matching_objects = []

    for obj_instance, bddl_entity in object_scope.items():
        if not bddl_entity.exists:
            continue

        # Check if description matches synset, category, or instance name
        synset = bddl_entity.synset.lower()
        categories = [cat.lower() for cat in bddl_entity.og_categories]
        instance_lower = obj_instance.lower()

        if fuzzy_match:
            # Substring matching
            if (task_description_lower in synset or
                task_description_lower in instance_lower or
                any(task_description_lower in cat for cat in categories)):
                matching_objects.append(obj_instance)
        else:
            # Exact matching
            if (task_description_lower == synset or
                task_description_lower == instance_lower or
                task_description_lower in categories):
                matching_objects.append(obj_instance)

    return matching_objects


def compute_navigation_metrics(
    robot,
    goal_positions: List[np.ndarray],
    scene,
    path_taken: Optional[List[np.ndarray]] = None
) -> Dict[str, float]:
    """
    Compute navigation performance metrics.

    Args:
        robot: Robot instance
        goal_positions: List of [x, y, z] goal positions
        scene: Scene instance
        path_taken: Optional list of [x, y] positions the robot actually traversed

    Returns:
        Dict with metrics:
            - 'distance_to_nearest_goal': Distance to closest goal
            - 'goals_reached': Number of goals within tolerance (0.5m)
            - 'optimal_path_length': Geodesic distance for optimal path
            - 'path_efficiency': Ratio of optimal to actual path length (if path_taken provided)
            - 'success': Boolean, whether at least one goal was reached
    """
    robot_pos, _ = robot.states[Pose].get_value()
    robot_pos = robot_pos.cpu().numpy() if hasattr(robot_pos, 'cpu') else robot_pos
    robot_pos_2d = robot_pos[:2]

    # Distance to nearest goal
    goal_distances = []
    for goal in goal_positions:
        goal_2d = goal[:2]
        dist = np.linalg.norm(robot_pos_2d - goal_2d)
        goal_distances.append(dist)

    min_distance = min(goal_distances) if goal_distances else float('inf')

    # Count goals reached (within 0.5m tolerance)
    tolerance = 0.5
    goals_reached = sum(1 for d in goal_distances if d < tolerance)

    # Compute optimal path length
    optimal_distance = 0.0
    if len(goal_positions) > 0:
        current_pos = robot_pos_2d
        for goal in goal_positions:
            _, dist = scene.get_shortest_path(
                floor=0,
                source_world=current_pos,
                target_world=goal[:2],
                entire_path=False,
                robot=robot
            )
            optimal_distance += dist
            current_pos = goal[:2]

    metrics = {
        'distance_to_nearest_goal': min_distance,
        'goals_reached': goals_reached,
        'total_goals': len(goal_positions),
        'optimal_path_length': optimal_distance,
        'success': goals_reached > 0
    }

    # Compute path efficiency if actual path provided
    if path_taken is not None and len(path_taken) > 1:
        actual_length = sum(
            np.linalg.norm(np.array(path_taken[i]) - np.array(path_taken[i-1]))
            for i in range(1, len(path_taken))
        )
        metrics['actual_path_length'] = actual_length
        metrics['path_efficiency'] = optimal_distance / actual_length if actual_length > 0 else 0.0

    return metrics


def extract_room_goals(task: BehaviorTask) -> Dict[str, List[str]]:
    """
    Extract room-based navigation goals from BEHAVIOR task.

    Args:
        task: BehaviorTask instance

    Returns:
        Dict mapping room types (e.g., "kitchen") to list of object names in that room
    """
    room_goals = {}
    scene = task.env.scene

    # Get all task objects
    for obj_instance, bddl_entity in task.object_scope.items():
        if not bddl_entity.exists or "agent" in obj_instance.lower():
            continue

        obj = bddl_entity.wrapped_obj

        # Get room(s) this object is in
        if hasattr(obj, 'in_rooms') and obj.in_rooms is not None:
            for room_type in obj.in_rooms:
                if room_type not in room_goals:
                    room_goals[room_type] = []
                room_goals[room_type].append(obj_instance)

    return room_goals


def get_navigable_positions_near_object(
    obj,
    scene,
    robot,
    num_samples: int = 8,
    distance_range: Tuple[float, float] = (0.5, 1.5)
) -> List[np.ndarray]:
    """
    Sample navigable positions around an object.

    Useful for finding valid navigation goals near objects that may not be
    directly reachable (e.g., on tables, in containers).

    Args:
        obj: Target object
        scene: Scene instance
        robot: Robot instance
        num_samples: Number of positions to sample around object
        distance_range: (min, max) distance from object center

    Returns:
        List of [x, y] navigable positions near the object
    """
    obj_pos, _ = obj.states[Pose].get_value()
    obj_pos = obj_pos.cpu().numpy() if hasattr(obj_pos, 'cpu') else obj_pos
    obj_pos_2d = obj_pos[:2]

    navigable_positions = []
    min_dist, max_dist = distance_range

    # Sample positions in a circle around the object
    for i in range(num_samples):
        angle = 2 * np.pi * i / num_samples

        for radius in np.linspace(min_dist, max_dist, 5):
            candidate_pos = obj_pos_2d + radius * np.array([np.cos(angle), np.sin(angle)])

            # Check if position is navigable
            floor, sampled_pos = scene.get_random_point(
                floor=0,
                reference_point=candidate_pos,
                robot=robot
            )

            if sampled_pos is not None:
                # Check if sampled position is close enough to candidate
                if np.linalg.norm(sampled_pos - candidate_pos) < 0.3:
                    navigable_positions.append(sampled_pos)
                    break

    return navigable_positions


def visualize_navigation_plan(scene, robot, waypoints: List[np.ndarray], goal_objects: List = None):
    """
    Visualize planned navigation path and goals in the scene.

    Note: This requires OmniGibson visualization to be enabled.

    Args:
        scene: Scene instance
        robot: Robot instance
        waypoints: List of [x, y] waypoint positions
        goal_objects: Optional list of goal objects to highlight
    """
    # This is a placeholder for visualization logic
    # In practice, you might use OmniGibson's built-in visualization tools
    # or create custom visualization spheres/lines

    print(f"\n=== Navigation Plan ===")
    print(f"Robot position: {robot.states[Pose].get_value()[0][:2]}")
    print(f"Number of waypoints: {len(waypoints)}")
    if waypoints:
        print(f"First waypoint: {waypoints[0]}")
        print(f"Final waypoint: {waypoints[-1]}")
    if goal_objects:
        print(f"Goal objects: {[obj.name if hasattr(obj, 'name') else str(obj) for obj in goal_objects]}")
    print("=" * 25)
