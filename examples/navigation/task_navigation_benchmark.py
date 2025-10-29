"""
Task Navigation Benchmark

Evaluate navigation performance across multiple BEHAVIOR tasks.
Tests path planning from natural language task descriptions to object locations.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/task_navigation_benchmark.py [--num_tasks 10] [--headless]

Author: Auto-generated for path planning research
"""

import os
import sys
import yaml
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import omnigibson as og
from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES

from utils.navigation_utils import (
    parse_task_to_navigation_goals,
    plan_multipoint_path,
    compute_navigation_metrics,
)


def run_navigation_task(config: Dict, activity_name: str, activity_id: int = 0) -> Dict:
    """
    Run navigation planning for a single BEHAVIOR task.

    Args:
        config: Environment configuration dict
        activity_name: Name of BEHAVIOR activity
        activity_id: Activity definition ID

    Returns:
        Dict with task results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Task: {activity_name} (ID: {activity_id})")
    print(f"{'='*60}")

    # Update config for this task
    config['task']['activity_name'] = activity_name
    config['task']['activity_definition_id'] = activity_id

    result = {
        'activity_name': activity_name,
        'activity_id': activity_id,
        'success': False,
        'error': None,
    }

    try:
        # Create environment
        print("  Loading environment...")
        start_time = time.time()
        env = og.Environment(configs=config)
        load_time = time.time() - start_time

        scene = env.scene
        robot = env.robots[0]
        task = env.task

        result['load_time'] = load_time
        result['num_objects'] = len(scene.objects)

        # Reset environment
        env.reset()

        # Parse navigation goals
        print("  Parsing navigation goals...")
        nav_goals = parse_task_to_navigation_goals(task, max_goals=10)

        if len(nav_goals) == 0:
            print("  ⚠ No navigation goals found")
            result['error'] = "No navigation goals"
            env.close()
            return result

        result['num_goals'] = len(nav_goals)
        print(f"  Found {len(nav_goals)} navigation goals")

        # Plan path
        print("  Planning path...")
        waypoint_positions = [goal['position'][:2] for goal in nav_goals]

        plan_start = time.time()
        planned_path, total_distance = plan_multipoint_path(
            robot=robot,
            waypoints=waypoint_positions,
            scene=scene
        )
        plan_time = time.time() - plan_start

        result['planning_time'] = plan_time
        result['num_waypoints'] = len(planned_path)
        result['planned_distance'] = total_distance

        print(f"  ✓ Path planned: {len(planned_path)} waypoints, {total_distance:.2f}m")

        # Compute metrics (without execution)
        goal_positions_3d = [goal['position'] for goal in nav_goals]
        metrics = compute_navigation_metrics(
            robot=robot,
            goal_positions=goal_positions_3d,
            scene=scene,
            path_taken=None
        )

        result.update(metrics)
        result['success'] = True

        # Close environment
        env.close()

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        result['error'] = str(e)

    return result


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="BEHAVIOR Task Navigation Benchmark")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks to benchmark")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--config", type=str, default="navigation_only_holonomic.yaml",
                       help="Config file to use")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results (default: auto-generated)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("BEHAVIOR TASK NAVIGATION BENCHMARK")
    print("="*70)
    print(f"  Tasks to evaluate: {args.num_tasks}")
    print(f"  Config: {args.config}")
    print(f"  Headless: {args.headless}")
    print("="*70 + "\n")

    # Load configuration
    config_path = os.path.join(
        os.path.dirname(__file__), "../../configs", args.config
    )

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set headless mode
    if args.headless:
        config['render']['viewer_width'] = 128
        config['render']['viewer_height'] = 128

    # Select tasks to benchmark
    # Use a diverse subset of BEHAVIOR activities
    selected_tasks = BEHAVIOR_ACTIVITIES[:args.num_tasks]

    print(f"Selected tasks: {', '.join(selected_tasks[:5])}...")
    print()

    # Run benchmark
    results = []
    successful_tasks = 0
    total_planning_time = 0.0
    total_planned_distance = 0.0

    for i, task_name in enumerate(selected_tasks, 1):
        print(f"\n[{i}/{len(selected_tasks)}] Benchmarking: {task_name}")

        result = run_navigation_task(config, task_name, activity_id=0)
        results.append(result)

        if result['success']:
            successful_tasks += 1
            total_planning_time += result.get('planning_time', 0)
            total_planned_distance += result.get('planned_distance', 0)

    # Compute aggregate statistics
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)

    print(f"\nOverall:")
    print(f"  Tasks attempted: {len(results)}")
    print(f"  Tasks successful: {successful_tasks}/{len(results)} ({100*successful_tasks/len(results):.1f}%)")

    if successful_tasks > 0:
        avg_planning_time = total_planning_time / successful_tasks
        avg_distance = total_planned_distance / successful_tasks

        print(f"\nPlanning Performance:")
        print(f"  Average planning time: {avg_planning_time:.3f}s")
        print(f"  Average path length: {avg_distance:.2f}m")

        # Count goals
        total_goals = sum(r.get('num_goals', 0) for r in results if r['success'])
        avg_goals = total_goals / successful_tasks

        print(f"\nGoal Statistics:")
        print(f"  Average goals per task: {avg_goals:.1f}")

    # Failed tasks
    failed_tasks = [r for r in results if not r['success']]
    if failed_tasks:
        print(f"\nFailed Tasks ({len(failed_tasks)}):")
        for result in failed_tasks[:5]:  # Show first 5
            print(f"  - {result['activity_name']}: {result.get('error', 'Unknown error')}")

    # Save results
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"navigation_benchmark_{timestamp}.json"

    output_path = os.path.join(os.path.dirname(__file__), "../../results", output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            'benchmark_info': {
                'num_tasks': len(results),
                'successful_tasks': successful_tasks,
                'config_file': args.config,
                'timestamp': datetime.now().isoformat(),
            },
            'summary': {
                'success_rate': successful_tasks / len(results) if results else 0,
                'avg_planning_time': avg_planning_time if successful_tasks > 0 else None,
                'avg_path_length': avg_distance if successful_tasks > 0 else None,
            },
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
