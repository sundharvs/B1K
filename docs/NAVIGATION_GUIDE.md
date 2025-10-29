# Language-Conditioned Navigation with BEHAVIOR-1K

**Guide for Path Planning Research using BEHAVIOR-1K without Manipulation or Perception**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [System Overview](#system-overview)
4. [Configuration Files](#configuration-files)
5. [Navigation Utilities API](#navigation-utilities-api)
6. [Example Scripts](#example-scripts)
7. [Training Framework](#training-framework)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

This framework enables path planning research using BEHAVIOR-1K's rich semantic environments and natural language task descriptions, **without requiring perception or manipulation**. All object positions are accessed via ground truth, allowing you to focus purely on navigation and planning algorithms.

### Key Features

- ✅ **Language-conditioned navigation**: Natural language tasks → spatial goals
- ✅ **Ground truth object positions**: No perception needed
- ✅ **Multi-waypoint path planning**: Built-in A* on traversability maps
- ✅ **Multiple robot platforms**: Holonomic and differential drive options
- ✅ **1018 diverse tasks**: From BEHAVIOR dataset
- ✅ **Gym-compatible training**: RL-ready environment wrapper
- ✅ **Benchmark tools**: Evaluate across multiple tasks

---

## Quick Start

### 1. Run the Interactive Demo

```bash
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/language_nav_demo.py
```

This will:
- Load a BEHAVIOR task (default: "picking_up_trash")
- Extract navigation goals from the task
- Plan a path through multiple waypoints
- Simulate basic navigation

### 2. Try the Interactive Interface

```bash
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/interactive_nav_interface.py
```

Commands you can try:
```
> list                    # List all objects in scene
> goto trash              # Navigate to trash objects
> task                    # Navigate to all task goals
> change setting_table    # Load a different task
> quit                    # Exit
```

### 3. Run a Benchmark

```bash
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/task_navigation_benchmark.py --num_tasks 5
```

This evaluates navigation performance across 5 different BEHAVIOR tasks.

---

## System Overview

### Architecture

```
Natural Language Task
        ↓
  Parse Task Goals
        ↓
Ground Truth Object Positions
        ↓
  Plan Multi-Waypoint Path
        ↓
    Execute Navigation
```

### Workflow Example

```python
import omnigibson as og
from utils.navigation_utils import (
    parse_task_to_navigation_goals,
    plan_multipoint_path
)

# 1. Load environment with BEHAVIOR task
env = og.Environment(configs=config)

# 2. Parse language task → navigation goals (ground truth!)
nav_goals = parse_task_to_navigation_goals(env.task)

# 3. Plan path through goals
waypoints = [goal['position'][:2] for goal in nav_goals]
path, distance = plan_multipoint_path(
    robot=env.robots[0],
    waypoints=waypoints,
    scene=env.scene
)

# 4. Execute navigation (your algorithm here!)
for waypoint in path:
    # ... your path following controller ...
    pass
```

---

## Configuration Files

### Navigation-Only Configurations

Two pre-configured files are provided in `configs/`:

#### 1. `navigation_only_holonomic.yaml`

- **Robot**: R1Pro with holonomic base
- **Control**: [vx, vy, omega] velocity commands
- **Perception**: DISABLED (no sensors)
- **Manipulation**: DISABLED (base only)
- **Use case**: Flexible path planning research

```yaml
robots:
  - type: R1Pro
    obs_modalities: []  # No perception!
    controller_config:
      base:
        name: HolonomicBaseJointController
        # Only base controller - navigation only
```

#### 2. `navigation_only_differential.yaml`

- **Robot**: Turtlebot
- **Control**: [v_left, v_right] wheel velocities
- **Perception**: DISABLED
- **Use case**: Non-holonomic planning

### Customizing Configurations

To change the task:

```python
config['task']['activity_name'] = 'setting_table'
config['task']['activity_definition_id'] = 0
```

To change the scene:

```python
config['scene']['scene_model'] = 'Rs_int'  # or 'house_double_floor_lower'
config['scene']['load_room_types'] = ['kitchen', 'living_room']
```

---

## Navigation Utilities API

Located in `utils/navigation_utils.py`

### Core Functions

#### `parse_task_to_navigation_goals(task, max_goals=10)`

Extract navigation-relevant objects from a BEHAVIOR task.

**Returns**: List of dicts with:
- `name`: Object instance name
- `category`: Object category
- `position`: [x, y, z] world position (ground truth!)
- `synset`: Synset name
- `exists`: Whether object exists

**Example**:
```python
nav_goals = parse_task_to_navigation_goals(env.task, max_goals=5)
for goal in nav_goals:
    print(f"{goal['name']}: {goal['position']}")
```

#### `plan_multipoint_path(robot, waypoints, scene)`

Plan a path through multiple waypoints using A*.

**Args**:
- `robot`: Robot instance
- `waypoints`: List of [x, y] positions
- `scene`: Scene instance

**Returns**: Tuple of (path, total_distance)

**Example**:
```python
waypoints = [[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]]
path, distance = plan_multipoint_path(robot, waypoints, scene)
print(f"Path has {len(path)} waypoints, {distance:.2f}m total")
```

#### `get_ground_truth_object_positions(scene, object_names=None)`

Get positions of all objects without perception.

**Returns**: Dict mapping object names → [x, y, z] positions

**Example**:
```python
positions = get_ground_truth_object_positions(scene)
print(f"Refrigerator at: {positions['refrigerator_0']}")
```

#### `language_to_objects(task_description, object_scope, fuzzy_match=True)`

Map natural language to object instances.

**Args**:
- `task_description`: e.g., "trash", "refrigerator"
- `object_scope`: From `task.object_scope`
- `fuzzy_match`: Use substring matching

**Returns**: List of matching object instance names

**Example**:
```python
trash_objects = language_to_objects("trash", task.object_scope)
# Returns: ['trash.n.01_1', 'trash.n.01_2', ...]
```

#### `compute_navigation_metrics(robot, goal_positions, scene, path_taken=None)`

Compute navigation performance metrics.

**Returns**: Dict with:
- `distance_to_nearest_goal`: Current distance to closest goal
- `goals_reached`: Number of goals within tolerance
- `optimal_path_length`: Geodesic distance for optimal path
- `path_efficiency`: Ratio of optimal to actual (if path provided)
- `success`: Boolean

**Example**:
```python
metrics = compute_navigation_metrics(
    robot=robot,
    goal_positions=goal_positions,
    scene=scene,
    path_taken=recorded_path
)
print(f"Success: {metrics['success']}")
print(f"Efficiency: {metrics['path_efficiency']:.2%}")
```

---

## Example Scripts

### 1. Language Navigation Demo

**File**: `examples/navigation/language_nav_demo.py`

Simple demonstration of the complete pipeline:
1. Load BEHAVIOR task
2. Parse language → goals
3. Plan path
4. Simulate navigation
5. Compute metrics

**Run**:
```bash
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/language_nav_demo.py
```

### 2. Task Navigation Benchmark

**File**: `examples/navigation/task_navigation_benchmark.py`

Evaluate navigation across multiple BEHAVIOR tasks.

**Run**:
```bash
# Benchmark 10 tasks
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/task_navigation_benchmark.py --num_tasks 10

# Headless mode for faster evaluation
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/task_navigation_benchmark.py \
    --num_tasks 20 --headless --output my_results.json
```

**Output**: JSON file in `results/` with per-task metrics and summary statistics.

### 3. Interactive Navigation Interface

**File**: `examples/navigation/interactive_nav_interface.py`

Command-line interface for exploring navigation.

**Run**:
```bash
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/interactive_nav_interface.py
```

**Commands**:
- `list` - Show all objects in scene
- `goto <object>` - Navigate to specific object
- `task` - Navigate to all task goals
- `change <task_name>` - Load different task

---

## Training Framework

### Gym Environment

**File**: `training/nav_policy_trainer.py`

A Gym-compatible environment for training navigation policies.

#### Observation Space

Vector of 7 values:
```python
[robot_x, robot_y, robot_theta,  # Robot pose
 goal_x, goal_y,                  # Current goal position
 goal_distance,                   # Distance to goal
 goal_bearing]                    # Angle to goal
```

#### Action Space

- **Holonomic**: `[vx, vy, omega]` (3D)
- **Differential**: `[v_left, v_right]` (2D)

#### Reward Function

```python
reward = -goal_distance * 0.1  # Distance penalty
         -0.01                  # Time penalty
         +10.0                  # Goal reached bonus
         +100.0                 # All goals reached bonus
```

### Training a Random Policy (Baseline)

```bash
OMNI_KIT_ACCEPT_EULA=YES python training/nav_policy_trainer.py \
    --algorithm random --episodes 10
```

### Training with PPO (Requires stable-baselines3)

```bash
# Install dependencies
pip install stable-baselines3

# Train
OMNI_KIT_ACCEPT_EULA=YES python training/nav_policy_trainer.py \
    --algorithm ppo --episodes 100
```

### Using the Environment Directly

```python
from training.nav_policy_trainer import LanguageNavigationEnv

env = LanguageNavigationEnv(
    config_path="configs/navigation_only_holonomic.yaml",
    max_steps=500
)

# Standard Gym interface
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
```

---

## Advanced Usage

### Implementing Your Own Planner

Replace the built-in A* with your algorithm:

```python
def my_planner(robot, goal, scene):
    """Your custom path planning algorithm."""
    # Access traversability map
    trav_map = scene.trav_map

    # Get robot position
    robot_pos, _ = robot.states[Pose].get_value()
    start = robot_pos.cpu().numpy()[:2]

    # Your planning logic here
    path = my_algorithm(start, goal, trav_map)

    return path

# Use it
path = my_planner(robot, goal_position, scene)
```

### Multi-Room Navigation

```python
from utils.navigation_utils import extract_room_goals

# Get objects organized by room
room_goals = extract_room_goals(task)

# Navigate room by room
for room_type, objects in room_goals.items():
    print(f"Navigating to {len(objects)} objects in {room_type}")
    # ... plan path through objects in this room ...
```

### Accessing Scene Traversability Map

```python
# Get traversability map directly
trav_map = scene.trav_map

# Map properties
n_floors = trav_map.n_floors
resolution = trav_map.trav_map_resolution  # meters per pixel
floor_heights = trav_map.floor_heights

# Get shortest path with more control
path, distance = scene.get_shortest_path(
    floor=0,
    source_world=[x1, y1],
    target_world=[x2, y2],
    entire_path=True,
    robot=robot
)
```

### Changing Tasks Dynamically

```python
from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES

# List all available tasks
print(f"Total tasks: {len(BEHAVIOR_ACTIVITIES)}")
print(BEHAVIOR_ACTIVITIES[:10])

# Change task
env.close()
config['task']['activity_name'] = 'setting_table'
env = og.Environment(configs=config)
```

---

## Troubleshooting

### Issue: "EULA not accepted"

**Solution**: Set environment variable:
```bash
export OMNI_KIT_ACCEPT_EULA=YES
# or prefix command with:
OMNI_KIT_ACCEPT_EULA=YES python script.py
```

### Issue: "No navigation goals found"

**Cause**: Some BEHAVIOR tasks have primarily manipulation goals, not spatial navigation.

**Solution**: Try a different task or extract objects manually:
```python
# Get all task objects regardless
for obj_inst, bddl_entity in task.object_scope.items():
    if bddl_entity.exists:
        pos, _ = bddl_entity.wrapped_obj.states[Pose].get_value()
        print(f"{obj_inst}: {pos}")
```

### Issue: "Path planning fails"

**Possible causes**:
1. Goal is not reachable (on table, in container, etc.)
2. Goal is outside traversable area

**Solution**: Use `get_navigable_positions_near_object`:
```python
from utils.navigation_utils import get_navigable_positions_near_object

nav_positions = get_navigable_positions_near_object(
    obj=target_object,
    scene=scene,
    robot=robot,
    distance_range=(0.5, 1.5)  # meters from object
)
```

### Issue: Scene loading is slow

**Solution**: Use smaller scenes or disable object loading:
```yaml
scene:
  scene_model: Rs_int  # Smaller scene
  load_room_types: ['kitchen']  # Load only needed rooms
```

### Issue: ImportError for navigation_utils

**Solution**: Make sure you're running from the repository root:
```bash
cd /path/to/BEHAVIOR-1K
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/language_nav_demo.py
```

---

## Available BEHAVIOR Tasks

Example tasks particularly suitable for navigation research:

- `picking_up_trash` - Navigate to trash, then trash can
- `setting_table` - Multi-object table setting
- `cleaning_kitchen` - Navigate through kitchen
- `folding_clothes` - Closet/laundry navigation
- `watering_plants` - Multi-room plant locations

**Full list**:
```python
from omnigibson.utils.bddl_utils import BEHAVIOR_ACTIVITIES
print(BEHAVIOR_ACTIVITIES)  # 1018 tasks!
```

---

## Citation

If you use this framework for your research, please cite BEHAVIOR-1K:

```bibtex
@article{behavior1k,
  title={BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
  author={Li, Chengshu and Mart{\'\i}n-Mart{\'\i}n, Roberto and others},
  journal={CoRL},
  year={2022}
}
```

---

## Support

- **Issues**: Report bugs at https://github.com/StanfordVL/BEHAVIOR-1K/issues
- **Documentation**: https://behavior.stanford.edu/
- **OmniGibson Docs**: https://behavior.stanford.edu/omnigibson/

---

**Happy Path Planning! 🤖**
