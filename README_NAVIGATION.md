# Language-Conditioned Navigation Framework

**Path Planning Research with BEHAVIOR-1K (No Manipulation or Perception Required)**

This framework enables you to use BEHAVIOR-1K's 1018 diverse household tasks for **pure navigation and path planning research**, with all perception and manipulation disabled.

---

## 🚀 Quick Start

### 1. Test the Interactive Demo

```bash
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/language_nav_demo.py
```

### 2. Try Interactive Navigation

```bash
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/interactive_nav_interface.py
```

Then try commands like:
- `list` - Show all objects
- `goto trash` - Navigate to trash
- `task` - Navigate to task goals

### 3. Run a Benchmark

```bash
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/task_navigation_benchmark.py --num_tasks 5
```

---

## 📁 What Was Created

```
BEHAVIOR-1K/
├── configs/
│   ├── navigation_only_holonomic.yaml      # R1Pro base-only config
│   └── navigation_only_differential.yaml   # Turtlebot base-only config
│
├── utils/
│   └── navigation_utils.py                 # Core navigation utilities
│
├── examples/navigation/
│   ├── language_nav_demo.py                # Basic demo
│   ├── task_navigation_benchmark.py        # Multi-task evaluation
│   └── interactive_nav_interface.py        # Interactive CLI
│
├── training/
│   └── nav_policy_trainer.py               # Gym environment for RL
│
├── docs/
│   └── NAVIGATION_GUIDE.md                 # Complete documentation
│
└── results/                                # Benchmark outputs (auto-created)
```

---

## 🎯 What This Framework Does

### Natural Language → Navigation Goals

```python
Task: "picking up trash"
  ↓
Extract: trash.n.01_1 at [2.5, 3.1, 0.0]
         trash_can.n.01_1 at [1.2, 4.5, 0.0]
  ↓
Plan: A* path through waypoints
  ↓
Execute: Your path planning algorithm
```

### Key Features

✅ **1018 diverse tasks** from BEHAVIOR dataset
✅ **Ground truth object positions** (no perception)
✅ **Multiple robot platforms** (holonomic & differential)
✅ **Built-in A* path planning** on traversability maps
✅ **Gym-compatible training** environment
✅ **Benchmark tools** for multi-task evaluation

---

## 📖 Core API

Located in `utils/navigation_utils.py`:

```python
from utils.navigation_utils import (
    parse_task_to_navigation_goals,    # Task → goal positions
    plan_multipoint_path,               # Plan path through waypoints
    language_to_objects,                # "trash" → object instances
    compute_navigation_metrics,         # Evaluate performance
)

# Example usage
nav_goals = parse_task_to_navigation_goals(env.task)
# Returns: [{'name': 'trash.n.01_1', 'position': [x,y,z], ...}, ...]

waypoints = [goal['position'][:2] for goal in nav_goals]
path, distance = plan_multipoint_path(robot, waypoints, scene)
# Returns: (path_waypoints, total_distance)
```

---

## 🔧 Configuration Files

### Turtlebot with Differential Drive

Both config files use Turtlebot for simplicity (navigation-only robot):

```yaml
# configs/navigation_only_holonomic.yaml & navigation_only_differential.yaml
robots:
  - type: Turtlebot
    obs_modalities: []          # NO PERCEPTION
    controller_config:
      base:
        name: DifferentialDriveController
        # Actions: [left_wheel_vel, right_wheel_vel]
```

**Note**: Despite the filename "holonomic", both configs currently use Turtlebot with differential drive for simplicity. You can replace with other robots like Fetch or Husky if needed for true holonomic control.

---

## 🤖 Training Navigation Policies

The framework includes a Gym-compatible environment for RL:

```python
from training.nav_policy_trainer import LanguageNavigationEnv

env = LanguageNavigationEnv(
    config_path="configs/navigation_only_holonomic.yaml",
    max_steps=500
)

obs, info = env.reset()
# obs: [robot_x, robot_y, theta, goal_x, goal_y, distance, bearing]

action = [0.5, 0.0, 0.1]  # [vx, vy, omega]
obs, reward, done, truncated, info = env.step(action)
```

### Train with Random Policy (Baseline)

```bash
OMNI_KIT_ACCEPT_EULA=YES python training/nav_policy_trainer.py --algorithm random --episodes 10
```

### Train with PPO

```bash
pip install stable-baselines3
OMNI_KIT_ACCEPT_EULA=YES python training/nav_policy_trainer.py --algorithm ppo --episodes 100
```

---

## 📊 Benchmarking

Evaluate your planner across multiple tasks:

```bash
# Basic benchmark
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/task_navigation_benchmark.py --num_tasks 20

# Headless mode (faster)
OMNI_KIT_ACCEPT_EULA=YES python examples/navigation/task_navigation_benchmark.py \
    --num_tasks 50 --headless --output my_results.json
```

Output includes:
- Success rate
- Average path length
- Planning time
- Per-task metrics

---

## 📚 Documentation

**Complete guide**: `docs/NAVIGATION_GUIDE.md`

Includes:
- Detailed API reference
- Advanced usage examples
- Troubleshooting
- Custom planner integration

---

## 💡 Use Cases

This framework is ideal for:

1. **Path Planning Algorithms**: Test A*, RRT, PRM, etc. on realistic household environments
2. **Learning-Based Navigation**: Train RL policies with language conditioning
3. **Multi-Goal Planning**: Navigate to sequences of objects from task descriptions
4. **Semantic Navigation**: "Go to the kitchen" → navigate to kitchen objects
5. **Benchmarking**: Compare planners across 1000+ diverse scenarios

---

## 🔍 Example: Implementing Your Own Planner

```python
import omnigibson as og
from utils.navigation_utils import parse_task_to_navigation_goals

# Load environment
env = og.Environment(configs=config)

# Get navigation goals from language task
nav_goals = parse_task_to_navigation_goals(env.task)

# Your custom planner
def my_planner(start, goals, scene):
    # Access traversability map
    trav_map = scene.trav_map

    # Your algorithm here
    path = my_algorithm(start, goals, trav_map)

    return path

# Use it
robot_pos = env.robots[0].states[Pose].get_value()[0]
waypoints = [g['position'][:2] for g in nav_goals]
path = my_planner(robot_pos[:2], waypoints, env.scene)

# Execute navigation
for waypoint in path:
    # Your controller here
    pass
```

---

## ❓ FAQ

**Q: Do I need perception?**
A: No! All object positions are ground truth via `object.states[Pose].get_value()`

**Q: Do I need manipulation?**
A: No! Configs use base controller only. No arm, gripper, or grasping.

**Q: Can I use my own planner?**
A: Yes! Just replace the `plan_multipoint_path` call with your algorithm.

**Q: How many tasks are available?**
A: 1018 diverse household tasks from BEHAVIOR-1K dataset.

**Q: Can I train RL policies?**
A: Yes! See `training/nav_policy_trainer.py` for Gym environment.

---

## 🐛 Troubleshooting

### EULA Error
```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

### Import Error
Run from repository root:
```bash
cd /path/to/BEHAVIOR-1K
python examples/navigation/language_nav_demo.py
```

### No Navigation Goals
Some tasks are manipulation-focused. Try: `setting_table`, `picking_up_trash`, `cleaning_kitchen`

---

## 📄 Citation

```bibtex
@article{behavior1k,
  title={BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation},
  author={Li, Chengshu and Mart{\'\i}n-Mart{\'\i}n, Roberto and others},
  journal={CoRL},
  year={2022}
}
```

---

## 🎓 Next Steps

1. **Read the guide**: `docs/NAVIGATION_GUIDE.md`
2. **Run the demo**: `examples/navigation/language_nav_demo.py`
3. **Try benchmarking**: Evaluate across multiple tasks
4. **Implement your planner**: Replace built-in A* with your algorithm
5. **Train a policy**: Use the Gym environment for RL

---

**Ready to start? Run the quick start commands above!** 🚀
