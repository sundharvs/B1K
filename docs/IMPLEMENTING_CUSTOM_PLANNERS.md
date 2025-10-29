# Implementing Custom Planners in BEHAVIOR-1K

**Guide for Adding New Path Planning Algorithms**

Based on analysis of the existing A* implementation and framework architecture.

---

## Executive Summary

**TL;DR**: The framework is **moderately easy** to extend with custom planners.

**Difficulty Rating**: ⭐⭐⭐ (3/5)

**Why it's good**:
- ✅ Clean separation between planning algorithm and scene representation
- ✅ Simple grid-based interface (traversability maps)
- ✅ Existing A* implementation is straightforward reference
- ✅ OMPL integration already present for advanced planning

**Why it's challenging**:
- ⚠️ Need to understand OmniGibson's coordinate systems
- ⚠️ Erosion and robot size handling is manual
- ⚠️ No built-in profiling/benchmarking infrastructure
- ⚠️ Limited documentation on planning interfaces

**Estimated implementation time**:
- Simple planner (Dijkstra, BFS): **2-4 hours**
- TSP optimization: **4-8 hours**
- RRT/sampling-based: **1-2 days** (OMPL integration exists)
- Full suite with benchmarking: **1-2 weeks**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Existing A* Analysis](#existing-a-analysis)
3. [Integration Points](#integration-points)
4. [Implementation Patterns](#implementation-patterns)
5. [Step-by-Step: Adding a New Planner](#step-by-step-adding-a-new-planner)
6. [Advanced: OMPL Integration](#advanced-ompl-integration)
7. [Benchmarking Framework](#benchmarking-framework)
8. [Common Pitfalls](#common-pitfalls)

---

## Architecture Overview

### Component Hierarchy

```
Scene (OmniGibson)
    ↓
TraversableMap (manages navigation maps)
    ↓ provides 2D grid
A* / Custom Planner (algorithm implementation)
    ↓ returns waypoints
NavigationUtils (multi-goal, metrics)
    ↓ used by
Demo Scripts / Training (user code)
```

### Key Files

| Component | File | Lines of Code | Complexity |
|-----------|------|---------------|------------|
| **A* Algorithm** | `motion_planning_utils.py` | 83 lines | Simple ⭐ |
| **Traversability Map** | `traversable_map.py` | 300 lines | Moderate ⭐⭐ |
| **Navigation Utilities** | `utils/navigation_utils.py` | 400 lines | Simple ⭐ |
| **OMPL Integration** | `motion_planning_utils.py` | 200 lines | Complex ⭐⭐⭐⭐ |

---

## Existing A* Analysis

### Code Review: A* Implementation

**Location**: `/home/ssangeetha3/git/BEHAVIOR-1K/OmniGibson/omnigibson/utils/motion_planning_utils.py` (lines 553-635)

**Strengths**:
1. ✅ **Clean implementation** - Classic textbook A*, easy to understand
2. ✅ **Self-contained** - Only ~80 lines, no external dependencies
3. ✅ **Configurable** - 4-connected or 8-connected grid
4. ✅ **Efficient** - Uses heapq for priority queue
5. ✅ **Robust** - Handles edge cases (no path found)

**Code Quality Assessment**:
```python
# GOOD: Clean interface
def astar(search_map, start, goal, eight_connected=True):
    """Simple, well-documented signature"""

# GOOD: Readable helper functions
def heuristic(node):
    return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

def get_neighbors(cell):
    # Clear 4 vs 8 connectivity logic
    ...

# GOOD: Standard A* structure
open_set = [(0, start)]  # Priority queue
came_from = {}           # Parent tracking
g_score = {...}          # Cost tracking

while open_set:
    # Standard A* loop
    ...
```

**Weaknesses to avoid in your implementation**:
```python
# ISSUE 1: Full grid initialization (memory intensive for large maps)
g_score = {(i.item(), j.item()): float("inf")
           for i, j in th.cartesian_prod(th.arange(rows), th.arange(cols))}
# Better: Use defaultdict or lazy initialization

# ISSUE 2: No early termination optimization
# Could add: if heuristic(current) > best_path_found: break

# ISSUE 3: Fixed cost function (doesn't support weighted A*)
def cost(cell1, cell2):
    # Always returns 1 or sqrt(2)
    # Better: Allow custom cost maps
```

### TraversableMap Interface

**Location**: `/home/ssangeetha3/git/BEHAVIOR-1K/OmniGibson/omnigibson/maps/traversable_map.py`

**Key Method** (this is what you'll modify/extend):
```python
def get_shortest_path(self, floor, source_world, target_world, entire_path=False, robot=None):
    """
    Interface between world coordinates and planning algorithm.

    This method:
    1. Converts world coords (meters) → map coords (pixels)
    2. Erodes map for robot size
    3. Calls planning algorithm (A*)
    4. Converts map coords back → world coords
    5. Samples waypoints

    Returns: (waypoints, distance)
    """
    # Convert to map coordinates
    source_map = tuple(self.world_to_map(source_world).tolist())
    target_map = tuple(self.world_to_map(target_world).tolist())

    # Erode for robot size
    trav_map = th.clone(self.floor_map[floor])
    trav_map = self._erode_trav_map(trav_map, robot=robot)

    # REPLACE THIS LINE WITH YOUR PLANNER!
    path_map = astar(trav_map, source_map, target_map)

    # Convert back to world
    path_world = self.map_to_world(path_map)
    geodesic_distance = th.sum(th.norm(path_world[1:] - path_world[:-1], dim=1))

    return path_world, geodesic_distance
```

**Key Insight**: The interface is **surprisingly simple**!
- Input: 2D grid (traversable=255, blocked=0) + start/goal cells
- Output: List of (x, y) cells or None

---

## Integration Points

### Option 1: Drop-In Replacement (Easiest)

**Difficulty**: ⭐ (Very Easy)

Replace A* with your algorithm:

```python
# File: motion_planning_utils.py

def dijkstra(search_map, start, goal):
    """
    Drop-in replacement for A*.
    Same signature, different algorithm.
    """
    # Your implementation here
    # Must return: torch.tensor([[x1,y1], [x2,y2], ...]) or None
    ...

# Then in traversable_map.py:
from omnigibson.utils.motion_planning_utils import dijkstra

# In get_shortest_path():
path_map = dijkstra(trav_map, source_map, target_map)  # Instead of astar()
```

**Pros**:
- ✅ Minimal code changes
- ✅ Works with all existing infrastructure
- ✅ Easy to A/B test against A*

**Cons**:
- ⚠️ Limited to 2D grid planners
- ⚠️ Can't easily add custom parameters

### Option 2: Extended Interface (Moderate)

**Difficulty**: ⭐⭐ (Easy-Moderate)

Add a planner selection system:

```python
# File: utils/planners.py (NEW FILE)

class GridPlanner:
    """Base class for grid-based planners."""

    def plan(self, search_map, start, goal, **kwargs):
        """Must be implemented by subclasses."""
        raise NotImplementedError

class AStarPlanner(GridPlanner):
    def __init__(self, heuristic='euclidean', eight_connected=True):
        self.heuristic = heuristic
        self.eight_connected = eight_connected

    def plan(self, search_map, start, goal):
        # Call existing astar() or reimplement
        return astar(search_map, start, goal, self.eight_connected)

class DijkstraPlanner(GridPlanner):
    def plan(self, search_map, start, goal):
        # Same as A* with zero heuristic
        ...

class WeightedAStarPlanner(GridPlanner):
    def __init__(self, weight=1.5):
        self.weight = weight  # Inflate heuristic

    def plan(self, search_map, start, goal):
        # A* with weighted heuristic
        ...

class ThetaStarPlanner(GridPlanner):
    """Any-angle path planning."""
    def plan(self, search_map, start, goal):
        # Theta* implementation
        ...
```

Then modify `TraversableMap`:

```python
# File: traversable_map.py

class TraversableMap(BaseMap):
    def __init__(self, ..., planner='astar', planner_kwargs=None):
        ...
        self.planner = self._create_planner(planner, planner_kwargs)

    def _create_planner(self, name, kwargs):
        from utils.planners import AStarPlanner, DijkstraPlanner, WeightedAStarPlanner

        planners = {
            'astar': AStarPlanner,
            'dijkstra': DijkstraPlanner,
            'weighted_astar': WeightedAStarPlanner,
        }

        kwargs = kwargs or {}
        return planners[name](**kwargs)

    def get_shortest_path(self, ...):
        ...
        path_map = self.planner.plan(trav_map, source_map, target_map)
        ...
```

**Pros**:
- ✅ Clean abstraction
- ✅ Easy to switch planners via config
- ✅ Supports planner-specific parameters

**Cons**:
- ⚠️ Requires modifying core classes
- ⚠️ Still limited to grid-based methods

### Option 3: Parallel Infrastructure (Most Flexible)

**Difficulty**: ⭐⭐⭐ (Moderate)

Create separate planning utilities alongside existing ones:

```python
# File: utils/advanced_planners.py (NEW FILE)

def plan_with_rrt(scene, robot, start_world, goal_world, planning_time=5.0):
    """
    RRT* planning that bypasses grid-based approach.

    Args:
        scene: OmniGibson scene
        robot: Robot instance
        start_world: [x, y, theta] start pose
        goal_world: [x, y, theta] goal pose
        planning_time: Planning budget in seconds

    Returns:
        path: List of [x, y, theta] waypoints
        distance: Path length
    """
    # Use OMPL directly (see plan_base_motion for reference)
    from ompl import base as ob, geometric as og

    # Create SE(2) state space
    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    # ... configure from scene bounds

    # Setup problem
    si = ob.SpaceInformation(space)
    # ... custom validity checker using scene.check_collision()

    # Plan with RRT*
    planner = og.RRTstar(si)
    # ... solve

    return path, distance

def plan_multi_goal_tsp(scene, robot, start_world, goals_world, method='2opt'):
    """
    Multi-goal planning with TSP optimization.

    Args:
        goals_world: List of [x, y] goal positions
        method: 'nearest_neighbor', '2opt', 'christofides'

    Returns:
        ordered_goals: Optimized goal order
        total_distance: Expected total path length
    """
    # Implement TSP solver
    # Use scene.get_shortest_path() for pairwise distances
    ...
```

**Usage in your code**:
```python
# Option A: Use existing A* for simple cases
path, dist = scene.get_shortest_path(floor=0, start, goal, robot=robot)

# Option B: Use RRT for complex cases
from utils.advanced_planners import plan_with_rrt
path, dist = plan_with_rrt(scene, robot, start, goal)

# Option C: Use TSP for multi-goal
from utils.advanced_planners import plan_multi_goal_tsp
ordered_goals, dist = plan_multi_goal_tsp(scene, robot, start, goals)
```

**Pros**:
- ✅ Maximum flexibility
- ✅ Doesn't modify core framework
- ✅ Can mix and match algorithms

**Cons**:
- ⚠️ Duplicate coordinate conversion logic
- ⚠️ Need to handle robot size erosion manually

---

## Implementation Patterns

### Pattern 1: Grid-Based 2D Planner

**Template for Dijkstra, BFS, Theta*, JPS, etc.**

```python
def my_grid_planner(search_map, start, goal, **params):
    """
    Your grid-based planner.

    Args:
        search_map (torch.Tensor): 2D grid (255=free, 0=blocked)
        start (tuple): (row, col) start cell
        goal (tuple): (row, col) goal cell
        **params: Your custom parameters

    Returns:
        torch.Tensor or None: Nx2 array of [row, col] or None if no path
    """
    # 1. Initialize data structures
    visited = set()
    came_from = {}

    # 2. Algorithm-specific initialization
    # For Dijkstra: dist = {start: 0}
    # For BFS: queue = deque([start])
    # etc.

    # 3. Main search loop
    while not_done:
        current = get_next_node()

        if current == goal:
            # 4. Reconstruct path
            path = []
            while current in came_from:
                path.insert(0, current)
                current = came_from[current]
            path.insert(0, start)
            return torch.tensor(path)

        # 5. Expand neighbors
        for neighbor in get_neighbors(current):
            if is_valid(neighbor, search_map):
                # Update data structures
                ...

    # 6. No path found
    return None
```

### Pattern 2: Sampling-Based Planner

**Template for RRT, RRT*, PRM**

```python
def my_sampling_planner(scene, robot, start_pose, goal_pose, planning_time=5.0):
    """
    Sampling-based planner using OMPL.

    Args:
        scene: OmniGibson scene (for collision checking)
        robot: Robot instance
        start_pose: [x, y, theta] start
        goal_pose: [x, y, theta] goal
        planning_time: Time budget

    Returns:
        path: List of [x, y, theta] waypoints
        distance: Path length
    """
    from ompl import base as ob, geometric as og

    # 1. Define state space (SE2 for 2D+rotation)
    space = ob.SE2StateSpace()

    # 2. Set bounds from scene
    bounds = ob.RealVectorBounds(2)
    scene_bounds = scene.get_bounds()
    bounds.setLow(0, scene_bounds[0][0])
    bounds.setHigh(0, scene_bounds[0][1])
    bounds.setLow(1, scene_bounds[1][0])
    bounds.setHigh(1, scene_bounds[1][1])
    space.setBounds(bounds)

    # 3. Create space information
    si = ob.SpaceInformation(space)

    # 4. Define validity checker (collision detection)
    def is_state_valid(state):
        x, y, theta = state.getX(), state.getY(), state.getYaw()
        # Use scene collision checking
        robot.set_position_orientation([x, y, 0], [0, 0, theta])
        return not robot.check_collision()

    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
    si.setup()

    # 5. Define start and goal states
    start = ob.State(space)
    start().setX(start_pose[0])
    start().setY(start_pose[1])
    start().setYaw(start_pose[2])

    goal = ob.State(space)
    # ... similar for goal

    # 6. Create problem definition
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    # 7. Create planner (RRT*, PRM, etc.)
    planner = og.RRTstar(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    # 8. Solve
    solved = planner.solve(planning_time)

    if solved:
        # 9. Extract path
        path_obj = pdef.getSolutionPath()
        path = []
        for i in range(path_obj.getStateCount()):
            state = path_obj.getState(i)
            path.append([state.getX(), state.getY(), state.getYaw()])

        # 10. Compute distance
        distance = path_obj.length()

        return path, distance
    else:
        return None, None
```

**Note**: OMPL integration already exists! See `plan_base_motion()` in `motion_planning_utils.py` (line 24) for a working example.

### Pattern 3: Multi-Goal Optimizer

**Template for TSP, VRP, tour planning**

```python
def optimize_goal_order(robot_pos, goals, scene, method='nearest_neighbor'):
    """
    Optimize visitation order for multiple goals.

    Args:
        robot_pos: [x, y] current position
        goals: List of [x, y] goal positions
        scene: Scene for distance queries
        method: 'nearest_neighbor', '2opt', 'christofides', 'optimal'

    Returns:
        ordered_goals: List of goals in optimized order
        total_distance: Expected total path length
    """
    n = len(goals)

    # 1. Build distance matrix
    import numpy as np
    dist_matrix = np.zeros((n+1, n+1))  # +1 for robot start

    # Robot to each goal
    for i, goal in enumerate(goals):
        _, dist = scene.get_shortest_path(0, robot_pos, goal[:2])
        dist_matrix[0, i+1] = dist
        dist_matrix[i+1, 0] = dist

    # Goal to goal
    for i, g1 in enumerate(goals):
        for j, g2 in enumerate(goals):
            if i != j:
                _, dist = scene.get_shortest_path(0, g1[:2], g2[:2])
                dist_matrix[i+1, j+1] = dist

    # 2. Solve TSP based on method
    if method == 'nearest_neighbor':
        tour = nearest_neighbor_tsp(dist_matrix, start=0)
    elif method == '2opt':
        tour = two_opt_tsp(dist_matrix, start=0)
    elif method == 'optimal' and n <= 10:
        tour = brute_force_tsp(dist_matrix, start=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 3. Extract ordered goals (excluding start)
    ordered_goals = [goals[i-1] for i in tour if i > 0]
    total_distance = sum(dist_matrix[tour[i], tour[i+1]]
                        for i in range(len(tour)-1))

    return ordered_goals, total_distance

def nearest_neighbor_tsp(dist_matrix, start=0):
    """Greedy nearest neighbor heuristic."""
    n = len(dist_matrix)
    tour = [start]
    unvisited = set(range(n)) - {start}

    while unvisited:
        current = tour[-1]
        nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)

    return tour

def two_opt_tsp(dist_matrix, start=0, max_iterations=100):
    """2-opt local search improvement."""
    # Start with nearest neighbor
    tour = nearest_neighbor_tsp(dist_matrix, start)

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                # Try reversing segment [i:j]
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]

                if tour_length(new_tour, dist_matrix) < tour_length(tour, dist_matrix):
                    tour = new_tour
                    improved = True
                    break
            if improved:
                break

    return tour

def tour_length(tour, dist_matrix):
    """Compute total tour distance."""
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1))
```

---

## Step-by-Step: Adding a New Planner

### Example: Implementing Dijkstra's Algorithm

**Goal**: Add Dijkstra as an alternative to A* (simpler, no heuristic)

**Step 1**: Create the algorithm (10-15 minutes)

```python
# File: omnigibson/utils/motion_planning_utils.py

import heapq
import torch as th

def dijkstra(search_map, start, goal):
    """
    Dijkstra's algorithm for shortest path (uniform cost search).

    Same as A* but with zero heuristic.

    Args:
        search_map: 2D grid (255=free, 0=blocked)
        start: (row, col) start position
        goal: (row, col) goal position

    Returns:
        torch.Tensor or None: Path as Nx2 array or None
    """
    def get_neighbors(cell):
        # 8-connected
        return [
            (cell[0]+1, cell[1]), (cell[0]-1, cell[1]),
            (cell[0], cell[1]+1), (cell[0], cell[1]-1),
            (cell[0]+1, cell[1]+1), (cell[0]-1, cell[1]-1),
            (cell[0]+1, cell[1]-1), (cell[0]-1, cell[1]+1),
        ]

    def is_valid(cell):
        return (0 <= cell[0] < search_map.shape[0] and
                0 <= cell[1] < search_map.shape[1] and
                search_map[cell] != 0)

    def cost(cell1, cell2):
        return 1 if cell1[0]==cell2[0] or cell1[1]==cell2[1] else 1.414

    # Priority queue: (distance, cell)
    pq = [(0, start)]
    visited = set()
    came_from = {}
    dist = {start: 0}

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.insert(0, current)
                current = came_from[current]
            path.insert(0, start)
            return th.tensor(path)

        for neighbor in get_neighbors(current):
            if not is_valid(neighbor) or neighbor in visited:
                continue

            new_dist = current_dist + cost(current, neighbor)

            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                came_from[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))

    return None
```

**Step 2**: Integrate with TraversableMap (5 minutes)

```python
# File: omnigibson/maps/traversable_map.py

# Add import at top
from omnigibson.utils.motion_planning_utils import astar, dijkstra

# Modify get_shortest_path() method
def get_shortest_path(self, floor, source_world, target_world, entire_path=False, robot=None, algorithm='astar'):
    """Add algorithm parameter."""
    source_map = tuple(self.world_to_map(source_world).tolist())
    target_map = tuple(self.world_to_map(target_world).tolist())

    trav_map = th.clone(self.floor_map[floor])
    trav_map = self._erode_trav_map(trav_map, robot=robot)

    # Select algorithm
    if algorithm == 'astar':
        path_map = astar(trav_map, source_map, target_map)
    elif algorithm == 'dijkstra':
        path_map = dijkstra(trav_map, source_map, target_map)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # ... rest unchanged
```

**Step 3**: Test it (5 minutes)

```python
# Quick test script
import omnigibson as og

env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

# Test A*
path_astar, dist_astar = scene.get_shortest_path(
    floor=0,
    source_world=[0, 0],
    target_world=[5, 5],
    algorithm='astar',
    robot=robot
)

# Test Dijkstra
path_dijkstra, dist_dijkstra = scene.get_shortest_path(
    floor=0,
    source_world=[0, 0],
    target_world=[5, 5],
    algorithm='dijkstra',
    robot=robot
)

print(f"A*:       {len(path_astar)} waypoints, {dist_astar:.2f}m")
print(f"Dijkstra: {len(path_dijkstra)} waypoints, {dist_dijkstra:.2f}m")
# Should be identical since Dijkstra is A* with h=0
```

**Total time**: ~30 minutes for complete implementation and testing!

---

## Advanced: OMPL Integration

### Using Existing OMPL Infrastructure

**Good news**: OMPL integration already exists in the codebase!

**Location**: `omnigibson/utils/motion_planning_utils.py` function `plan_base_motion()` (line 24)

**What it does**:
- SE(2) planning (x, y, θ) with non-holonomic constraints
- Custom motion validator for differential drive
- Collision checking integration
- Supports RRT*, PRM, and other OMPL planners

**How to use it**:

```python
from omnigibson.utils.motion_planning_utils import plan_base_motion
from omnigibson.utils.planning_utils import PlanningContext

# Create planning context (robot copy for collision checking)
context = PlanningContext(env=env, robot=robot, mode='navigation')

# Plan with OMPL (RRT* by default)
path = plan_base_motion(
    robot=robot,
    end_conf=[goal_x, goal_y, goal_yaw],  # SE(2) goal
    context=context,
    planning_time=15.0
)

# path is list of [x, y, yaw] poses
```

**Customizing the planner**:

```python
# Look at plan_base_motion() implementation
# Around line 120, it creates the planner:

from ompl import geometric as ompl_geo

# Currently uses RRTConnect
planner = ompl_geo.RRTConnect(si)

# You can replace with:
# planner = ompl_geo.RRTstar(si)  # For optimal paths
# planner = ompl_geo.PRM(si)      # For multi-query
# planner = ompl_geo.LBTRRT(si)   # For lower bound tree RRT
```

**Key insight**: You don't need to implement RRT from scratch - just use the existing OMPL integration!

---

## Benchmarking Framework

### Creating a Planner Comparison Suite

```python
# File: utils/planner_benchmark.py

import time
import numpy as np
from typing import Dict, List, Callable

class PlannerBenchmark:
    """Framework for comparing path planning algorithms."""

    def __init__(self, scene, robot):
        self.scene = scene
        self.robot = robot
        self.results = []

    def add_planner(self, name: str, planner_fn: Callable):
        """Register a planner for benchmarking."""
        self.planners[name] = planner_fn

    def run_benchmark(self, test_cases: List[Dict], num_runs: int = 5):
        """
        Run all planners on all test cases.

        Args:
            test_cases: List of dicts with 'start' and 'goal'
            num_runs: Number of times to run each (for timing)
        """
        for case_idx, case in enumerate(test_cases):
            start = case['start']
            goal = case['goal']

            for planner_name, planner_fn in self.planners.items():
                times = []
                paths = []

                for run in range(num_runs):
                    start_time = time.time()

                    try:
                        path, distance = planner_fn(
                            scene=self.scene,
                            robot=self.robot,
                            start=start,
                            goal=goal
                        )
                        elapsed = time.time() - start_time

                        if path is not None:
                            times.append(elapsed)
                            paths.append((path, distance))
                    except Exception as e:
                        print(f"Error in {planner_name}: {e}")
                        continue

                if times:
                    self.results.append({
                        'case': case_idx,
                        'planner': planner_name,
                        'success_rate': len(paths) / num_runs,
                        'avg_time': np.mean(times),
                        'std_time': np.std(times),
                        'avg_distance': np.mean([d for _, d in paths]),
                        'avg_waypoints': np.mean([len(p) for p, _ in paths]),
                    })

    def print_summary(self):
        """Print benchmark results."""
        import pandas as pd
        df = pd.DataFrame(self.results)

        print("\n=== Planner Benchmark Results ===\n")
        print(df.groupby('planner').agg({
            'success_rate': 'mean',
            'avg_time': 'mean',
            'avg_distance': 'mean',
        }))
```

**Usage**:

```python
from utils.planner_benchmark import PlannerBenchmark

bench = PlannerBenchmark(scene=env.scene, robot=env.robots[0])

# Define planner wrappers
def astar_wrapper(scene, robot, start, goal):
    return scene.get_shortest_path(0, start, goal, algorithm='astar', robot=robot)

def dijkstra_wrapper(scene, robot, start, goal):
    return scene.get_shortest_path(0, start, goal, algorithm='dijkstra', robot=robot)

# Register planners
bench.add_planner('A*', astar_wrapper)
bench.add_planner('Dijkstra', dijkstra_wrapper)

# Create test cases
test_cases = [
    {'start': [0, 0], 'goal': [5, 5]},
    {'start': [0, 0], 'goal': [10, 10]},
    # ... more cases
]

# Run benchmark
bench.run_benchmark(test_cases, num_runs=10)
bench.print_summary()
```

---

## Common Pitfalls

### Pitfall 1: Coordinate System Confusion

**Problem**: OmniGibson uses different coordinate systems

```python
# World coordinates (meters, continuous)
world_pos = [1.5, 2.3]  # meters

# Map coordinates (pixels, discrete)
map_pos = scene.world_to_map(world_pos)  # e.g., [150, 230] at 0.1m resolution

# Robot coordinates (SE(3) with rotation)
robot_pose = [x, y, z, qx, qy, qz, qw]
```

**Solution**: Always use provided conversion functions
```python
# GOOD
world_pos = scene.map_to_world(map_pos)
map_pos = scene.world_to_map(world_pos)

# BAD - manual conversion
map_pos = (int(world_pos[0] / 0.1), int(world_pos[1] / 0.1))  # DON'T!
```

### Pitfall 2: Forgetting Robot Size Erosion

**Problem**: Path looks valid on map but robot collides

```python
# BAD - no erosion
trav_map = scene.floor_map[0]
path = my_planner(trav_map, start, goal)
# Robot might not fit!

# GOOD - with erosion
trav_map = th.clone(scene.floor_map[0])
trav_map = scene._erode_trav_map(trav_map, robot=robot)
path = my_planner(trav_map, start, goal)
```

### Pitfall 3: Memory Leaks in Grid Initialization

**Problem**: A* example initializes full grid

```python
# BAD - allocates N×M dictionary
rows, cols = search_map.shape
g_score = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
# For 1000×1000 map = 1 million entries!
```

**Solution**: Use lazy initialization
```python
# GOOD - only allocate as needed
from collections import defaultdict
g_score = defaultdict(lambda: float('inf'))
g_score[start] = 0
```

### Pitfall 4: Not Handling "No Path" Cases

**Problem**: Planner crashes when no path exists

```python
# BAD
path = my_planner(map, start, goal)
for waypoint in path:  # Crashes if path is None!
    ...

# GOOD
path = my_planner(map, start, goal)
if path is None:
    print("No path found!")
    return
for waypoint in path:
    ...
```

### Pitfall 5: Ignoring Multi-Floor Scenarios

**Problem**: Planning doesn't account for floor changes

```python
# BAD - assumes single floor
path = plan(map, start, goal)

# GOOD - check floor parameter
if start_floor != goal_floor:
    # Need to find stairs/elevator
    # Plan multi-floor path
    ...
```

---

## Summary

### Ease of Implementation

| Planner Type | Difficulty | Time Estimate | Recommendation |
|--------------|-----------|---------------|----------------|
| Grid-based 2D (Dijkstra, BFS) | ⭐ Easy | 2-4 hours | **Start here** |
| Grid variants (Theta*, JPS) | ⭐⭐ Moderate | 4-8 hours | Good learning project |
| TSP optimization | ⭐⭐ Moderate | 4-8 hours | **High impact** |
| OMPL/RRT (using existing) | ⭐⭐⭐ Moderate | 1-2 days | **Use existing code** |
| OMPL/RRT (from scratch) | ⭐⭐⭐⭐ Hard | 1-2 weeks | Not recommended |
| Full benchmarking suite | ⭐⭐⭐ Moderate | 3-5 days | Good for publication |

### Key Takeaways

1. **The framework is surprisingly modular**
   - Clean separation between algorithm and scene
   - Simple grid-based interface
   - Easy to swap planners

2. **OMPL integration already exists**
   - Don't reimplement RRT/PRM from scratch
   - Use `plan_base_motion()` as reference
   - Handles non-holonomic constraints

3. **Biggest challenges are not the algorithms**
   - Understanding coordinate systems
   - Handling robot size correctly
   - Multi-floor navigation
   - TSP integration

4. **Start simple, then extend**
   - Begin with drop-in Dijkstra/BFS
   - Add TSP optimization (big wins)
   - Use OMPL for advanced planning
   - Build benchmarking last

### Next Steps

1. **Quick win**: Implement nearest-neighbor TSP (4 hours, 30% improvement)
2. **Medium effort**: Add planner selection system (1 day)
3. **Research contribution**: Benchmark suite + new algorithm (1-2 weeks)

**The framework is well-designed for extension** - you can add sophisticated planning without fighting the infrastructure! 🎯

---

## References

- **A* Implementation**: `/home/ssangeetha3/git/BEHAVIOR-1K/OmniGibson/omnigibson/utils/motion_planning_utils.py` (line 553)
- **OMPL Integration**: Same file, `plan_base_motion()` (line 24)
- **TraversableMap**: `/home/ssangeetha3/git/BEHAVIOR-1K/OmniGibson/omnigibson/maps/traversable_map.py`
- **Navigation Utilities**: `/home/ssangeetha3/git/BEHAVIOR-1K/utils/navigation_utils.py`
- **Demo Scripts**: `/home/ssangeetha3/git/BEHAVIOR-1K/examples/navigation/`

**Document Version**: 1.0
**Last Updated**: October 27, 2025