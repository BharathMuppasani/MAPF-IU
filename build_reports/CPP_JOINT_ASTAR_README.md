# C++ Joint A* Module

This document describes the C++ implementation of joint A* for multi-agent pathfinding with time-based constraints, integrated into `fix.py`.

## Overview

The C++ joint A* module provides a high-performance implementation of the joint planning algorithm used in `try_joint_astar_planning()`. It handles:

- **Multi-agent planning**: Searches in joint configuration space for multiple agents simultaneously
- **Time-based constraints**: Respects reservations and blocked cells at specific timesteps
- **WAIT action**: Includes a 5th action (wait/stay) in addition to the 4 directional moves
- **Collision avoidance**: Prevents vertex and edge conflicts both within the joint group and with external agents

## Action Encoding

**CRITICAL**: Both Python and C++ use the same action encoding:

```
0: up    (-1,  0)
1: down  ( 1,  0)
2: left  ( 0, -1)
3: right ( 0,  1)
4: wait  ( 0,  0)
```

## Building the Module

### Prerequisites

```bash
pip install pybind11
```

### Build Instructions

From the repository root:

```bash
python setup_cpp_joint_astar.py build_ext --inplace
```

This creates a `cpp_joint_astar.*.so` file (platform-dependent extension).

### Verify Installation

```bash
python test_cpp_joint_astar.py
```

## Usage in fix.py

The C++ joint A* is automatically used when available. The integration is controlled by two flags in `fix.py`:

```python
HAS_CPP_JOINT_ASTAR  # True if module is built and importable
USE_CPP_JOINT_ASTAR  # True to enable C++ (default), False to use Python
```

### Automatic Fallback

If the C++ module is not available, the code automatically falls back to the Python implementation without errors.

### Disabling C++ Joint A*

To use the Python version even when C++ is available:

1. Edit `fix.py`
2. Change `USE_CPP_JOINT_ASTAR = True` to `USE_CPP_JOINT_ASTAR = False`
3. Re-run your experiments

## API Reference

### C++ Function Signature

```cpp
JointAStarResult joint_astar_grid_time(
    int H, int W,
    const std::vector<std::vector<int>>& static_grid,
    const std::vector<Cell>& start_positions,
    const std::vector<Cell>& subgoal_positions,
    int t_start,
    int t_goal_sub,
    int max_expansions,
    double time_budget_seconds,
    const std::vector<std::unordered_set<Cell>>& reserved_cells_by_time,
    const std::vector<std::vector<std::pair<Cell,Cell>>>& reserved_moves_by_time,
    const std::unordered_set<Cell>& blocked_cells,
    const std::vector<std::unordered_set<Cell>>& blocked_by_time,
    bool use_time_based_blocking
);
```

### Python Wrapper (used in fix.py)

```python
result = cpp_joint_astar.joint_astar_grid_time(
    H,                          # Grid height
    W,                          # Grid width
    static_grid,                # 2D list: 0=free, -1=obstacle
    start_positions,            # List of {'r': row, 'c': col} dicts
    subgoal_positions,          # List of {'r': row, 'c': col} dicts
    t_start,                    # Start time of planning window
    t_goal_sub,                 # End time of planning window
    max_expansions,             # Max nodes to expand
    time_budget_seconds,        # Time limit in seconds
    reserved_cells_by_time,     # Reserved positions per timestep
    reserved_moves_by_time,     # Reserved moves per timestep
    blocked_cells,              # Permanently blocked cells
    blocked_by_time,            # Time-dependent blocked cells
    use_time_based_blocking     # Enable time-based blocking
)
```

### Return Value

```python
class JointAStarResult:
    success: bool                          # True if solution found
    plans: List[List[int]]                 # Per-agent action sequences
    trajectories: List[List[Cell]]         # Per-agent position sequences
```

## Implementation Details

### Joint State Representation

A joint state contains:
- **positions**: List of (row, col) for each agent
- **g**: Cost so far (number of timesteps)
- **f**: g + heuristic (sum of Manhattan distances to subgoals)

### Search Algorithm

1. **Initialization**: Start with all agents at their start positions at `t_start`
2. **Expansion**: For each state, enumerate all valid joint actions (action combinations)
3. **Validation**: Check for:
   - Static obstacles
   - Blocked cells (spatial or time-based)
   - Internal vertex conflicts (agents collide)
   - Internal edge conflicts (agents swap)
   - External vertex conflicts (reserved positions)
   - External edge conflicts (reserved moves)
4. **Goal Check**: All agents reach their subgoals within the time window
5. **Padding**: If goal reached early, pad with WAIT actions to match window length

### Collision Checking

**Vertex Conflicts:**
- Multiple agents occupy the same cell at the same time
- Agent occupies a reserved cell

**Edge Conflicts (Swaps):**
- Two agents swap positions between consecutive timesteps
- Agent moves conflicts with a reserved agent's move

### Time Window

The search operates within a local time window `[t_start, t_goal_sub]`:
- Starts planning from `t_start` (typically before collision)
- Must reach subgoals by `t_goal_sub` (typically after collision)
- Window expands iteratively if search fails

### Heuristic

Manhattan distance sum:
```
h = Σ |agent_pos[i] - subgoal_pos[i]|
```

This is admissible and consistent for grid pathfinding.

## Performance Characteristics

### Expected Speedup

- **10-100x faster** than Python implementation
- Speedup increases with:
  - Larger joint groups (more agents)
  - Tighter constraints (more reservations)
  - Longer planning windows

### Memory Usage

- Lower than Python due to efficient C++ data structures
- Scales with: `O(states_visited × num_agents × 2)` for visited set

### Bottlenecks

- **Successor generation**: Enumerating all action combinations is exponential in number of agents
- **Visited set**: Hash table lookups for state checking
- **Validation**: Checking conflicts with reserved agents

## Troubleshooting

### Build Errors

**macOS**: Install Xcode Command Line Tools
```bash
xcode-select --install
```

**Linux**: Install build tools
```bash
sudo apt-get install build-essential python3-dev  # Ubuntu/Debian
sudo yum install gcc-c++ python3-devel           # CentOS/RHEL
```

**Windows**: Install Visual Studio with C++ support

### Module Not Found at Runtime

1. Check that `.so` file exists in repository root
2. Run from repository root directory
3. Verify build completed successfully

### Validation Errors

If C++ solutions are rejected by `check_joint_segment_conflicts`:
- Check action encoding matches (0-4)
- Verify trajectory lengths match window size
- Ensure coordinate systems align (Python vs C++)

### Performance Issues

If C++ version is slower than expected:
- Check `max_expansions` setting (may be hitting limit)
- Verify compilation used `-O3` optimization
- Profile to identify bottlenecks

## Testing

### Run All Tests

```bash
python test_cpp_joint_astar.py
```

### Test Categories

1. **Module Tests**: Basic functionality, obstacles, reservations, WAIT actions
2. **Action Encoding**: Verify actions match Python expectations
3. **Integration Tests**: Check fix.py integration

### Manual Testing

To test in isolation:

```python
import cpp_joint_astar
import numpy as np

# Simple 2-agent test
H, W = 5, 5
grid = [[0]*W for _ in range(H)]
starts = [{'r': 0, 'c': 0}, {'r': 4, 'c': 0}]
goals = [{'r': 0, 'c': 4}, {'r': 4, 'c': 4}]

result = cpp_joint_astar.joint_astar_grid_time(
    H, W, grid, starts, goals,
    0, 8, 10000, 5.0,
    [], [], [], [], True
)

if result.success:
    print(f"Plans: {result.plans}")
```

## Comparison with Python Implementation

| Feature | Python | C++ |
|---------|--------|-----|
| Language | Pure Python | C++ with pybind11 |
| Speed | Baseline | 10-100x faster |
| Memory | Higher | Lower |
| Debugging | Easier | Harder |
| Action Encoding | 0-4 | 0-4 (same) |
| Constraints | Full support | Full support |
| WAIT Action | ✓ | ✓ |
| Fallback | N/A | Automatic |

## Integration Points

### fix.py

- **Import**: Lines 11-19 (module import and flags)
- **Function**: Lines 1914-2027 (`search_window_cpp`)
- **Dispatch**: Lines 2254-2260 (choose Python vs C++)

### Data Conversion

The Python wrapper (`search_window_cpp`) converts between Python and C++ formats:

**Python → C++:**
- Tuples → `{'r': int, 'c': int}` dicts
- NumPy arrays → nested lists
- Sets → lists (converted to unordered_set in C++)

**C++ → Python:**
- `Cell` objects → `(r, c)` tuples
- Action vectors → Python lists

## Future Enhancements

Potential improvements:

1. **Parallel expansion**: Multi-threaded successor generation
2. **Better heuristics**: Conflict-based heuristics, true distance tables
3. **Incremental search**: Reuse previous search results
4. **Memory optimization**: More compact state representation
5. **Adaptive expansion**: Dynamic max_expansions based on problem size

## References

- **Python implementation**: `fix.py` lines 2029-2214 (`search_window`)
- **Action encoding**: Matches `ACTIONS` in `try_joint_astar_planning` (line 1792)
- **Constraint checking**: `check_joint_segment_conflicts` (lines 1825-1902)

## License

Same as the parent repository.
