# C++ A* Module Integration

This guide explains how to build and use the C++ A* module for faster pathfinding.

## Overview

The C++ A* module provides a significant performance boost over the pure Python implementation by implementing the A* algorithm in C++ and exposing it to Python via pybind11.

## Building the C++ Module

### Prerequisites

You need to have `pybind11` installed:

```bash
pip install pybind11
```

### Build Instructions

From the repository root, run:

```bash
python setup_cpp_astar.py build_ext --inplace
```

This will compile the C++ code and create a `cpp_astar.*.so` file (the exact filename depends on your Python version and platform) in the repository root.

### Troubleshooting Build Issues

**macOS**: If you encounter compiler errors, make sure you have Xcode Command Line Tools installed:
```bash
xcode-select --install
```

**Linux**: Ensure you have g++ installed:
```bash
sudo apt-get install build-essential  # Ubuntu/Debian
sudo yum install gcc-c++              # CentOS/RHEL
```

**Windows**: You'll need Visual Studio with C++ support installed.

## Usage

### Using C++ A* in Your Code

Once built, you can use the C++ A* in your experiments by specifying `--search_type astar-cpp`:

```bash
# Run with C++ A* on a single instance
python run_exp.py --map_file path/to/instance.txt --search_type astar-cpp

# Run with C++ A* and custom parameters
python run_exp.py \
    --map_file path/to/instance.txt \
    --search_type astar-cpp \
    --heuristic_weight 1.5 \
    --max_expansions 500000 \
    --timeout 60.0
```

### Fallback Behavior

If the C++ module is not available (not built or build failed), the code will automatically fall back to the pure Python A* implementation with a warning message.

### Comparison with RL-Guided A*

- **`--search_type astar`**: Uses RL-guided A* (PPO or DQN) that uses the learned policy to guide search
- **`--search_type astar-cpp`**: Uses pure geometric A* (faster, no model needed)
- **`--search_type greedy-bfs`**: Uses greedy best-first search with RL guidance
- **`--search_type bfs`**: Alias for greedy-bfs

## Performance Expectations

The C++ A* module should provide:
- **10-100x faster** pathfinding compared to pure Python A*
- Lower memory usage
- Better scalability for large maps

## Technical Details

### Action Encoding

Both Python and C++ A* use the same action encoding:
- `0`: up (-1, 0)
- `1`: down (1, 0)
- `2`: left (0, -1)
- `3`: right (0, 1)

### Grid Convention

- `0` or positive values: traversable cells
- `-1`: obstacles
- The grid is converted to int32 before passing to C++

### Parameters

- **`heuristic_weight`**: Weight applied to the Manhattan distance heuristic (default: 1.0 for optimal paths, >1.0 for faster but potentially suboptimal paths)
- **`max_expansions`**: Maximum number of nodes to expand before giving up (default: 500000)
- **`timeout`**: Maximum time in seconds (note: C++ version uses `max_expansions` as the primary limit)

## Code Structure

- **`cpp_astar_module.cpp`**: C++ implementation of A* algorithm
- **`setup_cpp_astar.py`**: Build script using pybind11
- **`utils/search_utils.py`**: Python wrapper and integration
  - `astar_cpp()`: Python wrapper function
  - `plan_with_search()`: Dispatcher that routes to appropriate planner
- **`run_exp.py`**: Command-line interface with `--search_type astar-cpp` option

## Verifying the Installation

To verify the C++ module is working:

```python
import cpp_astar

# Test on a simple 5x5 grid with no obstacles
grid = [[0 for _ in range(5)] for _ in range(5)]
actions = cpp_astar.astar_grid(grid, 0, 0, 4, 4)
print(f"Path length: {len(actions)}")  # Should be 8 (4 down + 4 right)
print(f"Actions: {actions}")
```

If this runs without errors, the module is correctly installed!
