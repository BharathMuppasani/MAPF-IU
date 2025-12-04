# Quick Start: C++ A* Installation

This is a quick installation guide for the C++ A* module. For detailed documentation, see [CPP_ASTAR_README.md](CPP_ASTAR_README.md).

## Step-by-Step Installation

### 1. Install pybind11

```bash
pip install pybind11
```

### 2. Build the C++ module

```bash
python setup_cpp_astar.py build_ext --inplace
```

You should see output indicating the build succeeded. A file named `cpp_astar.*.so` (or `cpp_astar.*.pyd` on Windows) will be created in the repository root.

### 3. Test the installation

```bash
python test_cpp_astar.py
```

If all tests pass, you're ready to go!

### 4. Use in experiments

```bash
# Example: Run with C++ A* on a single instance
python run_exp.py --map_file instances/scene1_even_100agents.txt --search_type astar-cpp

# Example: With custom parameters
python run_exp.py \
    --map_file instances/scene1_even_100agents.txt \
    --search_type astar-cpp \
    --heuristic_weight 1.5 \
    --max_expansions 1000000
```

## Troubleshooting

### Build fails on macOS

Install Xcode Command Line Tools:
```bash
xcode-select --install
```

### Build fails on Linux

Install build tools:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum install gcc-c++ python3-devel
```

### Module not found at runtime

Make sure you're running Python from the repository root directory where the `.so` file was created.

### Fallback to Python A*

If you see "Warning: cpp_astar module not available. Falling back to Python A*", it means the C++ module wasn't found. Check:
1. The `.so` file exists in the repository root
2. You're running from the correct directory
3. The build completed successfully

## Performance Comparison

To compare performance between Python A* and C++ A*:

```bash
# Run with Python A* (but no RL guidance, for fair comparison)
# Note: The existing "astar" uses RL guidance, so we need to benchmark separately

# Run with C++ A*
python run_exp.py --map_file instances/test.txt --search_type astar-cpp
```

Expected speedup: **10-100x faster** depending on map size and complexity.

## Files Created

- `cpp_astar_module.cpp` - C++ implementation
- `setup_cpp_astar.py` - Build configuration
- `test_cpp_astar.py` - Test suite
- `CPP_ASTAR_README.md` - Detailed documentation
- `INSTALL_CPP_ASTAR.md` - This file

## Integration Points

The C++ A* is integrated into your codebase at:
- [utils/search_utils.py](utils/search_utils.py) - `astar_cpp()` function and `plan_with_search()` dispatcher
- [run_exp.py](run_exp.py) - Command-line argument `--search_type astar-cpp`

## Next Steps

1. Build and test the module
2. Run benchmarks comparing Python vs C++ A*
3. Use `astar-cpp` in your production runs for faster pathfinding
4. Optionally: Replace direct `astar()` calls in the codebase with `astar_cpp()` for consistent speedup
