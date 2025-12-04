#!/usr/bin/env python3
"""
Simple test script to verify the C++ A* module is working correctly.
"""

def test_cpp_astar_module():
    """Test the C++ A* module directly."""
    try:
        import cpp_astar
        print("✓ cpp_astar module imported successfully!")

        # Test 1: Simple 5x5 grid
        print("\nTest 1: Simple 5x5 grid (no obstacles)")
        grid = [[0 for _ in range(5)] for _ in range(5)]
        actions = cpp_astar.astar_grid(grid, 0, 0, 4, 4)
        print(f"  Path length: {len(actions)} (expected: 8)")
        print(f"  Actions: {actions}")
        assert len(actions) == 8, "Expected path length of 8"
        print("  ✓ Test 1 passed!")

        # Test 2: Grid with obstacle
        print("\nTest 2: 5x5 grid with obstacle")
        grid = [[0 for _ in range(5)] for _ in range(5)]
        # Add vertical wall at column 2
        for r in range(1, 4):
            grid[r][2] = -1
        actions = cpp_astar.astar_grid(grid, 0, 0, 0, 4)
        print(f"  Path length: {len(actions)}")
        print(f"  Actions: {actions}")
        assert len(actions) > 0, "Expected to find a path"
        print("  ✓ Test 2 passed!")

        # Test 3: No path (completely blocked)
        print("\nTest 3: Grid with no path")
        grid = [[0 for _ in range(5)] for _ in range(5)]
        # Create complete vertical wall
        for r in range(5):
            grid[r][2] = -1
        actions = cpp_astar.astar_grid(grid, 0, 0, 0, 4)
        print(f"  Path length: {len(actions)} (expected: 0, no path)")
        assert len(actions) == 0, "Expected no path"
        print("  ✓ Test 3 passed!")

        # Test 4: Start == Goal
        print("\nTest 4: Start equals goal")
        grid = [[0 for _ in range(5)] for _ in range(5)]
        actions = cpp_astar.astar_grid(grid, 2, 2, 2, 2)
        print(f"  Path length: {len(actions)} (expected: 0)")
        assert len(actions) == 0, "Expected empty path when start == goal"
        print("  ✓ Test 4 passed!")

        print("\n✓ All tests passed! C++ A* module is working correctly.")
        return True

    except ImportError as e:
        print(f"✗ Failed to import cpp_astar module: {e}")
        print("\nThe C++ module needs to be built first. Run:")
        print("  pip install pybind11")
        print("  python setup_cpp_astar.py build_ext --inplace")
        return False
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_python_integration():
    """Test the Python wrapper integration."""
    print("\n" + "="*60)
    print("Testing Python Integration")
    print("="*60)

    try:
        from utils.search_utils import astar_cpp, HAS_CPP_ASTAR
        import numpy as np

        if not HAS_CPP_ASTAR:
            print("⚠ C++ module not available, will use Python fallback")
        else:
            print("✓ C++ module available through Python wrapper")

        # Create a mock environment for testing
        class MockEnv:
            def __init__(self, grid, agent_pos, goal_pos):
                self.grid = np.array(grid, dtype=np.int32)
                self.agent_pos = np.array(agent_pos)
                self.goal_pos = np.array(goal_pos)
                self.rows, self.cols = self.grid.shape

        class MockWrapper:
            def __init__(self, grid, agent_pos, goal_pos):
                self.env = MockEnv(grid, agent_pos, goal_pos)

        # Test with simple grid
        print("\nTest: Python wrapper with 5x5 grid")
        grid = np.zeros((5, 5), dtype=np.int32)
        wrapper = MockWrapper(grid, (0, 0), (4, 4))

        actions = astar_cpp(wrapper, timeout=10.0, heuristic_weight=1.0)
        print(f"  Path length: {len(actions) if actions else 'None'}")
        if actions:
            print(f"  Actions: {actions}")
            assert len(actions) == 8, f"Expected 8 actions, got {len(actions)}"
            print("  ✓ Python integration test passed!")
        else:
            print("  ✗ No path found (unexpected)")
            return False

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("C++ A* Module Test Suite")
    print("="*60)

    # Test the C++ module directly
    cpp_test_passed = test_cpp_astar_module()

    # Test Python integration
    if cpp_test_passed:
        py_test_passed = test_python_integration()
    else:
        print("\nSkipping Python integration test (C++ module not available)")
        py_test_passed = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"C++ Module Tests: {'✓ PASSED' if cpp_test_passed else '✗ FAILED'}")
    print(f"Python Integration: {'✓ PASSED' if py_test_passed else '✗ FAILED'}")

    if cpp_test_passed and py_test_passed:
        print("\n✓ All tests passed! You can now use --search_type astar-cpp")
    elif not cpp_test_passed:
        print("\n⚠ Build the C++ module first:")
        print("  pip install pybind11")
        print("  python setup_cpp_astar.py build_ext --inplace")
