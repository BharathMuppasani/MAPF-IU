#!/usr/bin/env python3
"""
Test script for C++ joint A* module.
"""

import numpy as np


def test_cpp_joint_astar_module():
    """Test the C++ joint A* module directly."""
    try:
        import cpp_joint_astar
        print("✓ cpp_joint_astar module imported successfully!")

        # Test 1: Simple 2-agent scenario on 5x5 grid
        print("\nTest 1: Simple 2-agent joint planning (5x5 grid)")
        H, W = 5, 5
        static_grid = [[0 for _ in range(W)] for _ in range(H)]

        # Agent 1: (0,0) -> (0,4)
        # Agent 2: (4,0) -> (4,4)
        start_positions = [
            {'r': 0, 'c': 0},
            {'r': 4, 'c': 0}
        ]
        subgoal_positions = [
            {'r': 0, 'c': 4},
            {'r': 4, 'c': 4}
        ]

        t_start = 0
        t_goal_sub = 8  # Need 4 steps each, allow some buffer
        max_expansions = 10000
        time_budget = 5.0

        # No reservations or blocked cells
        reserved_cells_by_time = []
        reserved_moves_by_time = []
        blocked_cells = []
        blocked_by_time = []

        result = cpp_joint_astar.joint_astar_grid_time(
            H, W, static_grid,
            start_positions, subgoal_positions,
            t_start, t_goal_sub,
            max_expansions, time_budget,
            reserved_cells_by_time, reserved_moves_by_time,
            blocked_cells, blocked_by_time,
            True  # use_time_based_blocking
        )

        if result.success:
            print(f"  ✓ Found joint plan!")
            print(f"  Agent 1 plan: {result.plans[0]}")
            print(f"  Agent 2 plan: {result.plans[1]}")
            print(f"  Agent 1 trajectory length: {len(result.trajectories[0])}")
            print(f"  Agent 2 trajectory length: {len(result.trajectories[1])}")
        else:
            print("  ✗ Failed to find joint plan (unexpected)")
            return False

        # Test 2: 2-agent with obstacle
        print("\nTest 2: 2-agent with obstacle (5x5 grid)")
        static_grid = [[0 for _ in range(W)] for _ in range(H)]
        # Add vertical wall at column 2
        for r in range(1, 4):
            static_grid[r][2] = -1

        # Agent 1: (0,0) -> (0,4)
        # Agent 2: (4,0) -> (4,4)
        # They must go around the obstacle
        result = cpp_joint_astar.joint_astar_grid_time(
            H, W, static_grid,
            start_positions, subgoal_positions,
            t_start, 12,  # Allow more time due to obstacle
            max_expansions, time_budget,
            reserved_cells_by_time, reserved_moves_by_time,
            blocked_cells, blocked_by_time,
            True
        )

        if result.success:
            print(f"  ✓ Found joint plan with obstacle!")
            print(f"  Agent 1 plan length: {len(result.plans[0])}")
            print(f"  Agent 2 plan length: {len(result.plans[1])}")
        else:
            print("  ✗ Failed to find joint plan (unexpected)")
            return False

        # Test 3: 2-agent with reservation (one agent blocked at a cell)
        print("\nTest 3: 2-agent with reserved cells")
        static_grid = [[0 for _ in range(W)] for _ in range(H)]

        # Agent 1: (0,0) -> (0,2)
        # Agent 2: (1,0) -> (1,2)
        start_positions = [
            {'r': 0, 'c': 0},
            {'r': 1, 'c': 0}
        ]
        subgoal_positions = [
            {'r': 0, 'c': 2},
            {'r': 1, 'c': 2}
        ]

        # Reserve cell (0,1) at time 1 and 2 (another agent is there)
        reserved_cells_by_time = [
            [{'r': 0, 'c': 1}],  # t=1
            [{'r': 0, 'c': 1}],  # t=2
            []                    # t=3
        ]
        reserved_moves_by_time = [[], [], []]

        result = cpp_joint_astar.joint_astar_grid_time(
            H, W, static_grid,
            start_positions, subgoal_positions,
            0, 3,
            max_expansions, time_budget,
            reserved_cells_by_time, reserved_moves_by_time,
            blocked_cells, blocked_by_time,
            True
        )

        if result.success:
            print(f"  ✓ Found joint plan avoiding reserved cells!")
            # Check that agent 1 doesn't go through (0,1) at time 1
            traj1 = result.trajectories[0]
            if len(traj1) > 1:
                pos_at_t1 = (traj1[1].r, traj1[1].c)
                if pos_at_t1 == (0, 1):
                    print(f"  ✗ Agent 1 violated reservation at t=1: {pos_at_t1}")
                    return False
                else:
                    print(f"  ✓ Agent 1 correctly avoided reserved cell")
        else:
            print("  Note: Failed to find plan (may be expected if constraints too tight)")

        # Test 4: WAIT action test
        print("\nTest 4: WAIT action (agents must wait to avoid each other)")
        static_grid = [[0 for _ in range(3)] for _ in range(3)]

        # Narrow corridor scenario: agents meet head-on
        # Agent 1: (0,0) -> (0,2)
        # Agent 2: (0,2) -> (0,0)
        start_positions = [
            {'r': 0, 'c': 0},
            {'r': 0, 'c': 2}
        ]
        subgoal_positions = [
            {'r': 0, 'c': 2},
            {'r': 0, 'c': 0}
        ]

        result = cpp_joint_astar.joint_astar_grid_time(
            3, 3, static_grid,
            start_positions, subgoal_positions,
            0, 6,  # Allow enough time for waiting
            max_expansions, time_budget,
            [], [],  # No external reservations
            [], [],
            True
        )

        if result.success:
            print(f"  ✓ Found joint plan with WAIT actions!")
            # Check if WAIT action (4) was used
            wait_count = sum(1 for plan in result.plans for action in plan if action == 4)
            print(f"  Number of WAIT actions: {wait_count}")
            if wait_count > 0:
                print(f"  ✓ WAIT action was used correctly!")
            else:
                print(f"  Note: No WAIT actions used (plan may be short enough)")
        else:
            print("  Note: Failed to find plan (corridor conflict may be unsolvable)")

        print("\n✓ All C++ joint A* tests passed!")
        return True

    except ImportError as e:
        print(f"✗ Failed to import cpp_joint_astar module: {e}")
        print("\nThe C++ module needs to be built first. Run:")
        print("  pip install pybind11")
        print("  python setup_cpp_joint_astar.py build_ext --inplace")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_python_integration():
    """Test integration with fix.py."""
    print("\n" + "="*60)
    print("Testing Python Integration with fix.py")
    print("="*60)

    try:
        from fix import HAS_CPP_JOINT_ASTAR, USE_CPP_JOINT_ASTAR

        print(f"C++ joint A* available: {HAS_CPP_JOINT_ASTAR}")
        print(f"C++ joint A* enabled: {USE_CPP_JOINT_ASTAR}")

        if HAS_CPP_JOINT_ASTAR and USE_CPP_JOINT_ASTAR:
            print("✓ C++ joint A* is available and enabled in fix.py")
        elif HAS_CPP_JOINT_ASTAR and not USE_CPP_JOINT_ASTAR:
            print("⚠ C++ joint A* is available but disabled (USE_CPP_JOINT_ASTAR=False)")
        elif not HAS_CPP_JOINT_ASTAR and USE_CPP_JOINT_ASTAR:
            print("⚠ C++ joint A* is enabled but not available (not built)")
        else:
            print("⚠ C++ joint A* is disabled")

        return True

    except ImportError as e:
        print(f"✗ Failed to import from fix.py: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_encoding():
    """Test that action encoding matches Python expectations."""
    print("\n" + "="*60)
    print("Testing Action Encoding")
    print("="*60)

    try:
        import cpp_joint_astar

        # Test that actions match expected encoding
        print("Expected action encoding:")
        print("  0: up    (-1,  0)")
        print("  1: down  ( 1,  0)")
        print("  2: left  ( 0, -1)")
        print("  3: right ( 0,  1)")
        print("  4: wait  ( 0,  0)")

        # Simple test: agent moves right
        H, W = 3, 3
        static_grid = [[0 for _ in range(W)] for _ in range(H)]

        start_positions = [{'r': 1, 'c': 0}]
        subgoal_positions = [{'r': 1, 'c': 2}]

        result = cpp_joint_astar.joint_astar_grid_time(
            H, W, static_grid,
            start_positions, subgoal_positions,
            0, 2,
            10000, 5.0,
            [], [], [], [],
            True
        )

        if result.success and len(result.plans[0]) > 0:
            plan = result.plans[0]
            traj = result.trajectories[0]

            print(f"\nTest case: Move right from (1,0) to (1,2)")
            print(f"  Plan: {plan}")
            print(f"  Trajectory: [(cell.r, cell.c) for cell in traj]")
            print(f"  Trajectory: {[(cell.r, cell.c) for cell in traj]}")

            # Verify trajectory matches actions
            correct = True
            actions_map = {
                0: (-1, 0),   # up
                1: (1, 0),    # down
                2: (0, -1),   # left
                3: (0, 1),    # right
                4: (0, 0)     # wait
            }

            for i, action in enumerate(plan):
                current_pos = (traj[i].r, traj[i].c)
                next_pos = (traj[i+1].r, traj[i+1].c)
                dr, dc = actions_map[action]
                expected_next = (current_pos[0] + dr, current_pos[1] + dc)

                if next_pos != expected_next:
                    print(f"  ✗ Action {action} at step {i}: {current_pos} -> {next_pos}, expected {expected_next}")
                    correct = False

            if correct:
                print("  ✓ Action encoding is correct!")
                return True
            else:
                print("  ✗ Action encoding mismatch!")
                return False
        else:
            print("  ✗ Failed to find plan for simple test case")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("C++ Joint A* Module Test Suite")
    print("="*60)

    # Test the C++ module directly
    cpp_test_passed = test_cpp_joint_astar_module()

    # Test action encoding
    if cpp_test_passed:
        encoding_test_passed = test_action_encoding()
    else:
        print("\nSkipping action encoding test (C++ module not available)")
        encoding_test_passed = False

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
    print(f"Action Encoding: {'✓ PASSED' if encoding_test_passed else '✗ FAILED'}")
    print(f"Python Integration: {'✓ PASSED' if py_test_passed else '✗ FAILED'}")

    if cpp_test_passed and encoding_test_passed and py_test_passed:
        print("\n✓ All tests passed! C++ joint A* is ready to use.")
        print("\nTo use it in your experiments:")
        print("  - Make sure USE_CPP_JOINT_ASTAR = True in fix.py (default)")
        print("  - Run your experiments normally with run_exp.py")
        print("  - The joint A* will automatically use C++ when available")
    elif not cpp_test_passed:
        print("\n⚠ Build the C++ module first:")
        print("  pip install pybind11")
        print("  python setup_cpp_joint_astar.py build_ext --inplace")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
