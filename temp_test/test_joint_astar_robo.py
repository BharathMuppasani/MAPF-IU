"""
Test joint A* planner directly on robo_test.txt map.
Tests if the planner can find valid solutions using actions 0-4.
"""

import numpy as np
from fix import try_joint_astar_planning, compute_heuristic_distances
from utils.env_utils import analyze_collisions


def load_map(filename):
    """Load a map file in the format used by the MAPF benchmarks"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    parts = lines[0].strip().split()
    rows, cols = int(parts[0]), int(parts[1])

    # Parse grid
    grid = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        line = lines[r + 1].strip()
        cells = line.split()
        for c, cell in enumerate(cells):
            if cell == '@':
                grid[r, c] = -1  # Obstacle
            else:
                grid[r, c] = 0   # Free

    # Parse agents
    num_agents = int(lines[rows + 1].strip())
    agents = []
    for i in range(num_agents):
        parts = lines[rows + 2 + i].strip().split()
        agents.append({
            'start': (int(parts[0]), int(parts[1])),
            'goal': (int(parts[2]), int(parts[3]))
        })

    return grid, agents


def test_joint_astar_on_robo():
    """Test joint A* planner on robo_test.txt"""

    print("=" * 70)
    print("Testing Joint A* Planner on robo_test.txt")
    print("=" * 70)
    print()

    # Load map
    grid, agents = load_map('test_data/maps/robo_test.txt')

    print(f"Grid size: {grid.shape}")
    print(f"Number of agents: {len(agents)}")
    print()

    # Display grid
    print("Grid (@ = obstacle, . = free):")
    for r in range(grid.shape[0]):
        row_str = ""
        for c in range(grid.shape[1]):
            if grid[r, c] == -1:
                row_str += "@ "
            else:
                row_str += ". "
        print(row_str)
    print()

    # Parse agent info
    agent_starts = [agent['start'] for agent in agents]
    agent_goals = [agent['goal'] for agent in agents]

    print("Agent Configuration:")
    for i, agent in enumerate(agents):
        print(f"  Agent {i+1}: Start {agent['start']}, Goal {agent['goal']}")
    print()

    # Create initial trajectories (straight line - will collide)
    print("Creating initial trajectories (agents moving in straight lines)...")
    current_trajectories = []

    for i, agent in enumerate(agents):
        start = agent['start']
        goal = agent['goal']

        # Simple straight line trajectory from start to goal
        trajectory = [start]

        if start[0] == goal[0]:  # Same row - move horizontally
            direction = 1 if goal[1] > start[1] else -1
            for c in range(start[1] + direction, goal[1] + direction, direction):
                trajectory.append((start[0], c))
        elif start[1] == goal[1]:  # Same column - move vertically
            direction = 1 if goal[0] > start[0] else -1
            for r in range(start[0] + direction, goal[0] + direction, direction):
                trajectory.append((r, start[1]))
        else:
            # Move row first, then column
            direction_r = 1 if goal[0] > start[0] else -1
            for r in range(start[0] + direction_r, goal[0] + direction_r, direction_r):
                trajectory.append((r, start[1]))

            direction_c = 1 if goal[1] > start[1] else -1
            for c in range(start[1] + direction_c, goal[1] + direction_c, direction_c):
                trajectory.append((trajectory[-1][0], c))

        current_trajectories.append(trajectory)
        print(f"  Agent {i+1}: Start {start}, Goal {goal} => {trajectory}")

    print()

    # Detect collisions
    collisions = analyze_collisions(current_trajectories, agent_goals, grid, verbose=True)
    print(f"\nInitial collisions: {len(collisions)}")
    for coll in collisions:
        print(f"  T={coll['time']}, Type={coll['type']}, Agents={list(coll['agents'])}, Cell={coll['cell']}")
    print()

    if not collisions:
        print("No collisions detected! Nothing to resolve.")
        return

    # Pick first collision
    collision = collisions[0]
    print(f"Testing Joint A* on collision: {collision}")
    print()

    # Compute heuristics
    print("Precomputing heuristic distances...")
    heuristic_dist_map = compute_heuristic_distances(grid, agent_goals)
    print("Done.")
    print()

    # Run joint A* planner
    print("=" * 70)
    print("Running Joint A* Planner")
    print("=" * 70)
    print()

    success, joint_plans, joint_trajs, t_start, t_goal_sub = try_joint_astar_planning(
        collision,
        current_trajectories,
        agent_goals,
        agent_starts,
        grid,
        heuristic_dist_map,
        max_agents=len(agents),
        time_budget=10.0,
        max_expansions=100000,
        verbose=True
    )

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Success: {success}")

    if success:
        print(f"Planning window: t_start={t_start}, t_goal_sub={t_goal_sub} (horizon={t_goal_sub - t_start})")
        print()

        print("Joint Plans (action sequences):")
        for aid in sorted(joint_plans.keys()):
            plan = joint_plans[aid]
            action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'WAIT'}
            action_str = ' '.join([action_names.get(a, str(a)) for a in plan])
            print(f"  Agent {aid}: {action_str}")
            print(f"           ({len(plan)} actions)")

        print()
        print("Joint Trajectories (positions):")
        for aid in sorted(joint_trajs.keys()):
            traj = joint_trajs[aid]
            print(f"  Agent {aid}: {traj}")

        print()

        # Verify no collisions in solution
        test_trajs = list(current_trajectories)
        for aid, traj in joint_trajs.items():
            test_trajs[aid - 1] = traj

        new_collisions = analyze_collisions(test_trajs, agent_goals, grid, verbose=True)
        print()
        print(f"Collisions in joint solution: {len(new_collisions)}")
        if new_collisions:
            for coll in new_collisions:
                print(f"  T={coll['time']}, Type={coll['type']}, Agents={list(coll['agents'])}")
            print("  ✗ Solution has collisions!")
        else:
            print("  ✓ No collisions! Solution is valid.")
    else:
        print("✗ Joint A* failed to find a solution")
        print()
        print("This could mean:")
        print("  - The planning window was too constrained")
        print("  - The search space was exhausted before finding a solution")
        print("  - The collision is inherently hard to resolve with the given parameters")


if __name__ == "__main__":
    test_joint_astar_on_robo()
    print()
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)
