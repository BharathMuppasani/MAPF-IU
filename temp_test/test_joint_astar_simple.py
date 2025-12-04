"""
Simple test of just the joint A* planner on robo_test.txt
Without needing the full neural network search infrastructure.
"""

import numpy as np
from fix import try_joint_astar_planning, compute_heuristic_distances
from utils.env_utils import analyze_collisions


def load_map(filename):
    """Load a map file in MAPF benchmark format"""
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
                grid[r, c] = -1
            else:
                grid[r, c] = 0

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


def main():
    print("=" * 80)
    print("Testing Joint A* Planner on robo_test.txt")
    print("=" * 80)
    print()

    # Load map
    grid, agents = load_map('test_data/maps/robo_test.txt')

    print(f"Map: robo_test.txt")
    print(f"Grid size: {grid.shape}")
    print(f"Agents: {len(agents)}")
    print()

    # Display grid
    print("Grid:")
    for r in range(grid.shape[0]):
        row_str = ""
        for c in range(grid.shape[1]):
            if grid[r, c] == -1:
                row_str += "@ "
            else:
                row_str += ". "
        print(row_str)
    print()

    # Agent config
    print("Agents:")
    for i, agent in enumerate(agents):
        print(f"  Agent {i+1}: Start {agent['start']}, Goal {agent['goal']}")
    print()

    # Create initial trajectories
    agent_starts = [agent['start'] for agent in agents]
    agent_goals = [agent['goal'] for agent in agents]

    trajectories = []
    for i, agent in enumerate(agents):
        start = agent['start']
        goal = agent['goal']

        trajectory = [start]

        if start[0] == goal[0]:  # Same row
            direction = 1 if goal[1] > start[1] else -1
            for c in range(start[1] + direction, goal[1] + direction, direction):
                trajectory.append((start[0], c))
        elif start[1] == goal[1]:  # Same column
            direction = 1 if goal[0] > start[0] else -1
            for r in range(start[0] + direction, goal[0] + direction, direction):
                trajectory.append((r, start[1]))
        else:
            direction_r = 1 if goal[0] > start[0] else -1
            for r in range(start[0] + direction_r, goal[0] + direction_r, direction_r):
                trajectory.append((r, start[1]))

            direction_c = 1 if goal[1] > start[1] else -1
            for c in range(start[1] + direction_c, goal[1] + direction_c, direction_c):
                trajectory.append((trajectory[-1][0], c))

        trajectories.append(trajectory)

    print("Initial Trajectories:")
    for i, traj in enumerate(trajectories):
        print(f"  Agent {i+1}: {traj}")
    print()

    # Detect collisions
    collisions = analyze_collisions(trajectories, agent_goals, grid, verbose=False)
    print(f"Collisions detected: {len(collisions)}")
    for coll in collisions:
        print(f"  T={coll['time']}, Type={coll['type']}, Agents={list(coll['agents'])}, Cell={coll['cell']}")
    print()

    if not collisions:
        print("No collisions to fix!")
        return

    # Pick first collision
    collision = collisions[0]
    print(f"Testing Joint A* on: {collision}")
    print()

    # Precompute heuristics
    heuristic_dist_map = compute_heuristic_distances(grid, agent_goals)

    # Run joint A*
    print("=" * 80)
    print("Running Joint A* Planning...")
    print("=" * 80)
    print()

    success, joint_plans, joint_trajs, t_start, t_goal_sub = try_joint_astar_planning(
        collision,
        trajectories,
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
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()

    print(f"Success: {success}")

    if success:
        print(f"Planning window: t_start={t_start}, t_goal_sub={t_goal_sub}")
        print()

        print("Joint Plans (action sequences):")
        action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'WAIT'}
        for aid in sorted(joint_plans.keys()):
            plan = joint_plans[aid]
            action_str = ' '.join([action_names.get(a, str(a)) for a in plan])
            print(f"  Agent {aid}: {action_str}")

        print()
        print("Joint Trajectories:")
        for aid in sorted(joint_trajs.keys()):
            traj = joint_trajs[aid]
            print(f"  Agent {aid}: {traj}")

        print()

        # Verify
        test_trajs = list(trajectories)
        for aid, traj in joint_trajs.items():
            test_trajs[aid - 1] = traj

        new_collisions = analyze_collisions(test_trajs, agent_goals, grid, verbose=False)
        print(f"Collisions in solution: {len(new_collisions)}")
        if new_collisions:
            for coll in new_collisions:
                print(f"  T={coll['time']}, Type={coll['type']}, Agents={list(coll['agents'])}")
            print("\n❌ Solution has collisions!")
        else:
            print("\n✅ No collisions! Solution is valid!")
    else:
        print("❌ Failed to find solution")


if __name__ == "__main__":
    main()
