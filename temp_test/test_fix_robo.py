"""
Test fix_collisions on robo_test.txt directly.
This tests the full collision resolution pipeline including joint A*.
"""

import numpy as np
from fix import fix_collisions
from utils.env_utils import analyze_collisions


class SimpleEnv:
    """Simple environment wrapper for testing"""
    def __init__(self, grid, agent_pos, goal_pos):
        self.grid = grid
        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
        self.dynamic_info = []
        self.num_dynamic_obstacles = 0

    def current_observation(self):
        """Return observation in the format expected by plan_with_search"""
        rows, cols = self.grid.shape

        # Create direction vector (normalized vector towards goal)
        dr = self.goal_pos[0] - self.agent_pos[0]
        dc = self.goal_pos[1] - self.agent_pos[1]
        dist = max(abs(dr), abs(dc))
        if dist > 0:
            direction = (dr / dist, dc / dist)
        else:
            direction = (0, 0)

        # Create distance array (Manhattan distance from agent to each cell)
        distance = np.zeros((rows, cols), dtype=float)
        for r in range(rows):
            for c in range(cols):
                distance[r, c] = abs(r - self.agent_pos[0]) + abs(c - self.agent_pos[1])

        return {
            'grid': self.grid.astype(float),
            'direction': np.array(direction),
            'distance': distance
        }


class EnvWrapper:
    """Wrapper to match the structure expected by fix_collisions"""
    def __init__(self, grid, agent_pos, goal_pos):
        self.env = SimpleEnv(grid, agent_pos, goal_pos)

    def current_observation(self):
        """Forward to env"""
        return self.env.current_observation()


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


def create_initial_trajectories(agents, grid):
    """Create simple straight-line trajectories that cause collisions"""
    trajectories = []

    for agent in agents:
        start = agent['start']
        goal = agent['goal']

        trajectory = [start]

        # Move horizontally or vertically (simple paths)
        if start[0] == goal[0]:  # Same row
            direction = 1 if goal[1] > start[1] else -1
            for c in range(start[1] + direction, goal[1] + direction, direction):
                trajectory.append((start[0], c))
        elif start[1] == goal[1]:  # Same column
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

        trajectories.append(trajectory)

    return trajectories


def main():
    print("=" * 80)
    print("Testing fix_collisions on robo_test.txt")
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
    initial_trajectories = create_initial_trajectories(agents, grid)
    print("Initial Trajectories:")
    for i, traj in enumerate(initial_trajectories):
        print(f"  Agent {i+1}: {traj}")
    print()

    # Detect initial collisions
    agent_goals = [agent['goal'] for agent in agents]
    initial_collisions = analyze_collisions(initial_trajectories, agent_goals, grid, verbose=False)
    print(f"Initial collisions: {len(initial_collisions)}")
    for coll in initial_collisions:
        print(f"  T={coll['time']}, Type={coll['type']}, Agents={list(coll['agents'])}, Cell={coll['cell']}")
    print()

    if not initial_collisions:
        print("No collisions detected. Nothing to fix.")
        return

    # Create environment wrappers
    agent_envs = []
    for i, agent in enumerate(agents):
        env_wrapper = EnvWrapper(grid, agent['start'], agent['goal'])
        agent_envs.append(env_wrapper)

    # Create initial plans (dummy - just using the trajectories as reference)
    initial_plans = []
    for traj in initial_trajectories:
        # Convert trajectory to actions (simple approach)
        plan = []
        for i in range(len(traj) - 1):
            curr = traj[i]
            next_pos = traj[i + 1]
            dr, dc = next_pos[0] - curr[0], next_pos[1] - curr[1]

            # Map to action
            if dr == -1:
                plan.append(0)  # UP
            elif dr == 1:
                plan.append(1)  # DOWN
            elif dc == -1:
                plan.append(2)  # LEFT
            elif dc == 1:
                plan.append(3)  # RIGHT
            else:
                plan.append(4)  # WAIT

        initial_plans.append(plan)

    print("Initial Plans:")
    for i, plan in enumerate(initial_plans):
        action_names = {0: 'U', 1: 'D', 2: 'L', 3: 'R', 4: 'W'}
        action_str = ''.join([action_names.get(a, str(a)) for a in plan])
        print(f"  Agent {i+1}: {action_str} ({len(plan)} actions)")
    print()

    # Run collision fix
    print("=" * 80)
    print("Running fix_collisions...")
    print("=" * 80)
    print()

    fixed_plans, fixed_trajectories, num_remaining_collisions, timed_out, log_data = fix_collisions(
        initial_plans,
        initial_trajectories,
        agent_envs,
        model=None,
        run_counters={},
        device='cpu',
        replan_strategy="best",
        info_setting="all",
        search_type="astar",
        algo="dqn",
        timeout=5.0,
        heuristic_weight=1.0,
        max_expansions=500,
        time_limit=60,
        max_passes=20,
        verbose=True
    )

    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()

    print(f"Time: {log_data['time']:.2f}s")
    print(f"Passes: {log_data['passes']}")
    print(f"Final collisions: {log_data['final_collisions']}")
    print(f"Timed out: {timed_out}")
    print()

    print("Fixed Plans:")
    for i, plan in enumerate(fixed_plans):
        action_names = {0: 'U', 1: 'D', 2: 'L', 3: 'R', 4: 'W'}
        action_str = ''.join([action_names.get(a, str(a)) for a in plan])
        print(f"  Agent {i+1}: {action_str} ({len(plan)} actions)")
    print()

    print("Fixed Trajectories:")
    for i, traj in enumerate(fixed_trajectories):
        print(f"  Agent {i+1}: {traj}")
    print()

    # Verify final collisions
    final_collisions = analyze_collisions(fixed_trajectories, agent_goals, grid, verbose=False)
    print(f"Final collision count (detected): {len(final_collisions)}")
    for coll in final_collisions:
        print(f"  T={coll['time']}, Type={coll['type']}, Agents={list(coll['agents'])}, Cell={coll['cell']}")

    if len(final_collisions) == 0:
        print("\n✅ SUCCESS: All collisions resolved!")
    else:
        print(f"\n⚠️  {len(final_collisions)} collisions remain")

    print()
    print("Information Sharing Metrics:")
    for key, value in log_data['info_sharing'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
